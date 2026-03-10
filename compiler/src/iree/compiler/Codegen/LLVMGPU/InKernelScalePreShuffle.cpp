// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Packs scale bytes across M/N intrinsic positions for amdgpu.scaled_mfma
// operations. Two strategies:
//
// 1. Packed LDS copy (Variant C): Rewrites the cooperative scale copy's
//    write addresses so scales arrive in LDS already packed per-lane.
//    The compute phase then reads scales with a single ds_read_b32 instead
//    of 4 scattered ds_read_u8. No extra barrier needed.
//
// 2. Register-only packing (fallback): Collects 4 unique bytes from
//    different M/N positions after they've been loaded individually,
//    packs them into a single vector<4xf8E8M0FNU>, and sets scalesIdx.
//
// When multiple K-tiles are present (from loop unrolling), ops are
// partitioned by K-tile depth and packed independently per K-tile.

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_INKERNELSCALEPRESHUFFLEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

static Value extractScaleByte(Value paddedScale) {
  auto insertOp = paddedScale.getDefiningOp<vector::InsertOp>();
  if (!insertOp)
    return nullptr;
  auto pos = insertOp.getStaticPosition();
  if (pos.size() != 1 || pos[0] != 0)
    return nullptr;
  return insertOp.getValueToStore();
}

static void moveDefChainBefore(Operation *op, Operation *target) {
  if (!op || op == target || op->getBlock() != target->getBlock())
    return;
  if (op->isBeforeInBlock(target))
    return;
  for (Value operand : op->getOperands()) {
    if (auto *defOp = operand.getDefiningOp())
      moveDefChainBefore(defOp, target);
  }
  op->moveBefore(target);
}

static int getAccDepth(amdgpu::ScaledMFMAOp op,
                       DenseMap<Operation *, int> &cache) {
  auto it = cache.find(op.getOperation());
  if (it != cache.end())
    return it->second;

  int depth = 0;
  Value acc = op.getDestC();
  while (auto prevMfma = acc.getDefiningOp<amdgpu::ScaledMFMAOp>()) {
    depth = getAccDepth(prevMfma, cache) + 1;
    break;
  }

  cache[op.getOperation()] = depth;
  return depth;
}

//===----------------------------------------------------------------------===//
// Packed LDS copy (Variant C)
//===----------------------------------------------------------------------===//

// Result of tracing a scale byte back to its workgroup-memory source.
struct ScaleByteOrigin {
  vector::TransferReadOp readOp;
  memref::AllocOp allocOp;
  SmallVector<int64_t> extractPos;
};

// Follow the def chain: byte → vector.extract → vector.transfer_read from
// workgroup memref → (expand_shape)* → memref.alloc.
static std::optional<ScaleByteOrigin> traceByteToLDS(Value byte) {
  auto extractOp = byte.getDefiningOp<vector::ExtractOp>();
  if (!extractOp)
    return std::nullopt;

  Value scaleVec = extractOp.getSource();
  auto readOp = scaleVec.getDefiningOp<vector::TransferReadOp>();
  if (!readOp)
    return std::nullopt;

  Value srcMemref = readOp.getBase();
  auto memrefType = dyn_cast<MemRefType>(srcMemref.getType());
  if (!memrefType)
    return std::nullopt;

  auto addrSpace =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
  if (!addrSpace || addrSpace.getValue() != gpu::AddressSpace::Workgroup)
    return std::nullopt;

  Value current = srcMemref;
  while (auto expandOp = current.getDefiningOp<memref::ExpandShapeOp>())
    current = expandOp.getSrc();

  auto allocOp = current.getDefiningOp<memref::AllocOp>();
  if (!allocOp)
    return std::nullopt;

  if (allocOp.getType().getRank() != 2)
    return std::nullopt;

  return ScaleByteOrigin{
      readOp, allocOp,
      SmallVector<int64_t>(extractOp.getStaticPosition())};
}

// Controls which bytes are grouped into each 4-byte packed read.
//   MFirst:  [MI0,MI1,MI2,MI3] per K-group per MI-group  (scalesIdx = mi%4)
//            When numMI > 4, MI positions are split into groups of 4.
//   KxM:     [MI0-K0,MI0-K1,MI1-K0,MI1-K1]  (scalesIdx = mi%2*numK+kg)
enum class PackMode { MFirst, KxM };

// Change this constant to switch packing strategy.
static constexpr PackMode kPackMode = PackMode::MFirst;

struct MFMAPackInfo {
  amdgpu::ScaledMFMAOp mfmaOp;
  int64_t kGroup;
  int64_t miInPack;
};

struct PackedAllocState {
  memref::AllocOp allocOp;
  vector::TransferReadOp readOp;
  int64_t allocRows;
  int64_t allocCols;
  int64_t numMI;
  int64_t numKGroups;
  int64_t numSG;
  int64_t kGroupDimIdx;
  bool isLHS;
  SmallVector<MFMAPackInfo> mfmaInfos;

  int64_t miGroupCount() const { return (numMI + 3) / 4; }

  int64_t numPacks() const {
    if (kPackMode == PackMode::KxM)
      return (numMI + 1) / 2;
    return numKGroups * miGroupCount();
  }

  int64_t packIndex(int64_t mi, int64_t kg) const {
    if (kPackMode == PackMode::KxM)
      return mi / 2;
    return kg * miGroupCount() + mi / 4;
  }

  int64_t byteInPack(int64_t mi, int64_t kg) const {
    if (kPackMode == PackMode::KxM)
      return (mi % 2) * numKGroups + kg;
    return mi % 4;
  }
};

// Collect all vector.transfer_write ops that target the given alloc
// (possibly through subviews).
static SmallVector<vector::TransferWriteOp>
findCopyWritesToAlloc(memref::AllocOp allocOp) {
  SmallVector<vector::TransferWriteOp> writes;
  SmallVector<Value> worklist{allocOp.getResult()};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;
    for (OpOperand &use : val.getUses()) {
      Operation *user = use.getOwner();
      if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user))
        writes.push_back(writeOp);
      else if (auto subviewOp = dyn_cast<memref::SubViewOp>(user))
        worklist.push_back(subviewOp.getResult());
    }
  }
  return writes;
}

// Resolve the (row, col_start) position of a copy write in the alloc's 2D
// coordinate space.  Requires: write → subview(alloc) with zero write-indices.
static std::optional<std::pair<Value, Value>>
getCopyWriteBaseInAlloc(vector::TransferWriteOp writeOp,
                        memref::AllocOp allocOp) {
  Value dest = writeOp.getBase();
  auto subviewOp = dest.getDefiningOp<memref::SubViewOp>();
  if (!subviewOp || subviewOp.getSource() != allocOp.getResult())
    return std::nullopt;

  for (Value idx : writeOp.getIndices()) {
    auto cst = idx.getDefiningOp<arith::ConstantIndexOp>();
    if (!cst || cst.value() != 0)
      return std::nullopt;
  }

  auto offsets = subviewOp.getMixedOffsets();
  if (offsets.size() != 2)
    return std::nullopt;

  auto toValue = [&](OpFoldResult ofr) -> Value {
    if (auto val = dyn_cast<Value>(ofr))
      return val;
    OpBuilder b(writeOp);
    return arith::ConstantIndexOp::create(
        b, writeOp.getLoc(),
        cast<IntegerAttr>(cast<Attribute>(ofr)).getInt());
  };

  return std::make_pair(toValue(offsets[0]), toValue(offsets[1]));
}

// Rewrite one cooperative-copy transfer_write into per-byte stores at
// packed LDS offsets.
//
// MFirst layout (numMI > 4, must be multiple of 4):
//   packed[(k_group * miGrpCount + mi_group) * numSG*256 + sg*256
//          + lane*4 + mi_in_group]
//   where mi_group = mi_in_sg / 4, mi_in_group = mi_in_sg % 4,
//         miGrpCount = ceil(numMI / 4).
// MFirst layout (numMI <= 4):  collapses to the above with miGrpCount=1.
// KxM layout:    packed[mi_pair * numSG*256 + sg*256 + lane*4 + (mi%2)*numK+kg]
//   Common: sg = mi / numMI, mi_in_sg = mi % numMI,
//           lane = m_thread + k_thread * 16,
//           mi = row / 16, m_thread = row % 16,
//           k_group = col / 4, k_thread = col % 4.
static void rewriteCopyWrite(vector::TransferWriteOp writeOp, Value row,
                             Value colStart, Value packedBuf, int64_t numSG,
                             int64_t numMI, int64_t numKGroups,
                             bool keepOriginalWrite) {
  OpBuilder b(writeOp);
  Location loc = writeOp.getLoc();

  Value vec = writeOp.getValueToStore();
  auto vecType = cast<VectorType>(vec.getType());
  int64_t numBytes = vecType.getNumElements();

  if (vecType.getRank() > 1) {
    auto flatType = VectorType::get({numBytes}, vecType.getElementType());
    vec = vector::ShapeCastOp::create(b, loc, flatType, vec);
  }

  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  Value c256 = arith::ConstantIndexOp::create(b, loc, 256);
  Value cNumMI = arith::ConstantIndexOp::create(b, loc, numMI);
  Value cNumSGx256 = arith::ConstantIndexOp::create(b, loc, numSG * 256);

  Value mi = arith::DivUIOp::create(b, loc, row, c16);
  Value mThread = arith::RemUIOp::create(b, loc, row, c16);
  Value sg = arith::DivUIOp::create(b, loc, mi, cNumMI);
  Value miInSg = arith::RemUIOp::create(b, loc, mi, cNumMI);
  Value sgOff = arith::MulIOp::create(b, loc, sg, c256);

  int64_t miGrpCount = (numMI + 3) / 4;

  for (int64_t i = 0; i < numBytes; i++) {
    Value byteVal = vector::ExtractOp::create(b, loc, vec, i);

    // Compute per-byte column, K-group, and K-thread.
    int64_t col_i = i; // colStart is always 0 for cooperative copy subviews.
    int64_t kGroup_i = col_i / 4;
    int64_t kThread_i = col_i % 4;

    Value lane = arith::AddIOp::create(
        b, loc, mThread,
        arith::ConstantIndexOp::create(b, loc, kThread_i * 16));
    Value laneOff = arith::MulIOp::create(b, loc, lane, c4);

    Value packDimOff;
    Value byteInPack;
    if (kPackMode == PackMode::KxM) {
      Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
      Value cNumK = arith::ConstantIndexOp::create(b, loc, numKGroups);
      Value miPair = arith::DivUIOp::create(b, loc, miInSg, c2);
      Value miMod2 = arith::RemUIOp::create(b, loc, miInSg, c2);
      packDimOff = arith::MulIOp::create(b, loc, miPair, cNumSGx256);
      byteInPack = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, miMod2, cNumK),
          arith::ConstantIndexOp::create(b, loc, kGroup_i));
    } else {
      Value miGroup = arith::DivUIOp::create(b, loc, miInSg, c4);
      Value miInGroup = arith::RemUIOp::create(b, loc, miInSg, c4);
      Value cKGroup = arith::ConstantIndexOp::create(b, loc, kGroup_i);
      Value cMiGrpCount = arith::ConstantIndexOp::create(b, loc, miGrpCount);
      Value packIdx = arith::AddIOp::create(
          b, loc, arith::MulIOp::create(b, loc, cKGroup, cMiGrpCount),
          miGroup);
      packDimOff = arith::MulIOp::create(b, loc, packIdx, cNumSGx256);
      byteInPack = miInGroup;
    }

    Value off = arith::AddIOp::create(
        b, loc,
        arith::AddIOp::create(
            b, loc, arith::AddIOp::create(b, loc, packDimOff, sgOff),
            byteInPack),
        laneOff);
    auto scalarVec = vector::BroadcastOp::create(
        b, loc, VectorType::get({1}, vecType.getElementType()), byteVal);
    vector::TransferWriteOp::create(b, loc, scalarVec, packedBuf,
                                    ValueRange{off}, ArrayRef<bool>{true});
  }

  if (!keepOriginalWrite)
    writeOp.erase();
}

// Create packed reads from the packed LDS buffer.
// Returns one vector<4xf8E8M0FNU> per "pack" (number depends on pack mode).
static SmallVector<Value>
createPackedReads(vector::TransferReadOp readOp, Value packedBuf,
                  int64_t numSG, int64_t numPacks, int64_t numMI) {
  OpBuilder b(readOp);
  Location loc = readOp.getLoc();

  auto readIndices = readOp.getIndices();
  Value sgBase = readIndices[0];
  Value laneM = readIndices[1];
  Value laneK = readIndices[3];

  Value c4 = arith::ConstantIndexOp::create(b, loc, 4);
  Value c16 = arith::ConstantIndexOp::create(b, loc, 16);
  Value c256 = arith::ConstantIndexOp::create(b, loc, 256);
  Value cNumMI = arith::ConstantIndexOp::create(b, loc, numMI);

  Value laneId = arith::AddIOp::create(
      b, loc, laneM, arith::MulIOp::create(b, loc, laneK, c16));
  Value sgIndex = arith::DivUIOp::create(b, loc, sgBase, cNumMI);
  Value sgOff = arith::MulIOp::create(b, loc, sgIndex, c256);
  Value laneOff = arith::MulIOp::create(b, loc, laneId, c4);
  Value baseOff = arith::AddIOp::create(b, loc, sgOff, laneOff);

  auto elemType = cast<MemRefType>(packedBuf.getType()).getElementType();
  auto vecType = VectorType::get({4}, elemType);

  Value padding = readOp.getPadding();

  int64_t packStride = numSG * 256;
  SmallVector<Value> packedVecs;
  for (int64_t p = 0; p < numPacks; p++) {
    Value readOff;
    if (p == 0) {
      readOff = baseOff;
    } else {
      Value pBase = arith::ConstantIndexOp::create(b, loc, p * packStride);
      readOff = arith::AddIOp::create(b, loc, baseOff, pBase);
    }
    Value packed = vector::TransferReadOp::create(
        b, loc, vecType, packedBuf, ValueRange{readOff}, padding,
        ArrayRef<bool>{true});
    packedVecs.push_back(packed);
  }

  return packedVecs;
}

// Attempt packed LDS copy rewrite for one group of MFMAs sharing the same
// alloc and operand side.  Returns true on success.
static bool
rewritePackedCopyForOperand(PackedAllocState &state,
                            DenseMap<Operation *, Value> &allocToPackedBuf) {
  memref::AllocOp allocOp = state.allocOp;

  Value packedBuf;
  bool needCopyRewrite = false;
  auto bufIt = allocToPackedBuf.find(allocOp.getOperation());
  if (bufIt != allocToPackedBuf.end()) {
    packedBuf = bufIt->second;
  } else {
    needCopyRewrite = true;
    OpBuilder b(allocOp);
    Location loc = allocOp.getLoc();
    int64_t packedSize = state.numPacks() * state.numSG * 256;
    auto packedType = MemRefType::get(
        {packedSize}, allocOp.getType().getElementType(),
        /*layout=*/nullptr,
        gpu::AddressSpaceAttr::get(b.getContext(),
                                   gpu::AddressSpace::Workgroup));
    packedBuf = memref::AllocOp::create(b, loc, packedType).getResult();
    allocToPackedBuf[allocOp.getOperation()] = packedBuf;
  }

  if (needCopyRewrite) {
    auto copyWrites = findCopyWritesToAlloc(allocOp);
    if (copyWrites.empty())
      return false;
    for (auto writeOp : copyWrites) {
      auto pos = getCopyWriteBaseInAlloc(writeOp, allocOp);
      if (!pos)
        return false;
      bool keepOrigWrite = state.numMI > 4;
      rewriteCopyWrite(writeOp, pos->first, pos->second, packedBuf,
                       state.numSG, state.numMI, state.numKGroups,
                       keepOrigWrite);
    }
  }

  auto packedVecs = createPackedReads(state.readOp, packedBuf, state.numSG,
                                      state.numPacks(), state.numMI);

  // Collect the old insert/extract ops for cleanup before rewriting MFMAs.
  SetVector<Operation *> deadInserts;
  for (auto &info : state.mfmaInfos) {
    Value oldScale = state.isLHS ? info.mfmaOp.getScalesA()
                                 : info.mfmaOp.getScalesB();
    if (auto insertOp = oldScale.getDefiningOp<vector::InsertOp>())
      deadInserts.insert(insertOp);

    int64_t pi = state.packIndex(info.miInPack, info.kGroup);
    int64_t bi = state.byteInPack(info.miInPack, info.kGroup);
    Value packedVec = packedVecs[pi];
    uint32_t idx = static_cast<uint32_t>(bi);
    if (state.isLHS) {
      info.mfmaOp.getScalesAMutable().assign(packedVec);
      info.mfmaOp.setScalesIdxA(idx);
    } else {
      info.mfmaOp.getScalesBMutable().assign(packedVec);
      info.mfmaOp.setScalesIdxB(idx);
    }
  }

  // Erase dead insert → extract → transfer_read chain bottom-up.
  for (Operation *op : deadInserts) {
    if (!op->use_empty())
      continue;
    auto insertOp = cast<vector::InsertOp>(op);
    Value byte = insertOp.getValueToStore();
    op->erase();
    if (auto extractOp = byte.getDefiningOp<vector::ExtractOp>()) {
      if (extractOp->use_empty())
        extractOp->erase();
    }
  }
  if (state.readOp->use_empty())
    state.readOp->erase();

  return true;
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct InKernelScalePreShufflePass final
    : impl::InKernelScalePreShufflePassBase<InKernelScalePreShufflePass> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    SmallVector<amdgpu::ScaledMFMAOp> allOps;
    funcOp.walk(
        [&](amdgpu::ScaledMFMAOp op) { allOps.push_back(op); });

    if (allOps.empty())
      return;

    DenseSet<std::pair<Operation *, int>> handledPairs;
    tryPackedCopy(allOps, handledPairs);
    doRegisterPacking(allOps, handledPairs);
  }

  // ---- Variant C: packed LDS copy ----

  void tryPackedCopy(
      SmallVector<amdgpu::ScaledMFMAOp> &allOps,
      DenseSet<std::pair<Operation *, int>> &handledPairs) {

    // Group MFMAs by (allocOp, isLHS).
    using GroupKey = std::pair<Operation *, int>;
    DenseMap<GroupKey, PackedAllocState> groups;

    for (auto mfmaOp : allOps) {
      for (bool isLHS : {true, false}) {
        Value scaleVec = isLHS ? mfmaOp.getScalesA() : mfmaOp.getScalesB();
        auto scaleVecType = dyn_cast<VectorType>(scaleVec.getType());
        if (!scaleVecType || scaleVecType.getNumElements() != 4)
          continue;
        uint32_t existingIdx =
            isLHS ? mfmaOp.getScalesIdxA() : mfmaOp.getScalesIdxB();
        if (existingIdx != 0)
          continue;

        Value byte = extractScaleByte(scaleVec);
        if (!byte)
          continue;

        auto origin = traceByteToLDS(byte);
        if (!origin)
          continue;

        auto readVecType =
            cast<VectorType>(origin->readOp.getResult().getType());
        auto shape = readVecType.getShape();
        int64_t numMI = shape[0];
        if (numMI < 2 || (numMI > 4 && numMI % 4 != 0))
          continue;

        int64_t kGroupDimIdx = -1;
        int64_t numKGroups = 1;
        for (int64_t i = 1; i < static_cast<int64_t>(shape.size()); i++) {
          if (shape[i] > 1) {
            numKGroups = shape[i];
            kGroupDimIdx = i;
            break;
          }
        }
        if (kGroupDimIdx == -1)
          kGroupDimIdx = (shape.size() > 2) ? 2 : 1;

        auto allocType = origin->allocOp.getType();
        int64_t allocRows = allocType.getShape()[0];
        int64_t allocCols = allocType.getShape()[1];
        int64_t numSG = allocRows / (16 * numMI);
        if (numSG < 1)
          numSG = 1;

        GroupKey key{origin->allocOp.getOperation(), isLHS ? 1 : 0};
        auto &state = groups[key];
        if (state.mfmaInfos.empty()) {
          state.allocOp = origin->allocOp;
          state.readOp = origin->readOp;
          state.isLHS = isLHS;
          state.allocRows = allocRows;
          state.allocCols = allocCols;
          state.numMI = numMI;
          state.numKGroups = numKGroups;
          state.numSG = numSG;
          state.kGroupDimIdx = kGroupDimIdx;
        }

        int64_t kGroup = 0;
        if (kGroupDimIdx >= 0 &&
            kGroupDimIdx < static_cast<int64_t>(origin->extractPos.size()))
          kGroup = origin->extractPos[kGroupDimIdx];
        int64_t miInPack = origin->extractPos[0];

        state.mfmaInfos.push_back(MFMAPackInfo{mfmaOp, kGroup, miInPack});
      }
    }

    if (groups.empty())
      return;

    DenseMap<Operation *, Value> allocToPackedBuf;

    for (auto &[key, state] : groups) {
      if (rewritePackedCopyForOperand(state, allocToPackedBuf)) {
        for (auto &info : state.mfmaInfos)
          handledPairs.insert({info.mfmaOp.getOperation(), state.isLHS ? 1 : 0});
      }
    }

    // Clean up dead original allocs whose writes and reads were rewritten.
    for (auto &[allocOp, packedBuf] : allocToPackedBuf) {
      auto alloc = cast<memref::AllocOp>(allocOp);
      // Iteratively erase dead users (subviews, expand_shapes) bottom-up.
      bool changed = true;
      while (changed) {
        changed = false;
        for (OpOperand &use :
             llvm::make_early_inc_range(alloc.getResult().getUses())) {
          Operation *user = use.getOwner();
          if (!user->use_empty())
            continue;
          user->erase();
          changed = true;
        }
      }
      if (alloc->use_empty())
        alloc->erase();
    }
  }

  // ---- Register-only packing (fallback) ----

  void doRegisterPacking(
      SmallVector<amdgpu::ScaledMFMAOp> &allOps,
      const DenseSet<std::pair<Operation *, int>> &handledPairs) {

    DenseMap<Block *, SmallVector<amdgpu::ScaledMFMAOp>> blockToOps;
    for (auto op : allOps)
      blockToOps[op->getBlock()].push_back(op);

    for (auto &[block, ops] : blockToOps) {
      if (ops.size() < 2)
        continue;

      DenseMap<Operation *, int> depthCache;
      DenseMap<int, SmallVector<amdgpu::ScaledMFMAOp>> depthToOps;
      for (auto op : ops) {
        int depth = getAccDepth(op, depthCache);
        depthToOps[depth].push_back(op);
      }

      for (auto &[depth, kTileOps] : depthToOps) {
        if (kTileOps.size() < 2)
          continue;
        bool anyUnhandledLHS = false, anyUnhandledRHS = false;
        for (auto op : kTileOps) {
          if (!handledPairs.count({op.getOperation(), 1}))
            anyUnhandledLHS = true;
          if (!handledPairs.count({op.getOperation(), 0}))
            anyUnhandledRHS = true;
        }
        if (anyUnhandledLHS)
          packScaleOperand(kTileOps, /*isLHS=*/true, handledPairs);
        if (anyUnhandledRHS)
          packScaleOperand(kTileOps, /*isLHS=*/false, handledPairs);
      }
    }
  }

  void packScaleOperand(
      SmallVector<amdgpu::ScaledMFMAOp> &ops, bool isLHS,
      const DenseSet<std::pair<Operation *, int>> &handledPairs) {
    SmallVector<amdgpu::ScaledMFMAOp> unhandled;
    for (auto op : ops) {
      if (!handledPairs.count({op.getOperation(), isLHS ? 1 : 0}))
        unhandled.push_back(op);
    }
    if (unhandled.size() < 2)
      return;

    SmallVector<Value> uniqueBytes;
    DenseMap<Value, int64_t> byteToIdx;
    SmallVector<int64_t> opGroup(unhandled.size());

    for (auto [i, op] : llvm::enumerate(unhandled)) {
      Value scaleVec = isLHS ? op.getScalesA() : op.getScalesB();

      auto scaleVecType = dyn_cast<VectorType>(scaleVec.getType());
      if (!scaleVecType || scaleVecType.getNumElements() != 4)
        return;

      uint32_t existingIdx =
          isLHS ? op.getScalesIdxA() : op.getScalesIdxB();
      if (existingIdx != 0)
        return;

      Value byte = extractScaleByte(scaleVec);
      if (!byte)
        return;

      auto [it, inserted] = byteToIdx.try_emplace(byte, uniqueBytes.size());
      if (inserted)
        uniqueBytes.push_back(byte);
      opGroup[i] = it->second;
    }

    int64_t numGroups = uniqueBytes.size();
    if (numGroups <= 1)
      return;
    // For > 4 unique bytes, require a multiple of 4 so we can split evenly.
    if (numGroups > 4 && numGroups % 4 != 0)
      return;

    Operation *firstMfma = unhandled.front().getOperation();
    for (Value byte : uniqueBytes)
      moveDefChainBefore(byte.getDefiningOp(), firstMfma);

    OpBuilder builder(firstMfma);
    Location loc = firstMfma->getLoc();
    Type elemType =
        cast<VectorType>(
            (isLHS ? unhandled.front().getScalesA()
                   : unhandled.front().getScalesB())
                .getType())
            .getElementType();
    auto vecType = VectorType::get({4}, elemType);

    // Build one packed vector per group of 4 bytes.
    int64_t numPackedVecs = (numGroups + 3) / 4;
    SmallVector<Value> packedVecs(numPackedVecs);
    for (int64_t v = 0; v < numPackedVecs; v++) {
      Value packed = arith::ConstantOp::create(
          builder, loc,
          SplatElementsAttr::get(
              vecType,
              APFloat::getSmallest(
                  cast<FloatType>(elemType).getFloatSemantics())));
      int64_t groupStart = v * 4;
      int64_t groupEnd = std::min(groupStart + 4, numGroups);
      for (int64_t i = groupStart; i < groupEnd; i++)
        packed = vector::InsertOp::create(builder, loc, uniqueBytes[i],
                                          packed, i - groupStart);
      packedVecs[v] = packed;
    }

    for (auto [i, op] : llvm::enumerate(unhandled)) {
      int64_t byteIdx = opGroup[i];
      int64_t vecIdx = byteIdx / 4;
      uint32_t idx = static_cast<uint32_t>(byteIdx % 4);
      if (isLHS) {
        op.getScalesAMutable().assign(packedVecs[vecIdx]);
        op.setScalesIdxA(idx);
      } else {
        op.getScalesBMutable().assign(packedVecs[vecIdx]);
        op.setScalesIdxB(idx);
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
