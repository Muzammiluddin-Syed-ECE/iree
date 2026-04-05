// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_RESOLVESWIZZLEHINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
struct ResolveSwizzleHintsPass final
    : impl::ResolveSwizzleHintsPassBase<ResolveSwizzleHintsPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

static Value createOrFoldNewStaticAdd(RewriterBase &rewriter, Value v,
                                      int64_t offset) {
  // Early exit for the common offset = 0 case.
  if (offset == 0) {
    return v;
  }

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    llvm::APInt constant;
    if (matchPattern(add.getRhs(), m_ConstantInt(&constant))) {
      Value combined = arith::ConstantIndexOp::create(
          rewriter, add.getLoc(), offset + constant.getSExtValue());
      return arith::AddIOp::create(rewriter, add.getLoc(), add.getLhs(),
                                   combined, add.getOverflowFlags());
    }
  }
  Value offsetVal =
      arith::ConstantIndexOp::create(rewriter, v.getLoc(), offset);
  return arith::AddIOp::create(rewriter, v.getLoc(), v, offsetVal);
}

/// Swizzles vector.load(iree_codegen.swizzle_hint, offset). The
/// SwizzleInterfaceAttr exposes two methods:
///   1. getAccessElementCount -> int64_t
///        - Gives the number of contiguous elements in the swizzling pattern.
///   2. swizzleOffset(src: memref<N x !eltype>, offset: index) -> index
///        - Swizzles the |offset| into |src|, returning the new offset.
///
/// For a 1-d load of type `vector<C x !eltype>`, the load is unrolled into
/// loads of size `k = getAccessElementCount()` and individually swizzled.
///
/// For example, if `C = 16` and `k = 4`, this produces:
///
/// %0 = vector.load %src[swizzleOffset(%src, %offset)] : vector<4>
/// %1 = vector.load %src[swizzleOffset(%src, %offset + 4)] : vector<4>
/// %2 = vector.load %src[swizzleOffset(%src, %offset + 8)] : vector<4>
/// %3 = vector.load %src[swizzleOffset(%src, %offset + 12)] : vector<4>
/// %load = concat[%0, %1, %2, %3] : vector<16>
static void swizzleLoad(RewriterBase &rewriter, vector::LoadOp load,
                        IREE::Codegen::SwizzleHintOp hintOp) {
  Location hintLoc = hintOp.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType type = load.getVectorType();
  int64_t loadWidth = type.getShape()[0];
  Value memrefOffset = load.getIndices()[0];
  VectorType swizzledLoadType =
      VectorType::get({accessWidth}, type.getElementType());

  // ~ vector.undef, overwritten by unrolling.
  Value replacement = arith::ConstantOp::create(rewriter, hintLoc, type,
                                                rewriter.getZeroAttr(type));

  // Load type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < loadWidth; i += accessWidth) {
    Value newBaseOffset = createOrFoldNewStaticAdd(rewriter, memrefOffset, i);
    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    auto subLoad = vector::LoadOp::create(
        rewriter, load.getLoc(), swizzledLoadType, load.getBase(), newOffset);

    replacement = vector::InsertStridedSliceOp::create(
        rewriter, load.getLoc(), subLoad, replacement, ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{1});
  }
  rewriter.replaceOp(load, replacement);
}

/// Swizzles vector.store(iree_codegen.swizzle_hint, offset).
///
/// For a 1-d store of type `vector<C x !eltype>`, the store is unrolled into
/// stores of size `k = getAccessElementCount()` and individually swizzled.
///
/// For example, if `C = 16` and `k = 4`, this produces:
///
/// %0, %1, %2, %3 = split[%value_to_store] : vector<16>
/// vector.store %0, %src[swizzleOffset(%src, %offset)] : vector<4>
/// vector.store %1, %src[swizzleOffset(%src, %offset + 4)] : vector<4>
/// vector.store %2, %src[swizzleOffset(%src, %offset + 8)] : vector<4>
/// vector.store %3, %src[swizzleOffset(%src, %offset + 12)] : vector<4>
static void swizzleStore(RewriterBase &rewriter, vector::StoreOp store,
                         IREE::Codegen::SwizzleHintOp hintOp) {
  Location hintLoc = hintOp.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType type = store.getVectorType();
  int64_t storeWidth = type.getShape()[0];
  Value memrefOffset = store.getIndices()[0];

  // Store type = vector<C>, k = accessWidth
  // i = 0 -> C += k is the offset into the vector of a contiguous group of
  // swizzled elements.
  for (int64_t i = 0; i < storeWidth; i += accessWidth) {
    Value subVec = vector::ExtractStridedSliceOp::create(
        rewriter, store.getLoc(), store.getValueToStore(), ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{accessWidth}, ArrayRef<int64_t>{1});
    Value newBaseOffset = createOrFoldNewStaticAdd(rewriter, memrefOffset, i);

    Value newOffset = getValueOrCreateConstantIndexOp(
        rewriter, hintLoc,
        hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(),
                                          newBaseOffset, hintOp.getOperand()));
    vector::StoreOp::create(rewriter, store.getLoc(), subVec, store.getBase(),
                            newOffset);
  }
  rewriter.eraseOp(store);
}

/// Swizzles:
///  amdgpu.gather_to_lds
///    (iree_codegen.swizzle_hint, srcIndices, dst, dstIndices, transferType)
///
/// For now, only support gather_to_lds width == accessWidth.
static void swizzleGatherToLDS(RewriterBase &rewriter,
                               amdgpu::GatherToLDSOp gatherOp,
                               IREE::Codegen::SwizzleHintOp hintOp) {
  Location hintLoc = hintOp.getLoc();
  Value memrefOffset = gatherOp.getSrcIndices()[0];
  Value newOffset = getValueOrCreateConstantIndexOp(
      rewriter, hintLoc,
      hintOp.getSwizzle().swizzleOffset(rewriter, hintOp.getLoc(), memrefOffset,
                                        hintOp.getOperand()));
  rewriter.modifyOpInPlace(gatherOp, [&]() {
    gatherOp.getSrcMutable().assign(hintOp.getOperand());
    gatherOp.getSrcIndicesMutable().assign(newOffset);
  });
}

/// Transitively walk users of |startVal| through view-like ops
/// (expand_shape, collapse_shape, subview, cast) and scf.for iter_args to
/// collect all vector.load / vector.store operations that ultimately read/write
/// the swizzle-hinted memory. This enables swizzle resolution for loads that
/// are not direct users of the swizzle_hint due to intervening reshape or
/// loop-carried argument chains.
static void collectTransitiveSwizzleableUsers(
    Value startVal, int64_t accessWidth,
    SmallVectorImpl<vector::LoadOp> &loads,
    SmallVectorImpl<vector::StoreOp> &stores) {
  SmallVector<Value> worklist = {startVal};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (auto load = dyn_cast<vector::LoadOp>(user)) {
        VectorType vt = load.getVectorType();
        if (vt.getRank() == 1 && vt.getShape()[0] % accessWidth == 0)
          loads.push_back(load);
        continue;
      }
      if (auto store = dyn_cast<vector::StoreOp>(user)) {
        VectorType vt = store.getVectorType();
        if (vt.getRank() == 1 && vt.getShape()[0] % accessWidth == 0)
          stores.push_back(store);
        continue;
      }
      if (auto op = dyn_cast<memref::ExpandShapeOp>(user)) {
        worklist.push_back(op.getResult());
        continue;
      }
      if (auto op = dyn_cast<memref::CollapseShapeOp>(user)) {
        worklist.push_back(op.getResult());
        continue;
      }
      if (auto op = dyn_cast<memref::SubViewOp>(user)) {
        worklist.push_back(op.getResult());
        continue;
      }
      if (auto op = dyn_cast<memref::CastOp>(user)) {
        worklist.push_back(op.getResult());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [initArg, regionArg] :
             llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
          if (initArg == current)
            worklist.push_back(regionArg);
        }
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          for (auto [idx, yielded] :
               llvm::enumerate(yieldOp.getOperands())) {
            if (yielded == current)
              worklist.push_back(forOp.getResult(idx));
          }
        }
        continue;
      }
      // gather_to_lds, barriers, deallocs, etc.: don't trace through.
    }
  }
}

/// Compute the "linearization basis" from a memref's static strides.
/// For strides [s0, s1, ..., s_{n-1}] with s_{n-1} == 1, the basis is:
///   basis[i] = strides[i-1] / strides[i]   for i in [1, rank)
///   basis[0] = shape[0]                     (outer bound)
/// This basis satisfies: linearize(indices, basis) == sum(idx_i * stride_i).
static std::optional<SmallVector<int64_t>>
computeLinearizationBasis(MemRefType type) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)))
    return std::nullopt;
  int64_t rank = type.getRank();
  if (rank < 2 || strides.back() != 1)
    return std::nullopt;
  for (int64_t i = 0; i < rank - 1; ++i) {
    if (strides[i] == ShapedType::kDynamic ||
        strides[i + 1] == ShapedType::kDynamic)
      return std::nullopt;
    if (strides[i] % strides[i + 1] != 0)
      return std::nullopt;
  }
  SmallVector<int64_t> basis(rank);
  for (int64_t i = rank - 1; i >= 1; --i)
    basis[i] = strides[i - 1] / strides[i];
  basis[0] = type.getShape()[0];
  if (basis[0] == ShapedType::kDynamic)
    return std::nullopt;
  return basis;
}

/// Swizzle an N-D vector.load by:
///   1. Linearizing its indices to a flat element offset
///   2. Applying the XOR swizzle in accessWidth-sized chunks
///   3. Delinearizing the swizzled offset back to N-D indices
///   4. Replacing the original load with sub-loads at swizzled positions
static void swizzleNDLoad(RewriterBase &rewriter, vector::LoadOp load,
                          IREE::Codegen::SwizzleHintOp hintOp) {
  MemRefType memrefType = cast<MemRefType>(load.getBase().getType());
  if (memrefType.getRank() <= 1)
    return;

  auto maybeBasis = computeLinearizationBasis(memrefType);
  if (!maybeBasis)
    return;
  SmallVector<int64_t> basis = *maybeBasis;

  Location loc = load.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType loadType = load.getVectorType();
  int64_t loadWidth = loadType.getShape()[0];
  VectorType chunkType =
      VectorType::get({accessWidth}, loadType.getElementType());

  // Linearize the N-D indices to a flat element offset.
  Value flatOffset = affine::AffineLinearizeIndexOp::create(
      rewriter, loc, load.getIndices(), basis, /*disjoint=*/true);

  Value replacement = arith::ConstantOp::create(rewriter, loc, loadType,
                                                rewriter.getZeroAttr(loadType));

  for (int64_t i = 0; i < loadWidth; i += accessWidth) {
    Value chunkBase = createOrFoldNewStaticAdd(rewriter, flatOffset, i);

    Value swizzledFlat = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        hintOp.getSwizzle().swizzleOffset(rewriter, loc, chunkBase,
                                          hintOp.getOperand()));

    auto delinOp = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, swizzledFlat, basis, /*hasOuterBound=*/true);

    auto subLoad = vector::LoadOp::create(rewriter, loc, chunkType,
                                          load.getBase(),
                                          delinOp.getResults());
    replacement = vector::InsertStridedSliceOp::create(
        rewriter, loc, subLoad, replacement, ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{1});
  }
  rewriter.replaceOp(load, replacement);
}

/// Swizzle an N-D vector.store (symmetric to swizzleNDLoad).
static void swizzleNDStore(RewriterBase &rewriter, vector::StoreOp store,
                           IREE::Codegen::SwizzleHintOp hintOp) {
  MemRefType memrefType = cast<MemRefType>(store.getBase().getType());
  if (memrefType.getRank() <= 1)
    return;

  auto maybeBasis = computeLinearizationBasis(memrefType);
  if (!maybeBasis)
    return;
  SmallVector<int64_t> basis = *maybeBasis;

  Location loc = store.getLoc();
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  VectorType storeType = store.getVectorType();
  int64_t storeWidth = storeType.getShape()[0];

  Value flatOffset = affine::AffineLinearizeIndexOp::create(
      rewriter, loc, store.getIndices(), basis, /*disjoint=*/true);

  for (int64_t i = 0; i < storeWidth; i += accessWidth) {
    Value subVec = vector::ExtractStridedSliceOp::create(
        rewriter, loc, store.getValueToStore(), ArrayRef<int64_t>{i},
        ArrayRef<int64_t>{accessWidth}, ArrayRef<int64_t>{1});
    Value chunkBase = createOrFoldNewStaticAdd(rewriter, flatOffset, i);

    Value swizzledFlat = getValueOrCreateConstantIndexOp(
        rewriter, loc,
        hintOp.getSwizzle().swizzleOffset(rewriter, loc, chunkBase,
                                          hintOp.getOperand()));

    auto delinOp = affine::AffineDelinearizeIndexOp::create(
        rewriter, loc, swizzledFlat, basis, /*hasOuterBound=*/true);

    vector::StoreOp::create(rewriter, loc, subVec, store.getBase(),
                            delinOp.getResults());
  }
  rewriter.eraseOp(store);
}

static LogicalResult
verifyFlatContiguousSwizzleHintOp(IREE::Codegen::SwizzleHintOp hintOp) {
  auto memrefType = cast<MemRefType>(hintOp.getOperand().getType());
  if (memrefType.getRank() != 1) {
    hintOp.emitError()
        << "swizzle hint operand must be a rank-1 memref, got "
        << hintOp.getOperand().getType();
    return failure();
  }
  // Require unit stride (contiguous elements). Dynamic offset is acceptable
  // because the XOR swizzle is applied to relative element indices within the
  // buffer, not to the absolute address. This allows multi-buffered subviews
  // (memref<N, strided<[1], offset: ?>>) to carry swizzle hints.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(memrefType.getStridesAndOffset(strides, offset)) ||
      strides.size() != 1 || strides[0] != 1) {
    hintOp.emitError()
        << "swizzle hint operand must be a contiguous flat memref, got "
        << hintOp.getOperand().getType();
    return failure();
  }
  return success();
}

/// Resolves all hints. Walks all direct users and splits them into loads,
/// stores, and gather_to_lds ops. For view-like users (expand_shape,
/// collapse_shape), transitively traces through to find N-D loads/stores that
/// need swizzling (e.g., through scf.for iter_args in double-buffered loops).
/// If any direct user is not a supported type, bail out and silently drop the
/// optimization hint.
static void resolveHintOp(RewriterBase &rewriter,
                          IREE::Codegen::SwizzleHintOp hintOp) {
  SmallVector<vector::LoadOp> loads;
  SmallVector<vector::StoreOp> stores;
  SmallVector<amdgpu::GatherToLDSOp> gatherToLDSOps;
  SmallVector<vector::LoadOp> transitiveLoads;
  SmallVector<vector::StoreOp> transitiveStores;
  int64_t accessWidth = hintOp.getSwizzle().getAccessElementCount();
  for (Operation *user : hintOp->getUsers()) {
    if (auto load = dyn_cast<vector::LoadOp>(user)) {
      VectorType loadType = load.getVectorType();
      if (loadType.getRank() != 1 ||
          loadType.getShape()[0] % accessWidth != 0) {
        return;
      }
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<vector::StoreOp>(user)) {
      VectorType storeType = store.getVectorType();
      if (storeType.getRank() != 1 ||
          storeType.getShape()[0] % accessWidth != 0) {
        return;
      }
      stores.push_back(store);
      continue;
    }
    if (auto gatherToLDSOp = dyn_cast<amdgpu::GatherToLDSOp>(user)) {
      if (gatherToLDSOp.getDst() == hintOp) {
        continue;
      }
      int64_t accessBitWidth = cast<MemRefType>(hintOp.getOperand().getType())
                                   .getElementTypeBitWidth() *
                               accessWidth;
      auto transferBitWidth = [&]() -> int64_t {
        if (auto vectorType =
                dyn_cast<VectorType>(gatherToLDSOp.getTransferType())) {
          return vectorType.getElementTypeBitWidth() *
                 vectorType.getNumElements();
        }
        return gatherToLDSOp.getTransferType().getIntOrFloatBitWidth();
      }();
      if (accessBitWidth != transferBitWidth) {
        return;
      }
      gatherToLDSOps.push_back(gatherToLDSOp);
      continue;
    }
    // View-like ops from FlattenSwizzleHintAllocs: trace through to find
    // transitive load/store users (possibly through scf.for iter_args).
    if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(user)) {
      collectTransitiveSwizzleableUsers(user->getResult(0), accessWidth,
                                        transitiveLoads, transitiveStores);
      continue;
    }
    hintOp.emitError() << "unsupported SwizzleHintOp user: " << user;
    return;
  }

  // Swizzle direct 1-D users.
  for (vector::LoadOp load : loads) {
    rewriter.setInsertionPoint(load);
    swizzleLoad(rewriter, load, hintOp);
  }
  for (vector::StoreOp store : stores) {
    rewriter.setInsertionPoint(store);
    swizzleStore(rewriter, store, hintOp);
  }
  for (amdgpu::GatherToLDSOp gatherToLDSOp : gatherToLDSOps) {
    rewriter.setInsertionPoint(gatherToLDSOp);
    swizzleGatherToLDS(rewriter, gatherToLDSOp, hintOp);
  }

  // Swizzle transitive N-D users found through expand_shape/scf.for chains.
  for (vector::LoadOp load : transitiveLoads) {
    rewriter.setInsertionPoint(load);
    swizzleNDLoad(rewriter, load, hintOp);
  }
  for (vector::StoreOp store : transitiveStores) {
    rewriter.setInsertionPoint(store);
    swizzleNDStore(rewriter, store, hintOp);
  }
}

void ResolveSwizzleHintsPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();

  // Collect all hint ops.
  SmallVector<IREE::Codegen::SwizzleHintOp> hintOps;
  funcOp.walk(
      [&](IREE::Codegen::SwizzleHintOp hint) { hintOps.push_back(hint); });

  // Swizzle all load/store uses of the hint ops if possible. If we can't
  // guarantee all accesses of a particular hint are swizzled, this will
  // silently pass through for that hint.
  IRRewriter rewriter(funcOp->getContext());
  for (IREE::Codegen::SwizzleHintOp hintOp : hintOps) {
    if (failed(verifyFlatContiguousSwizzleHintOp(hintOp))) {
      return signalPassFailure();
    }
    resolveHintOp(rewriter, hintOp);
  }

  // Drop all hints.
  for (auto hintOp : hintOps) {
    rewriter.replaceOp(hintOp, hintOp.getOperand());
  }
}

} // namespace mlir::iree_compiler
