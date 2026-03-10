// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Performance experiment pass: replaces strided byte-granularity LDS reads for
// MXFP4 scale operands with contiguous dword reads along the K dimension,
// producing ds_read_b32 instead of ds_read_u8. Also reroutes scale operands
// so each MFMA gets a full 4-byte dword with scalesIdx=0, providing valid
// scale bytes to all 4 K-blocks (matching a preshuffled m=2,k=2 layout).

#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUCOALESCESCALEREADSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct AMDGPUCoalesceScaleReadsPass
    : impl::AMDGPUCoalesceScaleReadsPassBase<AMDGPUCoalesceScaleReadsPass> {
  void runOnOperation() override;
};

// Check if a memref is in GPU workgroup address space.
static bool isWorkgroupMemory(MemRefType memrefType) {
  auto addrSpace =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(memrefType.getMemorySpace());
  return addrSpace &&
         addrSpace.getValue() == gpu::AddressSpace::Workgroup;
}

// Given a transfer_read and an extract position along dim 0, create a
// contiguous dword read (vector<1x1x1x4>) from the same memref at the
// correct row offset, then shape_cast to vector<4xf8E8M0FNU>.
static Value createDwordRead(OpBuilder &builder,
                             vector::TransferReadOp originalRead,
                             int64_t dim0Offset) {
  Location loc = originalRead.getLoc();
  auto memrefType = cast<MemRefType>(originalRead.getBase().getType());
  Type elemType = memrefType.getElementType();
  int64_t rank = memrefType.getRank();

  // Build the new result type: vector<1x1x...x1x4xelemType>
  // (all dims = 1 except the last = 4)
  SmallVector<int64_t> dwordShape(rank, 1);
  dwordShape.back() = 4;
  auto dwordVecType = VectorType::get(dwordShape, elemType);

  // Build the indices: same as original but with dim0 += offset, last dim = 0.
  SmallVector<Value> indices(originalRead.getIndices());
  if (dim0Offset != 0) {
    Value offset =
        arith::ConstantIndexOp::create(builder, loc, dim0Offset);
    indices[0] = arith::AddIOp::create(builder, loc, indices[0], offset);
  }
  // Set the K dimension (last index) to 0 to read K=0..3 contiguously.
  indices.back() = arith::ConstantIndexOp::create(builder, loc, 0);

  // Build in_bounds: all true (we know the tile fits).
  SmallVector<bool> inBounds(rank, true);

  auto dwordRead = vector::TransferReadOp::create(
      builder, loc, dwordVecType, originalRead.getBase(), indices,
      originalRead.getPermutationMap().getContext()
          ? AffineMapAttr::get(AffineMap::getMinorIdentityMap(
                rank, rank, builder.getContext()))
          : originalRead.getPermutationMapAttr(),
      originalRead.getPadding(), /*mask=*/Value(),
      builder.getBoolArrayAttr(inBounds));

  // Shape-cast to flat vector<4xelemType>.
  auto flatType = VectorType::get({4}, elemType);
  return vector::ShapeCastOp::create(builder, loc, flatType,
                                     dwordRead.getResult());
}

void AMDGPUCoalesceScaleReadsPass::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *ctx = &getContext();
  Type f8E8M0 = Float8E8M0FNUType::get(ctx);

  // Cache: (transfer_read op, dim0 extract index) -> dword flat vector.
  DenseMap<std::pair<Operation *, int64_t>, Value> dwordCache;

  funcOp->walk([&](amdgpu::ScaledMFMAOp mfmaOp) {
    // Process both scaleA (operand 3) and scaleB (operand 4).
    for (unsigned opIdx : {3u, 4u}) {
      Value scaleVal = mfmaOp->getOperand(opIdx);

      // Match the padScales pattern:
      //   %byte = vector.extract %scaleSrc[i, 0, 0, 0]
      //   %padded = vector.insert %byte, %zeros[0]
      auto insertOp = scaleVal.getDefiningOp<vector::InsertOp>();
      if (!insertOp)
        continue;
      if (isa<VectorType>(insertOp.getValueToStore().getType()))
        continue;

      auto extractOp =
          insertOp.getValueToStore().getDefiningOp<vector::ExtractOp>();
      if (!extractOp)
        continue;

      Value scaleSrc = extractOp.getOperand(0);
      auto scaleSrcType = dyn_cast<VectorType>(scaleSrc.getType());
      if (!scaleSrcType || scaleSrcType.getElementType() != f8E8M0)
        continue;

      // Get the M_outer (or N_outer) index from the extract position.
      auto extractPos = extractOp.getStaticPosition();
      if (extractPos.empty())
        continue;
      int64_t dim0Idx = extractPos[0];

      // Trace scaleSrc to its transfer_read from workgroup memory.
      auto transferRead = scaleSrc.getDefiningOp<vector::TransferReadOp>();
      if (!transferRead)
        continue;
      auto memrefType =
          dyn_cast<MemRefType>(transferRead.getBase().getType());
      if (!memrefType || !isWorkgroupMemory(memrefType))
        continue;
      if (memrefType.getElementType() != f8E8M0)
        continue;

      // Verify the last dim has size 4 (K_tile = 4).
      if (memrefType.getShape().back() != 4)
        continue;

      // Create or reuse the dword read for this (transfer_read, dim0Idx).
      auto cacheKey =
          std::make_pair(transferRead.getOperation(), dim0Idx);
      Value dwordFlat;
      if (auto it = dwordCache.find(cacheKey); it != dwordCache.end()) {
        dwordFlat = it->second;
      } else {
        OpBuilder builder(transferRead->getBlock(),
                          std::next(transferRead->getIterator()));
        dwordFlat = createDwordRead(builder, transferRead, dim0Idx);
        dwordCache[cacheKey] = dwordFlat;
      }

      // Replace the MFMA's scale operand with the full dword.
      mfmaOp->setOperand(opIdx, dwordFlat);

      // Set scalesIdx=0 so byte i maps to K-block i.
      if (opIdx == 3)
        mfmaOp.setScalesIdxA(0);
      else
        mfmaOp.setScalesIdxB(0);
    }
  });
}

} // namespace
} // namespace mlir::iree_compiler
