// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// Returns an XORShuffleAttr if the layout would cause bank conflicts on shared
// memory, std::nullopt otherwise. Only handles rank-2 layouts.
static std::optional<IREE::Codegen::XORShuffleAttr>
computeSwizzleForLayout(MLIRContext *ctx,
                        IREE::VectorExt::NestedLayoutAttr layout,
                        Type elementType, int64_t numBanks) {
  ArrayRef<int64_t> subgroupTile = layout.getSubgroupTile();
  if (subgroupTile.size() != 2) {
    return std::nullopt;
  }

  ArrayRef<int64_t> batchTile = layout.getBatchTile();
  ArrayRef<int64_t> outerTile = layout.getOuterTile();
  ArrayRef<int64_t> threadTile = layout.getThreadTile();
  ArrayRef<int64_t> elementTile = layout.getElementTile();

  int64_t innerDimSize = subgroupTile[1] * batchTile[1] * outerTile[1] *
                         threadTile[1] * elementTile[1];
  int64_t accessWidth = elementTile[1];
  int64_t elemBytes = (elementType.getIntOrFloatBitWidth() + 7) / 8;

  int64_t rowBytes = innerDimSize * elemBytes;
  int64_t rowStrideBanks = (rowBytes / kSharedMemoryBankWidthBytes) % numBanks;

  // If stride is 0 every row hits the same banks (worst case).
  // If gcd(stride, numBanks) == 1, no conflicts.
  if (rowStrideBanks != 0 && std::gcd(rowStrideBanks, numBanks) == 1) {
    return std::nullopt;
  }

  // Compute XOR swizzle parameters.
  int64_t rowWidthElems = numBanks * kSharedMemoryBankWidthBytes / elemBytes;
  rowWidthElems = std::min(rowWidthElems, innerDimSize);

  if (accessWidth == 0 || rowWidthElems % accessWidth != 0) {
    return std::nullopt;
  }

  return IREE::Codegen::XORShuffleAttr::get(ctx, rowWidthElems, accessWidth,
                                            /*row_stride=*/int64_t(0),
                                            /*per_phase=*/int64_t(0));
}

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used. When |swizzle| is provided, wraps the allocation with
// a SwizzleHintOp using the flat-1D + expand_shape pattern.
static FailureOr<Value>
allocateTensorForVector(OpBuilder &b, Location loc, Value vector,
                        std::optional<IREE::Codegen::XORShuffleAttr> swizzle) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);

  Value dest;
  if (swizzle) {
    // Allocate a flat 1D tensor, attach swizzle hint, then expand back.
    int64_t numElements = tensorType.getNumElements();
    RankedTensorType flatType = RankedTensorType::get(
        {numElements}, tensorType.getElementType(), sharedMemoryAddrSpace);
    auto allocTensorOp = bufferization::AllocTensorOp::create(
        b, loc, flatType, ValueRange{}, Value());
    allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

    Value swizzled =
        IREE::Codegen::SwizzleHintOp::create(b, loc, allocTensorOp, *swizzle);
    dest = tensor::ExpandShapeOp::create(
        b, loc, tensorType, swizzled,
        {llvm::to_vector(llvm::seq(tensorType.getRank()))});
  } else {
    auto allocTensorOp = bufferization::AllocTensorOp::create(
        b, loc, tensorType, ValueRange{}, Value());
    allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);
    dest = allocTensorOp;
  }

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied =
      vector::TransferWriteOp::create(b, loc, vector, dest, indices, inBounds)
          .getResult();
  return copied;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value tensor) {
  Value c0 = arith::ConstantIndexOp::create(b, tensor.getLoc(), 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  return vector::TransferReadOp::create(b, tensor.getLoc(), vectorType, tensor,
                                        indices, /*padding=*/std::nullopt,
                                        inBounds)
      .getResult();
}

/// Materialize shared memory for all to_layout ops marked with
/// shared_memory_conversion. Clears the attribute after materialization.
static LogicalResult
materializeSharedMemoryConversions(FunctionOpInterface funcOp) {
  SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
  funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
    if (op.getSharedMemoryConversion()) {
      opsToPromote.push_back(op);
    }
  });

  OpBuilder builder(funcOp);
  for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
    // HACK: Until proper barrier placement is handled later we have to
    // synchronize explicitly in this pass.

    // Synchronize before the write to shared memory to avoid stepping over
    // reads in the previous iteration of a loop. We set this barrier
    // at the start of this block.
    builder.setInsertionPointToStart(op->getBlock());
    gpu::BarrierOp::create(builder, op->getLoc(), gpu::AddressSpace::Workgroup);

    builder.setInsertionPoint(op);
    OpOperand &operand = op.getInputMutable();

    // Detect bank conflicts from the layout and compute a swizzle if needed.
    std::optional<IREE::Codegen::XORShuffleAttr> swizzle;
    VectorType vecTy = cast<VectorType>(op.getType());
    if (auto nestedLayout =
            dyn_cast<IREE::VectorExt::NestedLayoutAttr>(op.getLayout())) {
      int64_t numBanks = 32;
      if (IREE::GPU::TargetAttr target = getGPUTargetAttr(op)) {
        if (auto bc = target.getWgp().getWorkgroupMemoryBankCount()) {
          numBanks = *bc;
        }
      }
      swizzle = computeSwizzleForLayout(op.getContext(), nestedLayout,
                                        vecTy.getElementType(), numBanks);
    }

    FailureOr<Value> ret =
        allocateTensorForVector(builder, op->getLoc(), operand.get(), swizzle);
    if (failed(ret)) {
      return failure();
    }

    // Synchronize after the write to shared memory before we read from it.
    auto synced =
        IREE::GPU::ValueBarrierOp::create(builder, op->getLoc(), *ret);

    VectorType inputTy = cast<VectorType>(op.getType());
    Value read = readVectorFromTensor(builder, inputTy, synced.getResult(0));
    operand.set(read);

    // Remove the shared_memory_conversion attribute from the to_layout
    // operation.
    op.setSharedMemoryConversion(false);
  }
  return success();
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Remove stretching broadcasts before layout analysis — the analysis
    // asserts that broadcasts don't stretch.
    {
      RewritePatternSet patterns(funcOp.getContext());
      populateVectorLayoutCanonicalizations(patterns);
      walkAndApplyPatterns(funcOp, std::move(patterns));
    }

    // Run layout analysis to find additional conflict points.
    // The analysis sees the materialized shared memory roundtrips and
    // only detects genuinely new conflicts.
    llvm::MapVector<Value, IREE::VectorExt::VectorLayoutInterface> layouts;
    propagateVectorLayoutInfo(funcOp, layouts);

    // Mark newly-inserted to_layout ops where input/output layouts don't
    // match — these are genuine conflicts needing shared memory.
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      auto inputLayout = layouts.lookup(op.getInput());
      auto outputLayout = layouts.lookup(op.getResult());
      if (inputLayout && outputLayout &&
          inputLayout.needsSharedMemoryForConversion(outputLayout)) {
        op.setSharedMemoryConversion(true);
      }
    });

    // Phase 3: Materialize any newly-found conflicts.
    if (failed(materializeSharedMemoryConversions(funcOp))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
