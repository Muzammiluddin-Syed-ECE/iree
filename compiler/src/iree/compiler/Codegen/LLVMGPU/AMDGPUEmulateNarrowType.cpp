// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/EmulateNarrowType.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_AMDGPUEMULATENARROWTYPEPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

struct ConvertRawBufferCast final
    : OpConversionPattern<amdgpu::FatRawBufferCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::FatRawBufferCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!newTy) {
      return rewriter.notifyMatchFailure(
          op->getLoc(), llvm::formatv("failed to convert memref type: {0}",
                                      op.getResult().getType()));
    }
    if (newTy == op.getResult().getType()) {
      // Nothing to do.
      return failure();
    }

    // |validBytes| and |cacheSwizzleStride| are independent of element type
    // and don't need to be updated.
    rewriter.replaceOpWithNewOp<amdgpu::FatRawBufferCastOp>(
        op, newTy, adaptor.getSource(), adaptor.getValidBytes(),
        adaptor.getCacheSwizzleStride(), adaptor.getBoundsCheck(),
        adaptor.getResetOffset());
    return success();
  }
};

/// Converts GatherToLDSOp when its memrefs change from sub-byte types
/// (e.g. f4E2M1FN) to byte-sized types (i8) during narrow type emulation.
/// The pattern linearizes multi-dimensional indices into the converted 1D
/// memref space and adjusts the transfer type accordingly.
struct ConvertGatherToLDS final : OpConversionPattern<amdgpu::GatherToLDSOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(amdgpu::GatherToLDSOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType origSrcType = op.getSrc().getType();
    MemRefType origDstType = op.getDst().getType();
    auto newSrcType = cast<MemRefType>(adaptor.getSrc().getType());
    auto newDstType = cast<MemRefType>(adaptor.getDst().getType());

    // Only convert sub-byte element types.
    if (origSrcType.getElementTypeBitWidth() >= 8 &&
        origDstType.getElementTypeBitWidth() >= 8) {
      return failure();
    }

    // If types didn't change, nothing to do.
    if (newSrcType == origSrcType && newDstType == origDstType) {
      return failure();
    }

    Location loc = op.getLoc();
    int64_t origSrcBits = origSrcType.getElementTypeBitWidth();
    int64_t newSrcBits = newSrcType.getElementTypeBitWidth();
    int64_t origDstBits = origDstType.getElementTypeBitWidth();
    int64_t newDstBits = newDstType.getElementTypeBitWidth();

    // Only convert when the transfer vector's total bits are a multiple of
    // the new element bit width. E.g. vector<3xf4E2M1FN> (12 bits) cannot
    // be cleanly packed into i8 elements.
    if (auto vecType = dyn_cast<VectorType>(op.getTransferType())) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      if (totalBits % newSrcBits != 0) {
        return rewriter.notifyMatchFailure(
            op,
            "transfer vector bit-width is not a multiple of the new element "
            "bit width");
      }
    }

    // Check both source and destination convertibility before modifying IR.
    if (!canLinearizeAndPack(op.getSrcIndices(), origSrcType, origSrcBits,
                             newSrcBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize source indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }
    if (!canLinearizeAndPack(op.getDstIndices(), origDstType, origDstBits,
                             newDstBits)) {
      return rewriter.notifyMatchFailure(
          op, "failed to linearize destination indices (dynamic or mismatched "
              "strides/offset, or invalid bit-width ratio)");
    }

    // Linearize source indices into a 1D byte-offset index.
    Value srcIdx = linearizeAndPack(rewriter, loc, op.getSrcIndices(),
                                    origSrcType, origSrcBits, newSrcBits);

    // Linearize destination indices.
    Value dstIdx = linearizeAndPack(rewriter, loc, op.getDstIndices(),
                                    origDstType, origDstBits, newDstBits);

    // Adjust transfer type to use the new element type.
    Type newTransferType = convertTransferType(
        rewriter.getContext(), op.getTransferType(), origSrcBits, newSrcBits);

    amdgpu::GatherToLDSOp::create(
        rewriter, loc, adaptor.getSrc(), ValueRange{srcIdx}, adaptor.getDst(),
        ValueRange{dstIdx}, TypeAttr::get(newTransferType), op.getAsyncAttr());

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Checks whether linearizeAndPack can succeed without modifying IR.
  static bool canLinearizeAndPack(ValueRange indices, MemRefType origType,
                                  int64_t origBits, int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();
    for (int64_t stride : strides) {
      if (ShapedType::isDynamic(stride)) {
        return false;
      }
    }
    if (indices.size() != strides.size()) {
      return false;
    }
    if (origBits != newBits &&
        (newBits <= origBits || newBits % origBits != 0)) {
      return false;
    }
    return true;
  }

  // Linearizes multi-dimensional indices into a 1D index for the packed
  // byte-addressable memref. The caller must ensure canLinearizeAndPack()
  // returns true before calling this.
  //   linearIdx = offset + sum(idx[i] * stride[i])
  //   packedIdx = linearIdx / (newBits / origBits)
  static Value linearizeAndPack(ConversionPatternRewriter &rewriter,
                                Location loc, ValueRange indices,
                                MemRefType origType, int64_t origBits,
                                int64_t newBits) {
    auto [strides, offset] = origType.getStridesAndOffset();

    // For dynamic offsets, use 0: the converted memref descriptor already
    // carries the (converted) offset, so indices are relative to it.
    int64_t staticOffset = ShapedType::isDynamic(offset) ? 0 : offset;
    auto overflowFlags =
        arith::IntegerOverflowFlags::nsw | arith::IntegerOverflowFlags::nuw;
    Value linearIdx =
        arith::ConstantIndexOp::create(rewriter, loc, staticOffset);
    for (auto [idx, stride] : llvm::zip(indices, strides)) {
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value product =
          arith::MulIOp::create(rewriter, loc, idx, strideVal, overflowFlags);
      linearIdx = arith::AddIOp::create(rewriter, loc, linearIdx, product,
                                        overflowFlags);
    }

    // Pack: convert from origBits-element units to newBits-element units.
    if (origBits != newBits) {
      int64_t packRatio = newBits / origBits;
      Value ratioVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      linearIdx = arith::DivUIOp::create(rewriter, loc, linearIdx, ratioVal);
    }

    return linearIdx;
  }

  // Converts the transfer type from sub-byte elements to byte-sized elements,
  // preserving the total transfer size in bits. The caller must ensure
  // totalBits is a multiple of newBits (the op verifier enforces that
  // transfer sizes are 8, 16, 32, 96, or 128 bits, all multiples of 8).
  static Type convertTransferType(MLIRContext *context, Type origType,
                                  int64_t origBits, int64_t newBits) {
    if (auto vecType = dyn_cast<VectorType>(origType)) {
      int64_t totalBits =
          vecType.getNumElements() * vecType.getElementTypeBitWidth();
      assert(totalBits % newBits == 0 &&
             "transfer size must be a multiple of the new element bit width");
      int64_t newElems = totalBits / newBits;
      return VectorType::get({newElems}, IntegerType::get(context, newBits));
    }
    return IntegerType::get(context, newBits);
  }
};

struct ConvertMemRefCast final : OpConversionPattern<memref::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newTy = getTypeConverter()->convertType<MemRefType>(op.getType());
    if (!newTy || newTy == op.getType())
      return failure();
    rewriter.replaceOpWithNewOp<memref::CastOp>(op, newTy,
                                                 adaptor.getSource());
    return success();
  }
};

/// Converts extract_strided_metadata on sub-byte memrefs during narrow type
/// emulation. Upstream patterns (ConvertVectorLoad, etc.) create
/// extract_strided_metadata on the original sub-byte memref to compute
/// linearized indices. This op is illegal because its base buffer result has
/// sub-byte type. We legalize it by creating the ESM on the converted (i8)
/// source and providing metadata in original element units so the upstream
/// linearization (getLinearizedMemRefOffsetAndSize) produces correct results.
struct ConvertExtractStridedMetadata final
    : OpConversionPattern<memref::ExtractStridedMetadataOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExtractStridedMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origType = cast<MemRefType>(op.getSource().getType());
    unsigned origBits = origType.getElementTypeBitWidth();
    if (origBits >= 8)
      return failure();

    auto adaptedType = cast<MemRefType>(adaptor.getSource().getType());
    if (adaptedType == origType)
      return failure();

    unsigned newBits = adaptedType.getElementTypeBitWidth();
    int64_t packRatio = newBits / origBits;
    Location loc = op.getLoc();

    // Look through unrealized_conversion_cast ops to bypass materialization
    // casts inserted by the framework for converted block arguments. Using
    // adaptor.getSource() directly would keep the materialization alive,
    // preventing cleanup and causing "unresolved materialization" errors.
    Value source = adaptor.getSource();
    while (auto castOp =
               source.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getNumOperands() != 1)
        break;
      source = castOp.getOperand(0);
    }
    if (source.getType() != adaptedType)
      source = adaptor.getSource();

    auto newESM = memref::ExtractStridedMetadataOp::create(
        rewriter, loc, source);

    SmallVector<Value> results;
    results.push_back(newESM.getBaseBuffer());

    auto [strides, offset] = origType.getStridesAndOffset();
    if (ShapedType::isDynamic(offset)) {
      Value packVal = arith::ConstantIndexOp::create(rewriter, loc, packRatio);
      results.push_back(
          arith::MulIOp::create(rewriter, loc, newESM.getOffset(), packVal));
    } else {
      results.push_back(arith::ConstantIndexOp::create(rewriter, loc, offset));
    }

    for (int64_t size : origType.getShape()) {
      results.push_back(arith::ConstantIndexOp::create(
          rewriter, loc, ShapedType::isDynamic(size) ? 0 : size));
    }

    for (int64_t stride : strides) {
      results.push_back(arith::ConstantIndexOp::create(
          rewriter, loc, ShapedType::isDynamic(stride) ? 1 : stride));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ConvertReinterpretCast final
    : OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType oldType = op.getType();
    unsigned oldBits = oldType.getElementTypeBitWidth();
    if (oldBits >= 8)
      return failure();

    MemRefType newTy =
        getTypeConverter()->convertType<MemRefType>(oldType);
    if (!newTy || newTy == oldType)
      return failure();

    int64_t packFactor = newTy.getElementTypeBitWidth() / oldBits;
    int64_t totalOldElements = 1;
    for (int64_t dim : oldType.getShape()) {
      if (ShapedType::isDynamic(dim))
        return failure();
      totalOldElements *= dim;
    }
    int64_t newSize = llvm::divideCeilSigned(totalOldElements, packFactor);
    Location loc = op.getLoc();

    OpFoldResult newOffset;
    OpFoldResult oldOffset = op.getMixedOffsets()[0];
    if (auto val = dyn_cast<Value>(oldOffset)) {
      Value divisor =
          arith::ConstantIndexOp::create(rewriter, loc, packFactor);
      newOffset = arith::DivUIOp::create(rewriter, loc, val, divisor)
                      ->getResult(0);
    } else {
      int64_t staticVal = cast<IntegerAttr>(cast<Attribute>(oldOffset)).getInt();
      newOffset = rewriter.getIndexAttr(staticVal / packFactor);
    }

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, newTy, adaptor.getSource(), newOffset,
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(newSize)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)});
    return success();
  }
};

struct AMDGPUEmulateNarrowTypePass final
    : impl::AMDGPUEmulateNarrowTypePassBase<AMDGPUEmulateNarrowTypePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    affine::AffineDialect, IREE::HAL::HALDialect>();
  }

  void runOnOperation() override {
    auto populateAMDGPUPatterns =
        [](arith::NarrowTypeEmulationConverter &typeConverter,
           RewritePatternSet &patterns, ConversionTarget &target) {
          auto opLegalCallback = [&typeConverter](Operation *op) {
            return typeConverter.isLegal(op);
          };
          target.addDynamicallyLegalDialect<amdgpu::AMDGPUDialect>(
              opLegalCallback);
          patterns.add<ConvertRawBufferCast, ConvertGatherToLDS>(
              typeConverter, patterns.getContext());
          patterns.add<ConvertMemRefCast, ConvertExtractStridedMetadata>(
              typeConverter, patterns.getContext());
          patterns.add<ConvertReinterpretCast>(typeConverter,
                                               patterns.getContext(),
                                               /*benefit=*/2);
        };
    if (failed(emulateNarrowType(getOperation(), /*disableAtomic=*/true,
                                 populateAMDGPUPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
