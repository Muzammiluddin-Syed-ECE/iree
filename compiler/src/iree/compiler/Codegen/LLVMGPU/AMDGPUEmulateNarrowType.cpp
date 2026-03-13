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
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

struct ConvertGatherToLDS final : ConversionPattern {
  ConvertGatherToLDS(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter,
                          amdgpu::GatherToLDSOp::getOperationName(),
                          /*benefit=*/1, context) {}

  SmallVector<Value> linearizeIndices(ConversionPatternRewriter &rewriter,
                                      Location loc, ArrayRef<Value> indices,
                                      MemRefType oldType,
                                      MemRefType newType) const {
    int64_t oldRank = oldType.getRank();
    int64_t newRank = newType.getRank();
    if (oldRank == newRank)
      return SmallVector<Value>(indices);

    unsigned oldBits = oldType.getElementTypeBitWidth();
    unsigned newBits = newType.getElementTypeBitWidth();
    int64_t packFactor = 1;
    if (oldBits < 8 && newBits == 8) {
      packFactor = 8 / oldBits;
    }

    ArrayRef<int64_t> oldShape = oldType.getShape();
    Value flat = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (int64_t i = 0; i < oldRank; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < oldRank; ++j) {
        stride *= oldShape[j];
      }
      Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
      Value term = arith::MulIOp::create(rewriter, loc, indices[i], strideVal);
      flat = arith::AddIOp::create(rewriter, loc, flat, term);
    }

    if (packFactor > 1) {
      Value pf = arith::ConstantIndexOp::create(rewriter, loc, packFactor);
      flat = arith::DivUIOp::create(rewriter, loc, flat, pf);
    }

    return {flat};
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto gatherOp = cast<amdgpu::GatherToLDSOp>(op);

    auto segmentAttr = op->getAttrOfType<DenseI32ArrayAttr>(
        gatherOp.getOperandSegmentSizeAttr());
    if (!segmentAttr)
      return failure();
    ArrayRef<int32_t> segments = segmentAttr.asArrayRef();
    int32_t numSrcIndices = segments[1];
    int32_t numDstIndices = segments[3];

    Value newSrc = operands[0];
    SmallVector<Value> oldSrcIndices(operands.begin() + 1,
                                     operands.begin() + 1 + numSrcIndices);
    Value newDst = operands[1 + numSrcIndices];
    SmallVector<Value> oldDstIndices(
        operands.begin() + 2 + numSrcIndices,
        operands.begin() + 2 + numSrcIndices + numDstIndices);

    auto oldSrcType = cast<MemRefType>(gatherOp.getSrc().getType());
    auto oldDstType = cast<MemRefType>(gatherOp.getDst().getType());
    auto newSrcType = cast<MemRefType>(newSrc.getType());
    auto newDstType = cast<MemRefType>(newDst.getType());

    Type oldTransferType = gatherOp.getTransferType();
    auto oldVecType = dyn_cast<VectorType>(oldTransferType);
    unsigned elemBits = oldVecType ? oldVecType.getElementTypeBitWidth()
                                  : oldTransferType.getIntOrFloatBitWidth();

    TypeAttr newTransferTypeAttr = gatherOp.getTransferTypeAttr();
    if (elemBits < 8) {
      unsigned packFactor = 8 / elemBits;
      if (oldVecType) {
        int64_t newNumElements =
            std::max<int64_t>(1, oldVecType.getNumElements() / packFactor);
        newTransferTypeAttr = TypeAttr::get(
            VectorType::get({newNumElements}, rewriter.getIntegerType(8)));
      } else {
        newTransferTypeAttr = TypeAttr::get(rewriter.getIntegerType(8));
      }
    }

    if (newSrcType == oldSrcType && newDstType == oldDstType &&
        newTransferTypeAttr == gatherOp.getTransferTypeAttr()) {
      return failure();
    }

    Location loc = op->getLoc();
    SmallVector<Value> newSrcIndices =
        linearizeIndices(rewriter, loc, oldSrcIndices, oldSrcType, newSrcType);
    SmallVector<Value> newDstIndices =
        linearizeIndices(rewriter, loc, oldDstIndices, oldDstType, newDstType);

    rewriter.replaceOpWithNewOp<amdgpu::GatherToLDSOp>(
        op, newSrc, newSrcIndices, newDst, newDstIndices, newTransferTypeAttr,
        gatherOp.getAsyncAttr());
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
          patterns.add<ConvertRawBufferCast>(typeConverter,
                                             patterns.getContext());
          patterns.add<ConvertGatherToLDS>(typeConverter,
                                           patterns.getContext());
        };
    if (failed(emulateNarrowType(getOperation(), /*disableAtomic=*/true,
                                 populateAMDGPUPatterns))) {
      return signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler
