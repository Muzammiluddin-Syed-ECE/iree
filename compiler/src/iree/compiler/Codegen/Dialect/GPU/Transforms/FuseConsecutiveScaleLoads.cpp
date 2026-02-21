// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_FUSECONSECUTIVESCALELOADSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {

// Recursively ensures that `val` (and its defining op chain) dominates
// `insertBefore`. If an op is already before `insertBefore`, it is returned
// as-is. Otherwise, the op and its transitive operand producers are cloned
// at the insertion point. Block arguments always dominate.
static Value ensureDominates(PatternRewriter &rewriter, Value val,
                             Operation *insertBefore, IRMapping &cloneMap) {
  if (!val.getDefiningOp())
    return val;

  if (Value mapped = cloneMap.lookupOrNull(val))
    return mapped;

  Operation *defOp = val.getDefiningOp();
  if (defOp->getBlock() != insertBefore->getBlock() ||
      defOp->isBeforeInBlock(insertBefore))
    return val;

  for (Value operand : defOp->getOperands())
    ensureDominates(rewriter, operand, insertBefore, cloneMap);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertBefore);
  Operation *cloned = rewriter.clone(*defOp, cloneMap);
  for (auto [oldRes, newRes] :
       llvm::zip(defOp->getResults(), cloned->getResults()))
    cloneMap.map(oldRes, newRes);

  return cloneMap.lookup(val);
}

// Finds the next ScaledMFMAOp in the accumulator chain, i.e. the unique
// ScaledMFMAOp user of `op`'s result that feeds into its destC operand.
static amdgpu::ScaledMFMAOp findNextInChain(amdgpu::ScaledMFMAOp op) {
  for (auto *user : op.getDestD().getUsers()) {
    auto mfma = dyn_cast<amdgpu::ScaledMFMAOp>(user);
    if (mfma && mfma.getDestC() == op.getDestD())
      return mfma;
  }
  return nullptr;
}

// Traces a ScaledMFMAOp's scale operand (index 3 for scalesA, 4 for scalesB)
// through the padScales pattern to extract the underlying scalar byte value.
//
// Expected pattern:
//   %scalar = vector.extract %src[0] : f8 from vector<Nxf8>
//   %padded = vector.insert %scalar, %zeros[0] : f8 into vector<4xf8>
//
// Returns the scalar value, or nullptr on failure.
static Value traceScaleScalar(amdgpu::ScaledMFMAOp op, unsigned operandIdx) {
  auto insertOp =
      op->getOperand(operandIdx).getDefiningOp<vector::InsertOp>();
  if (!insertOp)
    return nullptr;

  Value stored = insertOp.getValueToStore();
  if (isa<VectorType>(stored.getType()))
    return nullptr;

  return stored;
}

/// Matches the first ScaledMFMAOp in an accumulator chain of 2-4 ops (all
/// with scalesIdx == 0), packs the 4 individual scale byte values into a
/// single vector<4xf8E8M0FNU>, and rewrites each op to use the shared packed
/// scale with the appropriate scalesIdx (0..chainLen-1).
struct FuseConsecutiveScaleLoadsPattern
    : public OpRewritePattern<amdgpu::ScaledMFMAOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(amdgpu::ScaledMFMAOp firstOp,
                                PatternRewriter &rewriter) const override {
    // Must be the first in a chain: its accumulator is not from a ScaledMFMAOp.
    if (firstOp.getDestC().getDefiningOp<amdgpu::ScaledMFMAOp>())
      return failure();

    // Collect the accumulator chain (up to 4).
    SmallVector<amdgpu::ScaledMFMAOp, 4> chain;
    chain.push_back(firstOp);
    while (chain.size() < 4) {
      amdgpu::ScaledMFMAOp next = findNextInChain(chain.back());
      if (!next)
        break;
      chain.push_back(next);
    }

    if (chain.size() < 2)
      return failure();

    // All ops must have scalesIdx == 0 (not already fused).
    for (auto op : chain) {
      if (op.getScalesIdxA() != 0 || op.getScalesIdxB() != 0)
        return failure();
    }

    // Process each scale operand index: 3 = scalesA, 4 = scalesB.
    for (unsigned scaleOpIdx : {3u, 4u}) {
      SmallVector<Value, 4> scalars;
      for (auto op : chain) {
        Value scalar = traceScaleScalar(op, scaleOpIdx);
        if (!scalar)
          return failure();
        scalars.push_back(scalar);
      }

      // Clone any scalar definitions that don't dominate firstOp.
      IRMapping cloneMap;
      for (auto &scalar : scalars)
        scalar =
            ensureDominates(rewriter, scalar, firstOp.getOperation(), cloneMap);

      // Build the packed vector<4xf8E8M0FNU> before the first MFMA.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(firstOp);
      Location loc = firstOp.getLoc();

      FloatType f8E8M0 = rewriter.getF8E8M0Type();
      auto packedType = VectorType::get({4}, f8E8M0);
      Value packed = arith::ConstantOp::create(
          rewriter, loc,
          SplatElementsAttr::get(
              packedType,
              llvm::APFloat::getSmallest(f8E8M0.getFloatSemantics())));

      for (int64_t k = 0, e = chain.size(); k < e; ++k)
        packed = vector::InsertOp::create(rewriter, loc, scalars[k], packed,
                                          ArrayRef<int64_t>{k});

      // Replace each op's scale operand and set scalesIdx.
      for (int64_t k = 0, e = chain.size(); k < e; ++k) {
        rewriter.modifyOpInPlace(chain[k], [&] {
          chain[k]->setOperand(scaleOpIdx, packed);
          if (scaleOpIdx == 3)
            chain[k].setScalesIdxA(k);
          else
            chain[k].setScalesIdxB(k);
        });
      }
    }

    return success();
  }
};

struct FuseConsecutiveScaleLoadsPass final
    : impl::FuseConsecutiveScaleLoadsPassBase<FuseConsecutiveScaleLoadsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FuseConsecutiveScaleLoadsPattern>(context);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

void populateFuseConsecutiveScaleLoadsPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseConsecutiveScaleLoadsPattern>(patterns.getContext());
}

} // namespace mlir::iree_compiler::IREE::GPU
