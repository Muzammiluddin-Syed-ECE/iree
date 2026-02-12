// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_UNROLLTOINTRINSICSPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct UnrollToIntrinsicsPass final
    : impl::UnrollToIntrinsicsPassBase<UnrollToIntrinsicsPass> {
  void runOnOperation() override;
};
} // namespace

void UnrollToIntrinsicsPass::runOnOperation() {
  MLIRContext *context = &getContext();

  // Step 1: Standard outer-dimension unrolling.  Unrolls inner_tiled ops so
  // that every outer iteration dimension is unit-size.  After this step each
  // inner_tiled op represents a single (possibly grouped) intrinsic tile.
  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUVectorUnrollPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Step 2: Decompose grouped inner tiles introduced by the `repeats`
  // attribute.  After the outer unroll each inner_tiled op has unit outer
  // dims but its inner tile may still be a multiple of the base intrinsic
  // tile (e.g. kScale=16 instead of 4 when repeats=[1,1,4]).  This step
  // slices the grouped tile into individual base-intrinsic inner_tiled ops
  // using vector::ExtractStridedSliceOp, handling both parallel (M/N) and
  // reduction (K) repeat dimensions.
  {
    RewritePatternSet patterns(context);
    GPU::populateDecomposeRepeatsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Step 3: Post-unrolling unit dim folding patterns in preparation for
  // later lowerings.
  {
    RewritePatternSet patterns(context);
    GPU::populateIREEGPUDropUnitDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
