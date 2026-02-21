// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::GPU {

#define GEN_PASS_DEF_DISTRIBUTEINNERTILEDTOLANESPASS
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h.inc"

namespace {
struct DistributeInnerTiledToLanesPass final
    : impl::DistributeInnerTiledToLanesPassBase<
          DistributeInnerTiledToLanesPass> {
  void runOnOperation() override;
};
} // namespace

LogicalResult fuseProducersGreedily(RewriterBase &rewriter,
                                    scf::ForallOp laneForall) {

  std::deque<tensor::ExtractSliceOp> candidates;
  laneForall->walk([&](tensor::ExtractSliceOp extractSliceOp) {
    auto producer = extractSliceOp.getSource().getDefiningOp<TilingInterface>();
    if (producer && producer->getBlock() != laneForall.getBody()) {
      candidates.push_back(extractSliceOp);
    }

    // Collect slices to fuse producers of loop destinations into lane foralls.
    if (laneForall.getRegionIterArgs()[0] == extractSliceOp.getSource() &&
        laneForall.getOutputs()[0].getDefiningOp<TilingInterface>()) {
      candidates.push_back(extractSliceOp);
    }
  });

  SmallVector<LoopLikeOpInterface> loops = {laneForall};

  OpBuilder::InsertionGuard g(rewriter);
  while (!candidates.empty()) {
    // Traverse the slices in BFS fashion.
    tensor::ExtractSliceOp candidateSliceOp = candidates.front();
    candidates.pop_front();

    // Materialize the slice of the producer in place.
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(rewriter, candidateSliceOp, loops);
    if (!fusedProducer) {
      continue;
    }

    // We have no way to know whether a multi-use value can be yielded from the
    // parallel loop so never yield a replacement.

    // Add more fusion candidates to the worklist.
    for (auto tiledOp : fusedProducer->tiledOps) {
      for (OpOperand &operand : tiledOp->getOpOperands()) {
        auto sliceOp = operand.get().getDefiningOp<tensor::ExtractSliceOp>();
        if (!sliceOp) {
          continue;
        }
        candidates.push_back(sliceOp);
      }
    }
  }
  return success();
}

void DistributeInnerTiledToLanesPass::runOnOperation() {
  MLIRContext *context = &getContext();
  mlir::FunctionOpInterface funcOp = getOperation();

  SmallVector<IREE::Codegen::InnerTiledOp> tiledOps;
  funcOp.walk([&](IREE::Codegen::InnerTiledOp tiledOp) {
    if (!tiledOp.hasTensorSemantics()) {
      return;
    }
    tiledOps.push_back(tiledOp);
  });
  if (tiledOps.empty()) {
    return;
  }

  // Group inner_tiled ops into accumulator chains. A chain is a maximal
  // sequence where op[i].getResult(0) feeds into op[i+1].getOutputs()[0]
  // and the intermediate result has exactly one use.
  SmallVector<SmallVector<IREE::Codegen::InnerTiledOp>> chains;
  DenseSet<Operation *> visited;

  for (auto tiledOp : tiledOps) {
    if (visited.contains(tiledOp.getOperation()))
      continue;

    auto root = tiledOp;
    while (true) {
      auto prevOp = root.getOutputs()[0]
                        .getDefiningOp<IREE::Codegen::InnerTiledOp>();
      if (!prevOp || visited.contains(prevOp.getOperation()) ||
          !prevOp.getResult(0).hasOneUse())
        break;
      root = prevOp;
    }

    SmallVector<IREE::Codegen::InnerTiledOp> chain;
    auto current = root;
    while (current) {
      chain.push_back(current);
      visited.insert(current.getOperation());

      IREE::Codegen::InnerTiledOp next;
      if (current.getResult(0).hasOneUse()) {
        Operation *user = *current.getResult(0).getUsers().begin();
        if (auto nextOp = dyn_cast<IREE::Codegen::InnerTiledOp>(user)) {
          if (nextOp.getOutputs()[0] == current.getResult(0) &&
              !visited.contains(nextOp.getOperation()))
            next = nextOp;
        }
      }
      current = next;
    }
    chains.push_back(std::move(chain));
  }

  IRRewriter rewriter(funcOp);
  for (auto &chain : chains) {
    FailureOr<Operation *> maybeForall;

    if (chain.size() == 1) {
      rewriter.setInsertionPoint(chain[0]);
      maybeForall = distributeInnerTiledOp(rewriter, chain[0]);
    } else {
      maybeForall = distributeInnerTiledChain(rewriter, chain);
    }

    if (failed(maybeForall)) {
      funcOp.emitError() << "failed to distribute inner_tiled ops to lanes";
      return signalPassFailure();
    }

    auto laneForall = cast<scf::ForallOp>(*maybeForall);
    rewriter.setInsertionPointToStart(laneForall.getBody());
    if (failed(fuseProducersGreedily(rewriter, laneForall))) {
      funcOp.emitError() << "failed to fuse producers into lane forall";
      return signalPassFailure();
    }
  }

  // Post distribution cleanup patterns.
  {
    RewritePatternSet patterns(context);
    tensor::populateFoldTensorEmptyPatterns(patterns);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
    tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
    scf::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      funcOp.emitError() << "cleanup failed\n";
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler::IREE::GPU
