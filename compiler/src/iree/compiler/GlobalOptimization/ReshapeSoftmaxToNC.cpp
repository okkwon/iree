// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ReshapeSoftmaxToNC.cpp - Pass to shape Softmax with NC --------------==//
//
// The pass is to reshape linalg::SoftmaxOp to have the NC shape.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-reshape-softmax-to-nc"

namespace mlir::iree_compiler::GlobalOptimization {

struct ReshapeSoftmaxToNC : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern<linalg::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputType = input.getType();

    if (!inputType.hasRank()) {
      return rewriter.notifyMatchFailure(op, "Expected ranked tensor type.");
    }

    if (!inputType.hasStaticShape()) {
      // TODO: support dynamic shape.
      return rewriter.notifyMatchFailure(op, "Expected static type.");
    }

    auto dimension = op.getDimension();
    auto inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    // Check if the softmax dimension is the innermost.
    if (dimension != rank - 1) {
      return rewriter.notifyMatchFailure(
          op, "Expected the softmax dimension is the innermost.");
    }

    if (rank == 1) {
      return rewriter.notifyMatchFailure(op, "Expected rank > 1.");
    }

    // Check if it is already 2D.
    if (rank == 2) {
      return rewriter.notifyMatchFailure(op, "Already has the NC shape.");
    }

    // new batch size
    int64_t n = 1;
    for (int i = 0; i < rank - 1; ++i) {
      n *= inputShape[i];
    }

    // collapsed shape
    auto resultType = RankedTensorType::get({n, inputShape[rank - 1]},
                                            inputType.getElementType());

    // Calculate the merged batch dimension size.
    auto getReassociationIndicesToCollapseLastTwoDims = [](Value v) {
      SmallVector<ReassociationIndices> reassociations;
      reassociations.resize(2);
      int64_t rank = cast<ShapedType>(v.getType()).getRank();
      for (int64_t i = 0; i < rank - 1; ++i)
        reassociations[0].emplace_back(i);
      reassociations[1].emplace_back(rank - 1);
      return reassociations;
    };

    auto reassociation = getReassociationIndicesToCollapseLastTwoDims(input);
    auto loc = op.getLoc();
    auto collapsedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, resultType, input, reassociation);
    auto collapsedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, resultType, op.getOutput(), reassociation);
    auto newSoftmaxOp = rewriter.create<linalg::SoftmaxOp>(
        loc, resultType, collapsedInput, collapsedOutput, /*dimension=*/1);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, inputType, newSoftmaxOp.getResult()[0], reassociation);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {
struct ReshapeSoftmaxToNCPass
    : public ReshapeSoftmaxToNCBase<ReshapeSoftmaxToNCPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};
} // namespace

void ReshapeSoftmaxToNCPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<ReshapeSoftmaxToNC>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<Pass> createReshapeSoftmaxToNCPass() {
  return std::make_unique<ReshapeSoftmaxToNCPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
