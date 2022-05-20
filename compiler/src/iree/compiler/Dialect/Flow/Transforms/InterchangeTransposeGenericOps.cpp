// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- InterchangeTransposeGenericOps.cpp -------------------===//
//
// Interchange loops in generic ops to make the transpose happen on the outputs
// instead of inputs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct TransposeGenericOpPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (genericOp.getNumInputs() == 0 || genericOp.getNumOutputs() == 0)
      return failure();

    // check the generic op can be fused with the input operands.
    auto indexingMap0 =
        genericOp.getTiedIndexingMap(genericOp.getInputOperand(0));
    if (indexingMap0.isIdentity()) {
      // already in a good form.
      return failure();
    }

    // check if the operation is fully parallel.
    if ((indexingMap0.getNumDims() != genericOp.getNumParallelLoops()) ||
        !indexingMap0.isPermutation())
      return failure();

    // check if all input indexing maps share the same permutation.
    for (OpOperand *operand : genericOp.getInputOperands()) {
      auto indexingMap = genericOp.getTiedIndexingMap(operand);
      if (indexingMap != indexingMap0) return failure();
    }

    // make the input indexing maps identity by interchanging.
    SmallVector<unsigned> interchange;

    for (unsigned i = 0, e = indexingMap0.getNumDims(); i != e; ++i)
      interchange.push_back(indexingMap0.getDimPosition(i));

    return interchangeGenericOp(rewriter, genericOp, interchange);
  }
};

struct InterchangeTransposeGenericOpsPass
    : public InterchangeTransposeGenericOpsBase<
          InterchangeTransposeGenericOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeGenericOpPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createInterchangeTransposeGenericOpsPass() {
  return std::make_unique<InterchangeTransposeGenericOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
