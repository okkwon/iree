// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Utils/ConversionUtils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
class FoldConstantAndExtractSlicePass
    : public FoldConstantAndExtractSliceBase<FoldConstantAndExtractSlicePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    Operation *op = getOperation();

    // ConstantOp + ExtractSliceOp -> ConstantOp
    tensor::ControlConstantExtractSliceFusionFn constantSliceControlFn =
        [](tensor::ExtractSliceOp op) {
          // testing
          return true;
        };
    tensor::populateFoldConstantExtractSlicePatterns(patterns,
                                                     constantSliceControlFn);
    if (failed(applyPatternsAndFoldGreedily(op->getRegions(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createFoldConstantAndExtractSlicePass() {
  return std::make_unique<FoldConstantAndExtractSlicePass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
