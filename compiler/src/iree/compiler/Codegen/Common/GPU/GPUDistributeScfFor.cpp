// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GPUDistributeScfFor.cpp ----------------------------------------===//
//
// This pass distributes tiled loop nests with `iree.gpu.distribute_dim`
// attributes to invocations.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-gpu-distribute-scf-for"

namespace mlir::iree_compiler {

namespace {

struct DistributeLoop final : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only distribute if we see the marker attribute.
    auto numDimAttr =
        forOp->getAttrOfType<IntegerAttr>(getGPUDistributeAttrName());
    if (!numDimAttr)
      return failure();

    Location loc = forOp.getLoc();
    auto indexType = rewriter.getIndexType();
    const std::array<gpu::Dimension, 3> symDims = {
        gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z};
    gpu::Dimension symDim = symDims[numDimAttr.getInt()];
    auto idOp = rewriter.create<gpu::ThreadIdOp>(loc, indexType, symDim);
    auto countOp = rewriter.create<gpu::BlockDimOp>(loc, indexType, symDim);

    MLIRContext *context = getContext();
    AffineExpr sym0, sym1, sym2;
    bindSymbols(context, sym0, sym1, sym2);
    auto mulAddMap = AffineMap::get(0, 3, {sym0 * sym1 + sym2}, context);
    auto mulMap = AffineMap::get(0, 2, {sym0 * sym1}, context);

    auto newLb = rewriter.create<affine::AffineApplyOp>(
        loc, mulAddMap,
        ValueRange{idOp, forOp.getStep(), forOp.getLowerBound()});
    auto newStep = rewriter.create<affine::AffineApplyOp>(
        loc, mulMap, ValueRange{countOp, forOp.getStep()});

    forOp.getLowerBoundMutable().assign(newLb);
    forOp.getStepMutable().assign(newStep);
    // Remove the attribute to avoid endless recursion.
    forOp->removeAttr(getGPUDistributeAttrName());
    return success();
  }
};

struct GPUDistributeScfForPass final
    : public GPUDistributeScfForBase<GPUDistributeScfForPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DistributeLoop>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUDistributeScfForPass() {
  return std::make_unique<GPUDistributeScfForPass>();
}

} // namespace mlir::iree_compiler
