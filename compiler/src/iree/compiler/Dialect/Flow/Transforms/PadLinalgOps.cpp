// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

static bool isSimpleTranspose(linalg::GenericOp op) {
  if (!op) return false;
  if (op.getNumDpsInputs() != 1) return false;
  if (op.getNumDpsInits() != 1) return false;
  if (!op.hasTensorSemantics()) return false;
  if (op.getNumReductionLoops() > 0) return false;
  auto inputOperand = op.getDpsInputOperand(0);
  auto inputIndexMap = op.getMatchingIndexingMap(inputOperand);
  if (!inputIndexMap.isPermutation() || inputIndexMap.isIdentity())
    return false;
  auto outputOperand = op.getDpsInitOperand(0);
  auto outputIndexingMap = op.getMatchingIndexingMap(outputOperand);
  if (!outputIndexingMap.isIdentity()) return false;
  return true;
}
namespace {
/// A pattern to switch transpose and pad with pad and transpose when the
/// tranpose output has an unaligned leading dimension.
struct TransposePadToPadTransposeOp : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  TransposePadToPadTransposeOp(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!padOp.hasZeroLowPad())
      return rewriter.notifyMatchFailure(padOp, "expected zero low pad");

    auto genericOp = padOp.getSource().getDefiningOp<linalg::GenericOp>();
    if (!isSimpleTranspose(genericOp)) {
      return rewriter.notifyMatchFailure(
          padOp, "expected source to be a simple transpose op");
    }

    // Create a new PadOp.

    // Apply reverse transpose to get the low/high paddings and the new shape.
    OpOperand *transposeInput = genericOp.getDpsInputOperand(0);
    AffineMap indexingMap = genericOp.getMatchingIndexingMap(transposeInput);

    auto oldHiPad = padOp.getMixedHighPad();
    SmallVector<OpFoldResult> newHiPad(oldHiPad);
    RankedTensorType oldPadType = padOp.getResultType();
    ArrayRef<int64_t> oldPadShape = oldPadType.getShape();
    SmallVector<int64_t> newShape(oldPadShape);

    for (auto en : enumerate(indexingMap.getResults())) {
      unsigned pos = en.value().cast<AffineDimExpr>().getPosition();
      unsigned index = en.index();
      newHiPad[pos] = oldHiPad[index];
      newShape[pos] = oldPadShape[index];
    }
    auto newPadResultType =
        RankedTensorType::get(newShape, oldPadType.getElementType());
    auto newPadOp = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), newPadResultType, transposeInput->get(),
        padOp.getMixedLowPad(), newHiPad, padOp.getConstantPaddingValue());

    // Reuse the old PadOp for the init operand for the transpose.
    auto newPadOpForInit = rewriter.create<tensor::PadOp>(
        padOp.getLoc(), padOp.getResultType(),
        genericOp.getDpsInitOperand(0)->get(), padOp.getMixedLowPad(),
        padOp.getMixedHighPad(), padOp.getConstantPaddingValue());

    newPadOpForInit.setOperand(0, genericOp.getDpsInitOperand(0)->get());

    auto newTranspose = rewriter.create<linalg::GenericOp>(
        padOp.getLoc(), padOp.getResultType(), newPadOp->getResult(0),
        newPadOpForInit->getResult(0), genericOp.getIndexingMapsArray(),
        genericOp.getIteratorTypesArray(),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
    rewriter.inlineRegionBefore(genericOp.getRegion(), newTranspose.getRegion(),
                                newTranspose.getRegion().begin());
    rewriter.replaceOp(padOp, newTranspose->getResult(0));
    return success();
  }
};

/// A pattern to pad statically shaped matmul operands to the next integer
/// multiple of padSize.
class PadMatmulOp : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  PadMatmulOp(MLIRContext *context, int size, PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern(context, benefit), paddingSize(size) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    Operation *op = linalgOp.getOperation();
    const bool isBatchMatmul = isa<linalg::BatchMatmulOp>(op);
    const bool isMatmul = isa<linalg::MatmulOp>(op);
    if (!isBatchMatmul && !isMatmul) return failure();

    Location loc = linalgOp.getLoc();
    Value lhs = linalgOp.getDpsInputOperand(0)->get();
    Value rhs = linalgOp.getDpsInputOperand(1)->get();
    Value result = linalgOp.getDpsInitOperand(0)->get();

    auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();
    auto resultType = result.getType().dyn_cast<RankedTensorType>();

    if (!lhsType || !rhsType) return failure();

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return failure();

    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    const int B = isBatchMatmul ? lhsShape[0] : -1;
    const int M = isBatchMatmul ? lhsShape[1] : lhsShape[0];
    const int K = lhsShape.back(), N = rhsShape.back();

    int newMSize = std::ceil(float(M) / paddingSize) * paddingSize;
    int newNSize = std::ceil(float(N) / paddingSize) * paddingSize;
    int newKSize = std::ceil(float(K) / paddingSize) * paddingSize;

    int paddingForM = newMSize - M;
    int paddingForN = newNSize - N;
    int paddingForK = newKSize - K;

    if (paddingForM == 0 && paddingForN == 0 && paddingForK == 0)
      return failure();

    auto getFullShape = [&](ArrayRef<int> dims) {
      SmallVector<int64_t, 3> shape;
      if (isBatchMatmul) shape.push_back(B);
      llvm::append_range(shape, dims);
      return shape;
    };

    auto lhsPaddedType = RankedTensorType::get(
        getFullShape({newMSize, newKSize}), lhsType.getElementType());

    auto rhsPaddedType = RankedTensorType::get(
        getFullShape({newKSize, newNSize}), rhsType.getElementType());

    Value lhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(lhsType.getElementType()));

    Value rhsPaddingValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rhsType.getElementType()));

    auto createPadding = [&](ArrayRef<int64_t> padding) {
      SmallVector<OpFoldResult> result;
      if (isBatchMatmul) {
        result.push_back(rewriter.getI64IntegerAttr(0));
      }
      for (auto pad : padding) {
        result.push_back(rewriter.getI64IntegerAttr(pad));
      }
      return result;
    };

    Value paddedLhs = lhs;
    if (paddingForM > 0 || paddingForK > 0) {
      paddedLhs = rewriter.create<tensor::PadOp>(
          loc, lhsPaddedType, lhs, createPadding({0, 0}),
          createPadding({paddingForM, paddingForK}), lhsPaddingValue);
    }

    Value paddedRhs = rhs;
    if (paddingForK > 0 || paddingForN > 0) {
      paddedRhs = rewriter.create<tensor::PadOp>(
          loc, rhsPaddedType, rhs, createPadding({0, 0}),
          createPadding({paddingForK, paddingForN}), rhsPaddingValue);
    }

    // Padding for K-dim doesn't change result size.
    if (paddingForM == 0 && paddingForN == 0) {
      auto paddedMatmulOp =
          linalgOp.clone(rewriter, loc, {resultType},
                         ArrayRef<Value>{paddedLhs, paddedRhs, result});
      rewriter.replaceOp(linalgOp, paddedMatmulOp->getResults());
    } else {
      auto newResultType = RankedTensorType::get(
          getFullShape({newMSize, newNSize}), resultType.getElementType());
      Value resultPaddingValue = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(resultType.getElementType()));
      Value paddedResult = rewriter.create<tensor::PadOp>(
          loc, newResultType, result, createPadding({0, 0}),
          createPadding({paddingForM, paddingForN}), resultPaddingValue);
      auto paddedMatmulOp =
          linalgOp.clone(rewriter, loc, {newResultType},
                         ArrayRef<Value>{paddedLhs, paddedRhs, paddedResult});

      auto zero = rewriter.getI64IntegerAttr(0);
      auto one = rewriter.getI64IntegerAttr(1);
      auto mAttr = rewriter.getIndexAttr(M);
      auto nAttr = rewriter.getIndexAttr(N);
      SmallVector<OpFoldResult> offsets, strides, sizes;
      if (isBatchMatmul) {
        offsets.assign(3, zero);
        strides.assign(3, one);
        sizes = {rewriter.getIndexAttr(B), mAttr, nAttr};
      } else {
        offsets.assign(2, zero);
        strides.assign(2, one);
        sizes = {mAttr, nAttr};
      }
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          linalgOp, paddedMatmulOp->getResults()[0], offsets, sizes, strides);
    }

    return success();
  }

 private:
  int paddingSize;
};

class PadLinalgOpsPass : public PadLinalgOpsBase<PadLinalgOpsPass> {
 public:
  PadLinalgOpsPass(int size) : paddingSize(size) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PadMatmulOp>(context, paddingSize);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    patterns.clear();
    patterns.insert<TransposePadToPadTransposeOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

 private:
  int paddingSize;
};

}  // namespace

std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass(int paddingSize) {
  return std::make_unique<PadLinalgOpsPass>(paddingSize);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
