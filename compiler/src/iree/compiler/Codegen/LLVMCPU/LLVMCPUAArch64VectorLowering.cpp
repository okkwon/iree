// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/Utils/InferCustomKernelsTargetInfoFromParent.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-llvmcpu-aarch64-vector-lowering"

// A flag to switch between inline asm and intrinsics while we develop these two
// parallel paths.
static llvm::cl::opt<bool> clMmt4dUseIntrinsics(
    "iree-codegen-mmt4d-use-intrinsics",
    llvm::cl::desc("Whether to use instrinsics when lowering vector contracts "
                   "generated from mmt4d matmuls (as opposed to inline asm). "
                   "Not for production use."),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {

namespace {
struct LLVMCPUAArch64VectorLoweringPass
    : public LLVMCPUAArch64VectorLoweringBase<
          LLVMCPUAArch64VectorLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void LLVMCPUAArch64VectorLoweringPass::runOnOperation() {
  MLIRContext *context = &getContext();
  auto funcOp = getOperation();

  Optional<int64_t> numLoops;
  funcOp.walk([&](vector::ContractionOp op) {
    if (!numLoops) {
      numLoops = op.getIndexingMapsArray()[0].getNumDims();
    } else {
      if (*numLoops != op.getIndexingMapsArray()[0].getNumDims())
        return signalPassFailure();
    }
  });
  // No vector.contract op to optimize.
  if (!numLoops) return;

  {
    // Fold consumer add ops into the contraction op itself.
    RewritePatternSet canonicalizationPatterns(context);
    vector::ContractionOp::getCanonicalizationPatterns(canonicalizationPatterns,
                                                       context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(canonicalizationPatterns)))) {
      return signalPassFailure();
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs()
          << "\n--- After folding consumer add ops into contraction op "
             "iteself ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  {
    RewritePatternSet castAwayUnitDimPatterns(&getContext());
    vector::populateCastAwayVectorLeadingOneDimPatterns(
        castAwayUnitDimPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(castAwayUnitDimPatterns)))) {
      return signalPassFailure();
    }

    RewritePatternSet reductionToContractPatterns(&getContext());
    vector::populateVectorReductionToContractPatterns(
        reductionToContractPatterns);
    vector::ExtractOp::getCanonicalizationPatterns(reductionToContractPatterns,
                                                   context);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(reductionToContractPatterns)))) {
      return signalPassFailure();
    }
  }

  {
    // Special-case vector.contract codegen paths. This needs to happen
    // just before the generic vector ops lowerings.
    CustomKernelsTargetInfo info;
    if (succeeded(InferCustomKernelsTargetInfoFromParent(funcOp, info))) {
      if (clMmt4dUseIntrinsics) {
        info.add(CustomKernelTargetFeature::Intrinsics);
      }
      RewritePatternSet patterns(context);
      populateVectorContractCustomKernelsPatterns(info, patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }

  // Apply vector unroll
  {
    RewritePatternSet vectorUnrollPatterns(context);
    // There are issues when unrolling 1Dx1D->0D vector.contract op. Only unroll
    // the op when there are more than one loop.
    constexpr int64_t kVectorSize = 4;
    SmallVector<int64_t> vectorTiles(numLoops.value(), kVectorSize);
    if (numLoops.value() > 1) {
      vector::populateVectorUnrollPatterns(
          vectorUnrollPatterns,
          vector::UnrollVectorOptions().setNativeShape(vectorTiles));
    }

    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(vectorUnrollPatterns)))) {
      return signalPassFailure();
    }
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After vector unrolling ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

  // Apply vector specific operation lowering.
  // TODO(hanchung): Have better control for choosing vector unrolling sizes.
  // The lowering config is destroyed when lowering the op to vector ops. This
  // can be fixed when moving to use transform dialect for scheduling. Because
  // we still have the config when scheduling transforms.
  {
    vector::VectorTransformsOptions vectorTransformsOptions =
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct);
    RewritePatternSet vectorContractLoweringPatterns(&getContext());
    vectorContractLoweringPatterns.insert<
        vector::ContractionOpToOuterProductOpLowering,
        vector::ContractionOpToMatmulOpLowering, vector::ContractionOpLowering>(
        vectorTransformsOptions, context);
    vector::populateVectorTransferPermutationMapLoweringPatterns(
        vectorContractLoweringPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorContractLoweringPatterns)))) {
      return signalPassFailure();
    }

    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "\n--- After vector specific operatrion lowering ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUAArch64VectorLoweringPass() {
  return std::make_unique<LLVMCPUAArch64VectorLoweringPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
