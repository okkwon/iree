// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Transforms.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Common/GPUPatterns.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TilingUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using mlir::iree_compiler::IREE::LinalgExt::TilingPatterns;

#define DEBUG_TYPE "iree-llvmgpu-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

/// Patterns for workgroup level tiling. Workgroup tiling is done at the flow
/// level but we may have extra tiling for the reduction dimension. Therefore we
/// tile again without distributing.
static void populateTilingReductionPatterns(RewritePatternSet &patterns) {
  auto tileSizesFn = [&](OpBuilder &builder,
                         Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(tileSizesFn);
  MLIRContext *context = patterns.getContext();

  IREE::LinalgExt::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getWorkgroupMemoryMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));
  filter.setMatchByDefault();
  TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp,
                 linalg::GenericOp>::insert(patterns, tilingOptions, filter);
}

/// Return the tile size associated to one thread or warp based on the number of
/// element in the group.
static SmallVector<Value, 4> calculateDistributedTileSize(
    ArrayRef<int64_t> numElements, OpBuilder &builder, Operation *operation) {
  SmallVector<int64_t> blockTileSize = getTileSizes(operation, 0);
  SmallVector<Value, 4> tileSizesVal;
  // Use partitionedLoop to know what loop needs to be distributed.
  auto interfaceOp = cast<PartitionableLoopsInterface>(operation);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    return tileSizesVal;
  }
  auto zero = builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0);
  tileSizesVal.resize(
      cast<TilingInterface>(operation).getLoopIteratorTypes().size(), zero);

  // partitionedLoops contains the dimensions we want to distribute.
  // We are distributing them in order onto the different workgroup
  // dimensions.
  SmallVector<int64_t> distributedDim(numElements.begin(), numElements.end());
  distributedDim.resize(partitionedLoops.size());
  unsigned idIdx = 0;
  std::reverse(distributedDim.begin(), distributedDim.end());
  for (unsigned depth : partitionedLoops) {
    if (depth >= blockTileSize.size()) continue;
    tileSizesVal[depth] = builder.create<arith::ConstantIndexOp>(
        operation->getLoc(),
        llvm::divideCeil(blockTileSize[depth], distributedDim[idIdx++]));
    if (idIdx == kNumMaxParallelDims) break;
  }
  return tileSizesVal;
}

/// Patterns for warp level tiling.
static void populateTilingToWarpPatterns(
    RewritePatternSet &patterns, SmallVectorImpl<int64_t> &workgroupSize) {
  std::array<int64_t, 3> warpPerWorkgroup = {
      workgroupSize[0] / kWarpSize, workgroupSize[1], workgroupSize[2]};

  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [warpPerWorkgroup](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(warpPerWorkgroup, builder,
                                            operation);
      };
  auto getWarpProcInfoFn = [warpPerWorkgroup](
                               OpBuilder &builder, Location loc,
                               ArrayRef<Range> parallelLoopRanges) {
    return getSubgroupIdsAndCounts(builder, loc, kWarpSize,
                                   parallelLoopRanges.size(), warpPerWorkgroup);
  };

  linalg::LinalgLoopDistributionOptions warpDistributionOptions;
  warpDistributionOptions.procInfo = getWarpProcInfoFn;

  auto tilingOptions = linalg::LinalgTilingOptions()
                           .setLoopType(linalg::LinalgTilingLoopType::Loops)
                           .setTileSizeComputationFunction(getInnerTileSizeFn)
                           .setDistributionOptions(warpDistributionOptions);
  MLIRContext *context = patterns.getContext();
  IREE::LinalgExt::LinalgTransformationFilter filter(
      {StringAttr::get(context, getGPUWarpLevelTilingReqMarker())},
      StringAttr::get(context, getVectorizeForTensorCoreMarker()));
  TilingPatterns<linalg::MatmulOp, linalg::FillOp, linalg::BatchMatmulOp,
                 linalg::GenericOp>::insert(patterns, tilingOptions, filter);
}

using FilterFunction = std::function<LogicalResult(Operation *)>;

/// Patterns for thread level tiling.
static void populateTilingToInvocationPatterns(
    RewritePatternSet &patterns, SmallVectorImpl<int64_t> &workgroupSize,
    bool matchByDefault = true) {
  linalg::TileSizeComputationFunction getInnerTileSizeFn =
      [&](OpBuilder &builder, Operation *operation) {
        return calculateDistributedTileSize(workgroupSize, builder, operation);
      };
  auto getThreadProcInfoFn = [&workgroupSize](
                                 OpBuilder &builder, Location loc,
                                 ArrayRef<Range> parallelLoopRanges) {
    return getGPUThreadIdsAndCounts(builder, loc, parallelLoopRanges.size(),
                                    workgroupSize);
  };
  linalg::LinalgLoopDistributionOptions invocationDistributionOptions;
  invocationDistributionOptions.procInfo = getThreadProcInfoFn;

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(getInnerTileSizeFn)
          .setDistributionOptions(invocationDistributionOptions);

  MLIRContext *context = patterns.getContext();
  IREE::LinalgExt::LinalgTransformationFilter f(
      {StringAttr::get(context, getWorkgroupKTiledMarker()),
       StringAttr::get(context, getWorkgroupMemoryMarker()),
       StringAttr::get(context, getGPUSimtLoweringReqMarker())},
      StringAttr::get(context, getVectorizeMarker()));
  f.addFilter([](Operation *op) {
    // FFT doesn't support second level of tiling yet.
    return success(!isa<IREE::LinalgExt::FftOp>(op));
  });
  if (matchByDefault) f.setMatchByDefault();
  patterns.insert<IREE::LinalgExt::LinalgTilingPattern,
                  IREE::LinalgExt::TilingInterfaceTilingPattern>(
      context, tilingOptions, f);
}

static void markCandidates(func::FuncOp funcOp) {
  funcOp.walk([](linalg::LinalgOp op) {
    if (!isa<linalg::BatchMatmulOp, linalg::MatmulOp>(op))
      return WalkResult::skip();

    if (succeeded(alignedOpFilter(op))) {
      setMarker(op, getGPUTensorCoreLoweringReqMarker());
    } else {
      setMarker(op, getGPUSimtLoweringReqMarker());
    }
    return WalkResult::advance();
  });
}

static LogicalResult tileTensorCoreKDim(func::FuncOp funcOp) {
  // mark which linarg op is a tensorcore
  markCandidates(funcOp);

  auto context = funcOp.getContext();
  RewritePatternSet patterns(context);
  auto tileSizesFn = [](OpBuilder &builder,
                        Operation *op) -> SmallVector<Value, 4> {
    auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
    auto partitionedLoops =
        interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
    SmallVector<Value, 4> tileSizes = getTileSizes(builder, op, 0);
    auto zero = builder.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for (unsigned depth : partitionedLoops) {
      if (depth < tileSizes.size()) {
        tileSizes[depth] = zero;
      }
    }
    return tileSizes;
  };

  auto tilingOptions =
      linalg::LinalgTilingOptions()
          .setLoopType(linalg::LinalgTilingLoopType::Loops)
          .setTileSizeComputationFunction(tileSizesFn)
          .setPeeledLoops({0});  // peel off the partial iterations

  IREE::LinalgExt::LinalgTransformationFilter filter(
      ArrayRef<StringAttr>{
          StringAttr::get(context, getGPUTensorCoreLoweringReqMarker())},
      StringAttr::get(context, getWorkgroupKTiledMarker()));

  TilingPatterns<linalg::MatmulOp, linalg::BatchMatmulOp>::insert(
      patterns, tilingOptions, filter);

  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }

  RewritePatternSet wgTilingCanonicalizationPatterns =
      linalg::getLinalgTilingCanonicalizationPatterns(funcOp.getContext());
  populateAffineMinSCFCanonicalizationPattern(wgTilingCanonicalizationPatterns);
  scf::populateSCFForLoopCanonicalizationPatterns(
      wgTilingCanonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(wgTilingCanonicalizationPatterns)))) {
    return failure();
  }

  return success();
}

// Get K dimension size. It returns kDynamicSize for unknown cases.
static int64_t getSizeK(linalg::LinalgOp op) {
  int64_t sizeK = ShapedType::kDynamicSize;

  if (!isa<linalg::BatchMatmulOp, linalg::MatmulOp>(op)) return sizeK;

  auto lhsShape =
      op.getDpsInputOperand(0)->get().getType().cast<ShapedType>().getShape();
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getDpsInputOperand(0))
              .getDimPosition(i) == exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  return sizeK;
}

namespace {
struct LLVMGPUTileAndDistributePass
    : public LLVMGPUTileAndDistributeBase<LLVMGPUTileAndDistributePass> {
 private:
  // Distribute the workloads to warp if true otherwise distribute to threads.
  bool distributeToWarp = false;

 public:
  LLVMGPUTileAndDistributePass(bool distributeToWarp)
      : distributeToWarp(distributeToWarp) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    if (!isEntryPoint(funcOp)) return;

    // Promote C matrix and propagate the potential fill producer into the temp
    // allocation. This needs to be done before reduction tiling.
    {
      RewritePatternSet promotionPatterns(&getContext());
      populateContractPromotionPatterns(promotionPatterns, {2});
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      LLVM_DEBUG({
        llvm::dbgs() << "After promote C:\n";
        funcOp.dump();
      });

      propagateSharedMemoryCopy(funcOp);

      LLVM_DEBUG({
        llvm::dbgs() << "After propagateSharedMemoryCopy():\n";
        funcOp.dump();
      });
    }

    // Tile again at the workgroup level since reduction dimension were
    // ignored. Dimensions already tiled will be ignore since we tile to the
    // same size. For distributing to warps, peel the partial iterations as
    // a separate loop, since the warp distribution is requested for wmma.
    if (failed(tileToSerialLoops(funcOp, /*peel=*/distributeToWarp))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile reductions:";
      funcOp.dump();
    });

    auto workgroupSize = llvm::to_vector<4>(llvm::map_range(
        getEntryPoint(funcOp)->getWorkgroupSize().value(),
        [&](Attribute attr) { return attr.cast<IntegerAttr>().getInt(); }));

    int64_t flatWorkgroupSize =
        workgroupSize[0] * workgroupSize[1] * workgroupSize[2];
    // Only promote to workgroup size if there are multiple warps.
    if (flatWorkgroupSize > kWarpSize) {
      RewritePatternSet promotionPatterns(&getContext());

      populateContractPromotionPatterns(promotionPatterns, {0, 1});
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(promotionPatterns)))) {
        return signalPassFailure();
      }
      // Insert barriers before and after copies to workgroup memory.
      insertBarriersAroundSharedMemoryCopy(funcOp);
    }

    {
      RewritePatternSet promotionCanonicalization =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(promotionCanonicalization)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After promotion:";
      funcOp.dump();
    });

    if (distributeToWarp) {
      // mark candidates for the warp level tiling
      funcOp.walk([&](linalg::LinalgOp op) {
        if (failed(alignedOpFilter(op))) return WalkResult::skip();
        if (!isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::FillOp,
                 linalg::GenericOp>(op))
          return WalkResult::skip();

        // check if K is a multiple of Tile-K.
        int64_t sizeK = getSizeK(op);
        if (sizeK != ShapedType::kDynamicSize) {
          // WG tile sizes
          auto wgTileSizes = getTileSizes(op, 0);

          if (sizeK % wgTileSizes[wgTileSizes.size() - 1] != 0)
            return WalkResult::skip();
        }

        setMarker(op, getGPUWarpLevelTilingReqMarker());
        return WalkResult::advance();
      });

      // Apply last level of tiling and distribute to warps for aligned ops.
      RewritePatternSet warpLevelTilingPatterns(context);
      populateTilingToWarpPatterns(warpLevelTilingPatterns, workgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(warpLevelTilingPatterns)))) {
        return signalPassFailure();
      }

      // Apply last level of tiling and distribute to threads for unaligned ops.
      RewritePatternSet threadLevelTilingPatterns(context);
      populateTilingToInvocationPatterns(threadLevelTilingPatterns,
                                         workgroupSize,
                                         /*matchByDefault=*/false);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadLevelTilingPatterns)))) {
        return signalPassFailure();
      }
    } else {
      // Apply last level of tiling and distribute to threads.
      RewritePatternSet threadLevelTilingPatterns(context);
      populateTilingToInvocationPatterns(threadLevelTilingPatterns,
                                         workgroupSize);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadLevelTilingPatterns)))) {
        return signalPassFailure();
      }
    }
    {
      // Apply canonicalization patterns.
      RewritePatternSet threadTilingCanonicalizationPatterns =
          linalg::getLinalgTilingCanonicalizationPatterns(context);
      populateAffineMinSCFCanonicalizationPattern(
          threadTilingCanonicalizationPatterns);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(threadTilingCanonicalizationPatterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After tile and distribute to threads:";
      funcOp.dump();
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileAndDistribute(
    bool distributeToWarp) {
  return std::make_unique<LLVMGPUTileAndDistributePass>(distributeToWarp);
}

}  // namespace iree_compiler
}  // namespace mlir
