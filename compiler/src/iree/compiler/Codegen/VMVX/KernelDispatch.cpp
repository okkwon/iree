// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/VMVX/KernelDispatch.h"

#include "iree/compiler/Codegen/Utils/CPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "vmvx-kernel-dispatch"

namespace mlir::iree_compiler {

constexpr int kDefaultDistTileSize = 64;

static llvm::cl::opt<int> clNumberOfRuntimeThreads(
    "iree-vmvx-number-of-threads",
    llvm::cl::desc(
        "number of threads that are used to determine the tile sizes"),
    llvm::cl::init(4));

/// Returns true if the genericOp is elementwise with a single output with
/// the identity indexing map.
static bool isElementWiseIdentity(linalg::GenericOp genericOp) {
  return genericOp.getNumDpsInputs() >= 1 && genericOp.getNumDpsInits() == 1 &&
         linalg::isElementwise(genericOp) &&
         llvm::all_of(genericOp.getIndexingMapsArray(),
                      [](AffineMap map) { return map.isIdentity(); });
}

static SmallVector<int64_t> getDefaultDistributionTileSizes(TilingInterface op,
                                                            int tileSize) {
  unsigned numLoops = op.getLoopIteratorTypes().size();
  auto partitionedLoops = cast<PartitionableLoopsInterface>(op.getOperation())
                              .getPartitionableLoops(kNumMaxParallelDims);
  SmallVector<int64_t> distTileSizes(numLoops, tileSize);
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto dim : llvm::seq<int64_t>(0, distTileSizes.size())) {
    if (!partitionedLoopsSet.count(dim))
      distTileSizes[dim] = 0;
  }

  return distTileSizes;
}

// TODO: move to a utility file

static FailureOr<Operation *> getUnaryOp(linalg::GenericOp op) {
  auto &children = op.getBlock()->getOperations();
  // Only match two children (op + yield).
  if (children.size() != 2)
    return failure();
  // Only match parallel loops.
  if (op.getNumParallelLoops() != op.getNumLoops())
    return failure();

  // Match:
  //   %0 = someop %arg2
  //   yield %0
  Operation *scalarOp = &children.front();
  Operation *yieldOp = op.getBlock()->getTerminator();
  if (scalarOp->getNumOperands() != 1 || yieldOp->getNumOperands() != 1 ||
      yieldOp->getOperand(0) != scalarOp->getResult(0)) {
    return failure();
  }
  BlockArgument operandScalar0 =
      llvm::dyn_cast<BlockArgument>(scalarOp->getOperands()[0]);
  if (!operandScalar0)
    return failure();

  return scalarOp;
}

static FailureOr<Operation *> getBinaryOp(linalg::GenericOp op) {
  auto &children = op.getBlock()->getOperations();
  // Only match two children (op + yield).
  if (children.size() != 2)
    return failure();
  // Only match parallel loops.
  if (op.getNumParallelLoops() != op.getNumLoops())
    return failure();

  // Match:
  //   %0 = someop %arg2
  //   yield %0
  Operation *scalarOp = &children.front();
  Operation *yieldOp = op.getBlock()->getTerminator();
  if (scalarOp->getNumOperands() != 2 || yieldOp->getNumOperands() != 1 ||
      yieldOp->getOperand(0) != scalarOp->getResult(0)) {
    return failure();
  }
  BlockArgument operandScalar0 =
      llvm::dyn_cast<BlockArgument>(scalarOp->getOperands()[0]);
  if (!operandScalar0)
    return failure();

  return scalarOp;
}

static FailureOr<Operation *> getUnaryOrBinaryOp(linalg::GenericOp op) {
  auto result = getUnaryOp(op);
  if (succeeded(result))
    return result;

  return getBinaryOp(op);
}

static bool isTanhf(linalg::GenericOp op) {
  auto result = getUnaryOp(op);
  if (failed(result)) {
    return false;
  }
  Operation *scalarOp = *result;
  if (!isa<math::TanhOp>(scalarOp))
    return false;

  Type resultType = scalarOp->getResult(0).getType();
  return resultType.isF32();
}

static int64_t getTileSize(linalg::GenericOp op) {
  auto result = getUnaryOrBinaryOp(op);
  if (failed(result))
    return kDefaultDistTileSize;

  Operation *scalarOp = *result;
  auto resultTensorType =
      dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTensorType)
    return kDefaultDistTileSize;

  if (!resultTensorType.hasStaticShape())
    return kDefaultDistTileSize;

  // FIXME: support 2D shapes too
  // This is a bit complicated since it needs a reshape in some cases.
  // In general, what we need is not the tile size. We need the workload per
  // thread.
  if (resultTensorType.getShape().size() != 1) {
    return kDefaultDistTileSize;
  }

  Type elemType = resultTensorType.getElementType();

  int64_t tileSize =
      TypeSwitch<Operation *, int64_t>(scalarOp)
          .Case<arith::AddFOp, arith::MulFOp, math::TanhOp>(
              [&](Operation *) -> int64_t {
                if (elemType.isF32()) {
                  // TODO: this is target depedent, need to use the target info.
                  const int64_t minTileSize = 32;

                  int64_t numElems = resultTensorType.getNumElements();
                  int64_t tileSize = numElems / clNumberOfRuntimeThreads;
                  tileSize = llvm::alignTo(tileSize, minTileSize);
                  return tileSize;
                }
                return kDefaultDistTileSize;
              })
          .Default([](Operation *) { return kDefaultDistTileSize; });
  return tileSize;
}

/// Sets the lowering configuration for dispatch region for linalg_ext.fft
/// root op.
static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   IREE::LinalgExt::FftOp fftOp) {
  assert(!getLoweringConfig(fftOp) && "expected lowering_config is not set");
  SmallVector<int64_t> distTileSizes =
      getDefaultDistributionTileSizes(fftOp, kDefaultDistTileSize);
  auto rank = fftOp.getOperandRank();
  if (distTileSizes.size() >= rank && distTileSizes[rank - 1] != 0) {
    APInt value;
    if (matchPattern(fftOp.getStage(), m_ConstantInt(&value))) {
      distTileSizes[rank - 1] = 1ll << value.getSExtValue();
      distTileSizes[rank - 1] = std::max(
          distTileSizes[rank - 1], static_cast<int64_t>(kDefaultDistTileSize));
    } else {
      return fftOp.emitOpError("non-constant stage might not work for fft op");
    }
  }
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, fftOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
}

LogicalResult tryElementwiseRootConfig(mlir::FunctionOpInterface entryPointFn,
                                       linalg::GenericOp op) {
  if (!isElementWiseIdentity(op)) {
    return failure();
  }

  // Try to distribute the work from the outermost dimension by the number
  // of threads.
  Type resultType = op.getDpsInitOperand(0)->get().getType();
  if (auto rankedType = dyn_cast<RankedTensorType>(resultType)) {
    if (rankedType.hasStaticShape()) {
      SmallVector<int64_t> shape(rankedType.getShape());
      int64_t numThreads = clNumberOfRuntimeThreads;

      // If the work is too small, process it by a single thread.
      if (rankedType.getNumElements() >= 32) {
        for (int dim = 0; dim < shape.size();) {
          if (numThreads == 1)
            break;
          if (shape[dim] > 1) {
            shape[dim] = shape[dim] / 2;
            numThreads = numThreads / 2;
          } else {
            dim++;
          }
        }
      }
      TileSizesListType tileSizes = {shape};
      return setOpConfigAndEntryPointFnTranslation(
          entryPointFn, op, tileSizes,
          IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
    }
  }
  return failure();
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   linalg::GenericOp op) {
  assert(!getLoweringConfig(op) && "expected lowering_config is not set");

  // Handle elementwise ops by distributing the work by the number of threads.
  if (succeeded(tryElementwiseRootConfig(entryPointFn, op)))
    return success();

  // Check if it is an elementwise tanh, which uses the xnnpack tanh
  // microkernel. The logic is to get the minimum tile size for the op and
  // divide the workload by the number of threads to get the per-thread
  // workload. The per-thread workload should be a multiple of the minimum tile
  // size, e.g., 32, for tanh for avx2+.

  // TODO: Need a better way to represent the op itself using stablehlo or
  // linalg named ops.

  // TODO: For VMVX + Ukernel, each dispatch tends to have a single operation,
  // which makes the tile size selection per op.
  int64_t tileSize = getTileSize(op);

  SmallVector<int64_t> distTileSizes = getDefaultDistributionTileSizes(
      cast<TilingInterface>(op.getOperation()), tileSize);
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
}

static LogicalResult setRootConfig(mlir::FunctionOpInterface entryPointFn,
                                   TilingInterface tilingInterfaceOp) {
  assert(!getLoweringConfig(tilingInterfaceOp) &&
         "expected lowering_config is not set");

  SmallVector<int64_t> distTileSizes =
      getDefaultDistributionTileSizes(tilingInterfaceOp, kDefaultDistTileSize);
  TileSizesListType tileSizes = {distTileSizes};
  return setOpConfigAndEntryPointFnTranslation(
      entryPointFn, tilingInterfaceOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
}

static LogicalResult
setVMVXRootConfigImpl(mlir::FunctionOpInterface entryPointFn, Operation *op) {
  auto setRootConfigFn = [&](Operation *op) -> LogicalResult {
    return TypeSwitch<Operation *, LogicalResult>(op)
        .Case<linalg::GenericOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<IREE::LinalgExt::FftOp>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Case<TilingInterface>(
            [&](auto op) { return setRootConfig(entryPointFn, op); })
        .Default([&](Operation *op) { return success(); });
  };
  return setRootConfigFn(op);
}

static LogicalResult
lowerUsingVMVXDefaultPipeline(mlir::FunctionOpInterface op) {
  auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
      op.getContext(),
      IREE::Codegen::DispatchLoweringPassPipeline::VMVXDefault);
  return setTranslationInfo(op, translationInfo);
}

/// Sets the translation information to use for a dispatch region.
static LogicalResult
setConfigForKernel(mlir::FunctionOpInterface entryPointFn) {
  SmallVector<Operation *> computeOps = getComputeOps(entryPointFn);
  if (computeOps.empty()) {
    return lowerUsingVMVXDefaultPipeline(entryPointFn);
  }

  FailureOr<Operation *> rootOp = getRootOperation(computeOps);
  if (failed(rootOp)) {
    return failure();
  }

  // Handle the case with no known root operation.
  Operation *rootOperation = rootOp.value();
  if (!rootOperation) {
    return lowerUsingVMVXDefaultPipeline(entryPointFn);
  }

  if (failed(setVMVXRootConfigImpl(entryPointFn, rootOperation))) {
    return failure();
  }

  return success();
}

LogicalResult initVMVXLaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) {
      continue;
    }

    if (getTranslationInfo(exportOp)) {
      continue;
    }

    if (failed(setConfigForKernel(funcOp))) {
      return failure();
    }
  }

  return success();
}

} // namespace mlir::iree_compiler
