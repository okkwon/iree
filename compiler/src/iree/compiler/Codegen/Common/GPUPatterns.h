// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_
#define IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

/// Adds patterns for preparing vector transfer ops for converting to GPU
/// subgroup MMA load/store ops.
void populateVectorTransferToGPUMMAPreparationPatterns(
    RewritePatternSet &patterns);

/// Adds patterns for promoting Linalg contract op's operands to use GPU shared
/// memory.
void populateContractPromotionPatterns(
    RewritePatternSet &patterns, ArrayRef<int64_t> operandsToPromote,
    IREE::LinalgExt::LinalgTransformationFilter *filter = nullptr);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_COMMON_GPUPATTERNS_H_
