// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/RemarkUtils.h"

#include "llvm/Support/FormatVariadic.h"

namespace mlir::iree_compiler::RemarkUtils {

remark::RemarkOpts createSerializationRemark(llvm::StringRef name,
                                             llvm::StringRef subCategory,
                                             llvm::StringRef functionName) {
  return remark::RemarkOpts::name(name)
      .category(RemarkCategories::Serialization)
      .subCategory(subCategory)
      .function(functionName);
}

remark::RemarkOpts createResourceUsageRemark(llvm::StringRef name,
                                             llvm::StringRef subCategory,
                                             llvm::StringRef functionName) {
  return remark::RemarkOpts::name(name)
      .category(RemarkCategories::ResourceUsage)
      .subCategory(subCategory)
      .function(functionName);
}

remark::RemarkOpts createOptimizationRemark(llvm::StringRef name,
                                            llvm::StringRef subCategory,
                                            llvm::StringRef functionName) {
  return remark::RemarkOpts::name(name)
      .category(RemarkCategories::OptimizationDecisions)
      .subCategory(subCategory)
      .function(functionName);
}

remark::RemarkOpts createHALRemark(llvm::StringRef name,
                                   llvm::StringRef subCategory,
                                   llvm::StringRef functionName) {
  return remark::RemarkOpts::name(name)
      .category(RemarkCategories::HAL)
      .subCategory(subCategory)
      .function(functionName);
}

void emitSerializationSuccessRemark(
    Location loc, llvm::StringRef backend, llvm::StringRef functionName,
    size_t binarySize, llvm::StringRef targetArch,
    std::function<void(remark::InFlightRemark &)> additionalMetrics) {
  auto opts =
      createSerializationRemark("SerializationSuccess", backend, functionName);
  auto remarkBuilder = remark::passed(loc, opts);
  remarkBuilder << "Successfully serialized binary"
                << remark::metric("BinarySize", static_cast<int64_t>(binarySize))
                << remark::metric("TargetArch", targetArch.str());
  if (additionalMetrics) {
    additionalMetrics(remarkBuilder);
  }
}

void emitBinaryDumpRemark(Location loc, llvm::StringRef backend,
                          llvm::StringRef functionName,
                          llvm::StringRef dumpPath, llvm::StringRef filename,
                          size_t fileSize, llvm::StringRef format) {
  auto opts =
      createSerializationRemark("BinaryDumped", backend, functionName);
  remark::analysis(loc, opts)
      << "Binary dumped to disk"
      << remark::metric("Path", dumpPath.str())
      << remark::metric("Filename", filename.str())
      << remark::metric("Size", static_cast<int64_t>(fileSize))
      << remark::metric("Format", format.str());
}

void emitRegisterUsageRemark(Location loc, llvm::StringRef backend,
                             llvm::StringRef dispatchName, uint32_t vgprCount,
                             uint32_t sgprCount, uint32_t vgprSpillCount,
                             uint32_t sgprSpillCount,
                             uint64_t sharedMemoryBytes) {
  auto opts = createResourceUsageRemark("RegisterUsage", backend, dispatchName);
  remark::analysis(loc, opts)
      << "Kernel resource usage"
      << remark::metric("VGPRCount", static_cast<int64_t>(vgprCount))
      << remark::metric("SGPRCount", static_cast<int64_t>(sgprCount))
      << remark::metric("VGPRSpillCount", static_cast<int64_t>(vgprSpillCount))
      << remark::metric("SGPRSpillCount", static_cast<int64_t>(sgprSpillCount))
      << remark::metric("SharedMemoryBytes",
                        static_cast<int64_t>(sharedMemoryBytes));
}

void emitRegisterSpillingWarning(Location loc, llvm::StringRef backend,
                                  llvm::StringRef dispatchName,
                                  uint32_t vgprSpillCount,
                                  uint32_t sgprSpillCount,
                                  llvm::StringRef suggestion) {
  auto opts = createResourceUsageRemark("RegisterSpillingDetected", backend,
                                        dispatchName);
  auto remarkBuilder = remark::missed(loc, opts);
  remarkBuilder << remark::reason(
      "Register spilling detected: VGPR={0}, SGPR={1}", vgprSpillCount,
      sgprSpillCount);
  if (!suggestion.empty()) {
    remarkBuilder << remark::suggest(suggestion.str());
  } else {
    remarkBuilder << remark::suggest(
        "Consider reducing workgroup size or kernel complexity");
  }
}

void emitOptimizationPipelineRemark(
    Location loc, llvm::StringRef backend, llvm::StringRef functionName,
    llvm::StringRef optLevel, llvm::StringRef passPipeline,
    std::function<void(remark::InFlightRemark &)> additionalMetrics) {
  auto opts = createOptimizationRemark("OptimizationPipeline", backend,
                                       functionName);
  auto remarkBuilder = remark::analysis(loc, opts);
  remarkBuilder << "Ran optimization pipeline"
                << remark::metric("OptLevel", optLevel.str())
                << remark::metric("PassPipeline", passPipeline.str());
  if (additionalMetrics) {
    additionalMetrics(remarkBuilder);
  }
}

void emitOptimizationSkippedRemark(Location loc, llvm::StringRef backend,
                                   llvm::StringRef functionName,
                                   llvm::StringRef optimizationName,
                                   llvm::StringRef reason,
                                   llvm::StringRef suggestion) {
  auto opts = createOptimizationRemark("OptimizationSkipped", backend,
                                       functionName);
  remark::missed(loc, opts)
      << remark::metric("Optimization", optimizationName.str())
      << remark::reason(reason.str()) << remark::suggest(suggestion.str());
}

void emitSerializationFailureRemark(Location loc, llvm::StringRef backend,
                                    llvm::StringRef functionName,
                                    llvm::StringRef phase,
                                    llvm::StringRef errorMessage) {
  auto opts =
      createSerializationRemark("SerializationFailed", backend, functionName);
  remark::failed(loc, opts) << remark::metric("Phase", phase.str())
                            << remark::reason(errorMessage.str());
}

void emitExecutableSerializationStartRemark(Location loc,
                                            llvm::StringRef executableName,
                                            llvm::StringRef targetName,
                                            int debugLevel,
                                            size_t variantCount) {
  auto opts = createHALRemark("ExecutableSerializationStarted", "Serialization",
                              executableName);
  remark::analysis(loc, opts)
      << "Starting executable serialization"
      << remark::metric("Target", targetName.str())
      << remark::metric("DebugLevel", static_cast<int64_t>(debugLevel))
      << remark::metric("VariantCount", static_cast<int64_t>(variantCount));
}

void emitExecutableSerializationCompleteRemark(Location loc,
                                               llvm::StringRef executableName,
                                               llvm::StringRef targetName,
                                               size_t variantsProcessed) {
  auto opts = createHALRemark("ExecutableSerializationCompleted",
                              "Serialization", executableName);
  remark::passed(loc, opts)
      << "Completed executable serialization"
      << remark::metric("Target", targetName.str())
      << remark::metric("VariantsProcessed",
                        static_cast<int64_t>(variantsProcessed));
}

} // namespace mlir::iree_compiler::RemarkUtils

