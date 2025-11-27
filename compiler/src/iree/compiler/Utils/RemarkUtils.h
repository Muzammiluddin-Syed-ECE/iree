// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_REMARKUTILS_H_
#define IREE_COMPILER_UTILS_REMARKUTILS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Remarks.h"

namespace mlir::iree_compiler {

/// Standard remark categories used throughout IREE compiler.
namespace RemarkCategories {

/// Serialization-related remarks (binary generation, encoding).
constexpr llvm::StringLiteral Serialization = "Serialization";

/// Resource usage analysis (registers, memory, compute).
constexpr llvm::StringLiteral ResourceUsage = "ResourceUsage";

/// Optimization decision explanations.
constexpr llvm::StringLiteral OptimizationDecisions = "OptimizationDecisions";

/// Performance warnings and diagnostics.
constexpr llvm::StringLiteral PerformanceWarnings = "PerformanceWarnings";

/// HAL-related remarks (executable transformation).
constexpr llvm::StringLiteral HAL = "HAL";

/// Target backend specific remarks.
constexpr llvm::StringLiteral TargetBackend = "TargetBackend";

} // namespace RemarkCategories

/// Helper functions for emitting structured remarks in IREE.
namespace RemarkUtils {

/// Creates a standard RemarkOpts for serialization events.
/// \param name The remark name identifier
/// \param subCategory Backend-specific subcategory (e.g., "ROCM", "CUDA")
/// \param functionName The function/executable being processed
remark::RemarkOpts createSerializationRemark(llvm::StringRef name,
                                             llvm::StringRef subCategory,
                                             llvm::StringRef functionName);

/// Creates a standard RemarkOpts for resource usage analysis.
/// \param name The remark name identifier
/// \param subCategory Analysis subcategory (e.g., "RegisterAllocator")
/// \param functionName The function being analyzed
remark::RemarkOpts createResourceUsageRemark(llvm::StringRef name,
                                             llvm::StringRef subCategory,
                                             llvm::StringRef functionName);

/// Creates a standard RemarkOpts for optimization decisions.
/// \param name The remark name identifier
/// \param subCategory Optimization subcategory (e.g., "Vectorization")
/// \param functionName The function being optimized
remark::RemarkOpts createOptimizationRemark(llvm::StringRef name,
                                            llvm::StringRef subCategory,
                                            llvm::StringRef functionName);

/// Creates a standard RemarkOpts for HAL transformation events.
/// \param name The remark name identifier
/// \param subCategory HAL-specific subcategory
/// \param functionName The executable being transformed
remark::RemarkOpts createHALRemark(llvm::StringRef name,
                                   llvm::StringRef subCategory,
                                   llvm::StringRef functionName);

/// Emits a remark for successful binary serialization.
/// \param loc Source location for the remark
/// \param backend Target backend name (e.g., "ROCM", "CUDA")
/// \param functionName Function/executable name
/// \param binarySize Size of serialized binary in bytes
/// \param targetArch Target architecture string
/// \param additionalMetrics Callback to add more metrics
void emitSerializationSuccessRemark(
    Location loc, llvm::StringRef backend, llvm::StringRef functionName,
    size_t binarySize, llvm::StringRef targetArch,
    std::function<void(remark::InFlightRemark &)> additionalMetrics = nullptr);

/// Emits a remark for binary dump to disk.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param functionName Function/executable name
/// \param dumpPath Path where binary was dumped
/// \param filename Name of the dumped file
/// \param fileSize Size of dumped file in bytes
/// \param format Binary format (e.g., "HSACO", "PTX", "SPIR-V")
void emitBinaryDumpRemark(Location loc, llvm::StringRef backend,
                          llvm::StringRef functionName,
                          llvm::StringRef dumpPath, llvm::StringRef filename,
                          size_t fileSize, llvm::StringRef format);

/// Emits a remark for register usage analysis.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param dispatchName Dispatch/kernel name
/// \param vgprCount Number of VGPRs used (or 0 if not applicable)
/// \param sgprCount Number of SGPRs used (or 0 if not applicable)
/// \param vgprSpillCount Number of VGPRs spilled (or 0)
/// \param sgprSpillCount Number of SGPRs spilled (or 0)
/// \param sharedMemoryBytes Shared memory usage in bytes
void emitRegisterUsageRemark(Location loc, llvm::StringRef backend,
                             llvm::StringRef dispatchName, uint32_t vgprCount,
                             uint32_t sgprCount, uint32_t vgprSpillCount,
                             uint32_t sgprSpillCount,
                             uint64_t sharedMemoryBytes);

/// Emits a "missed" remark when register spilling is detected.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param dispatchName Dispatch/kernel name
/// \param vgprSpillCount Number of VGPRs spilled
/// \param sgprSpillCount Number of SGPRs spilled
/// \param suggestion Optional suggestion text
void emitRegisterSpillingWarning(Location loc, llvm::StringRef backend,
                                  llvm::StringRef dispatchName,
                                  uint32_t vgprSpillCount,
                                  uint32_t sgprSpillCount,
                                  llvm::StringRef suggestion = "");

/// Emits a remark for optimization pipeline execution.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param functionName Function being optimized
/// \param optLevel Optimization level (e.g., "O2", "O3")
/// \param passPipeline String description of passes run
/// \param additionalMetrics Callback to add more metrics
void emitOptimizationPipelineRemark(
    Location loc, llvm::StringRef backend, llvm::StringRef functionName,
    llvm::StringRef optLevel, llvm::StringRef passPipeline,
    std::function<void(remark::InFlightRemark &)> additionalMetrics = nullptr);

/// Emits a "missed" remark for skipped optimizations.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param functionName Function name
/// \param optimizationName Name of the skipped optimization
/// \param reason Why it was skipped
/// \param suggestion How to enable it
void emitOptimizationSkippedRemark(Location loc, llvm::StringRef backend,
                                   llvm::StringRef functionName,
                                   llvm::StringRef optimizationName,
                                   llvm::StringRef reason,
                                   llvm::StringRef suggestion);

/// Emits a "failure" remark for serialization failures.
/// \param loc Source location for the remark
/// \param backend Target backend name
/// \param functionName Function name
/// \param phase Compilation phase where failure occurred
/// \param errorMessage Detailed error message
void emitSerializationFailureRemark(Location loc, llvm::StringRef backend,
                                    llvm::StringRef functionName,
                                    llvm::StringRef phase,
                                    llvm::StringRef errorMessage);

/// Emits a remark for HAL executable serialization start.
/// \param loc Source location for the remark
/// \param executableName Name of the executable
/// \param targetName Target backend name
/// \param debugLevel Debug level setting
/// \param variantCount Number of variants to serialize
void emitExecutableSerializationStartRemark(Location loc,
                                            llvm::StringRef executableName,
                                            llvm::StringRef targetName,
                                            int debugLevel,
                                            size_t variantCount);

/// Emits a remark for HAL executable serialization completion.
/// \param loc Source location for the remark
/// \param executableName Name of the executable
/// \param targetName Target backend name
/// \param variantsProcessed Number of variants successfully processed
void emitExecutableSerializationCompleteRemark(Location loc,
                                               llvm::StringRef executableName,
                                               llvm::StringRef targetName,
                                               size_t variantsProcessed);

} // namespace RemarkUtils

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_REMARKUTILS_H_

