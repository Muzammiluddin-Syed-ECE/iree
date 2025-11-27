# IREE Remarks Enhancement - Implementation Summary

## Overview

This implementation adds comprehensive structured remark support to IREE's compilation pipeline, particularly focusing on target backend serialization. The changes leverage MLIR's Remark infrastructure to provide machine-readable and human-readable insights into compilation decisions, performance characteristics, and diagnostic information.

## Changes Made

### 1. Core Infrastructure

**Files Created:**
- `compiler/src/iree/compiler/Utils/RemarkUtils.h` - Header defining helper utilities and standard remark categories
- `compiler/src/iree/compiler/Utils/RemarkUtils.cpp` - Implementation of remark emission helpers

**Standard Remark Categories Defined:**
- `Serialization` - Binary generation and encoding events
- `ResourceUsage` - Register, memory, and compute usage analysis
- `OptimizationDecisions` - Optimization decision explanations
- `PerformanceWarnings` - Performance warnings and diagnostics
- `HAL` - HAL-related transformation events
- `TargetBackend` - Backend-specific remarks

**Helper Functions Added:**
- `emitSerializationSuccessRemark()` - For successful binary serialization
- `emitBinaryDumpRemark()` - For binary dumps to disk
- `emitRegisterUsageRemark()` - For register usage analysis
- `emitRegisterSpillingWarning()` - For register spilling warnings
- `emitOptimizationPipelineRemark()` - For optimization pipeline execution
- `emitOptimizationSkippedRemark()` - For skipped optimizations
- `emitSerializationFailureRemark()` - For serialization failures
- `emitExecutableSerializationStartRemark()` - For HAL serialization start
- `emitExecutableSerializationCompleteRemark()` - For HAL serialization completion

### 2. ROCM Target Backend Integration

**File Modified:**
- `compiler/plugins/target/ROCM/ROCMTarget.cpp`

**Changes:**
1. **Added RemarkUtils include**
   - Integrated remark utilities into the target backend

2. **Enhanced Register Spilling Detection**
   - Replaced raw `emitWarning` with structured remarks
   - Now emits `RegisterUsage` analysis remarks for all kernels
   - Emits `RegisterSpillingDetected` missed remarks when spilling occurs
   - Includes metrics: VGPRCount, SGPRCount, VGPRSpillCount, SGPRSpillCount, SharedMemoryBytes

3. **Added Optimization Pipeline Tracing**
   - Modified `optimizeModule()` to accept Location parameter
   - Emits `OptimizationPipeline` analysis remark after LLVM passes
   - Includes metrics: OptLevel, PassPipeline, SLPVectorization status

4. **Added Serialization Success Remarks**
   - Emits `SerializationSuccess` passed remark after HSACO creation
   - Includes metrics: BinarySize, TargetArch, WavefrontSize

5. **Added Binary Dump Remarks**
   - Emits `BinaryDumped` analysis remark when binaries are dumped to disk
   - Includes metrics: Path, Filename, Size, Format

### 3. HAL SerializeExecutablesPass Integration

**File Modified:**
- `compiler/src/iree/compiler/Dialect/HAL/Transforms/SerializeExecutables.cpp`

**Changes:**
1. **Added RemarkUtils include**
2. **Enhanced SerializeTargetExecutablesPass::runOnOperation()**
   - Emits `ExecutableSerializationStarted` analysis remark at start
   - Tracks number of variants processed
   - Emits `ExecutableSerializationCompleted` passed remark at end
   - Includes metrics: Target, DebugLevel, VariantCount, VariantsProcessed

### 4. Build System Updates

**Files Modified:**
- `compiler/src/iree/compiler/Utils/BUILD.bazel` - Added RemarkUtils.{h,cpp} and MLIRRemark dependency
- `compiler/src/iree/compiler/Utils/CMakeLists.txt` - Added RemarkUtils.{h,cpp} and MLIRRemark dependency

### 5. Testing

**File Created:**
- `compiler/src/iree/compiler/Dialect/HAL/Transforms/test/serialize_executables_remarks.mlir`

**Test Coverage:**
- Verifies ExecutableSerializationStarted remark emission
- Verifies SerializationSuccess remark with metrics
- Verifies OptimizationPipeline remark emission
- Verifies ExecutableSerializationCompleted remark
- Verifies RegisterUsage analysis remarks
- Tests YAML format output

## Example Output

### YAML Remark Format

```yaml
--- !Analysis
pass: HAL:Serialization
name: ExecutableSerializationStarted
function: my_matmul_executable
loc: my_module.mlir:42:5
args:
  - Remark: Starting executable serialization
  - Target: rocm
  - DebugLevel: 3
  - VariantCount: 1

--- !Passed
pass: Serialization:ROCM
name: SerializationSuccess
function: my_matmul_executable
loc: my_module.mlir:42:5
args:
  - Remark: Successfully serialized binary
  - BinarySize: 8192
  - TargetArch: gfx942
  - WavefrontSize: 64

--- !Analysis
pass: OptimizationDecisions:ROCM
name: OptimizationPipeline
function: my_matmul_executable
loc: my_module.mlir:42:5
args:
  - Remark: Ran optimization pipeline
  - OptLevel: O2
  - PassPipeline: verify,function<eager-inv>(float2int,loop-vectorize)...
  - SLPVectorization: enabled

--- !Analysis
pass: ResourceUsage:ROCM
name: RegisterUsage
function: my_matmul_dispatch
loc: my_module.mlir:45:10
args:
  - Remark: Kernel resource usage
  - VGPRCount: 128
  - SGPRCount: 48
  - VGPRSpillCount: 0
  - SGPRSpillCount: 0
  - SharedMemoryBytes: 65536

--- !Analysis
pass: Serialization:ROCM
name: BinaryDumped
function: my_matmul_executable
loc: my_module.mlir:42:5
args:
  - Remark: Binary dumped to disk
  - Path: /tmp/binaries
  - Filename: module_my_matmul_executable_rocm_hsaco_fb.hsaco
  - Size: 8192
  - Format: HSACO

--- !Passed
pass: HAL:Serialization
name: ExecutableSerializationCompleted
function: my_matmul_executable
loc: my_module.mlir:42:5
args:
  - Remark: Completed executable serialization
  - Target: rocm
  - VariantsProcessed: 1
```

## Usage

### Enabling Remarks

```bash
# Enable remarks with YAML output
iree-compile \
  --iree-hal-target-backends=rocm \
  --iree-hip-target=gfx942 \
  --iree-hal-dump-binaries-path=/tmp/binaries \
  --iree-remarks-output-file=remarks.yaml \
  input.mlir -o output.vmfb

# Analyze remarks with Python
python analyze_remarks.py remarks.yaml
```

### Python Analysis Example

```python
import yaml

with open('remarks.yaml') as f:
    remarks = list(yaml.safe_load_all(f))

# Find register spills
spills = [r for r in remarks 
          if r.get('name') == 'RegisterSpillingDetected']

if spills:
    print(f"WARNING: Found {len(spills)} kernels with register spilling!")
    for s in spills:
        print(f"  - {s['function']} at {s['loc']}")

# Calculate total binary size
binaries = [r for r in remarks if r.get('name') == 'SerializationSuccess']
total_size = sum(next((arg['BinarySize'] for arg in r['args'] 
                      if 'BinarySize' in arg), 0) 
                for r in binaries)
print(f"Total binary size: {total_size} bytes")
```

## Benefits

### For Compiler Developers
- **Better debugging**: Structured traces of compilation decisions
- **Performance analysis**: Machine-readable metrics for automation
- **Regression detection**: Parse YAML remarks to track metric changes over time

### For End Users
- **Transparency**: Understand why kernels performed certain ways
- **Actionable feedback**: Suggestions for improving compilation results
- **Audit trail**: Complete record of what happened during compilation

### For CI/CD Systems
- **Automated analysis**: Parse remarks to detect regressions
- **Performance tracking**: Trend metrics like register usage, binary sizes
- **Quality gates**: Fail builds on critical missed optimizations or spills

## Compatibility

- **Backward compatible**: Remarks are opt-in via existing `--iree-remarks-output-file` flag
- **Zero overhead**: No performance cost when not enabled
- **MLIR standard**: Uses MLIR's mature Remark infrastructure
- **Tool friendly**: Standard YAML/bitstream format for tooling integration

## Future Extensions

### Phase 3: Extend to Other Backends (Planned)
1. Apply same patterns to CUDA target
2. Apply to LLVM-CPU target
3. Apply to Vulkan/Metal SPIRV targets

### Additional Enhancements (Proposed)
- Add failure remarks for compilation errors
- Add missed optimization remarks with suggestions
- Integrate with pass timing information
- Add binary format-specific metrics (ELF sections, SPIR-V modules, etc.)
- Create web-based visualization tool for remarks

## Testing

Run the lit test to verify remark emission:

```bash
cd /home/muzasyed/iree-remarks-enhancement
lit -v compiler/src/iree/compiler/Dialect/HAL/Transforms/test/serialize_executables_remarks.mlir
```

## Related Documents

- [Proposal Document](/home/muzasyed/iree_remarks_proposal.md)
- [MLIR Remarks Documentation](https://mlir.llvm.org/docs/Remarks/)
- [PR #21863](https://github.com/iree-org/iree/pull/21863/files)

## Authors

- Implementation Date: November 27, 2025
- Branch: `feature/remarks-enhancement`
- Worktree: `/home/muzasyed/iree-remarks-enhancement`

