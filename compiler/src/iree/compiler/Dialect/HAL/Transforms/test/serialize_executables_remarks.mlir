// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-serialize-target-executables{target=rocm})))" \
// RUN:   --iree-remarks-output-file=%t.remarks.yaml %s | FileCheck %s --check-prefix=CHECK-IR
// RUN: FileCheck %s --check-prefix=CHECK-REMARKS < %t.remarks.yaml

// This test verifies that the ROCM target backend emits structured remarks
// during serialization, including:
// - Executable serialization start/complete remarks
// - Binary serialization success remarks
// - Register usage analysis remarks
// - Binary dump remarks (when enabled)
// - Optimization pipeline remarks

// CHECK-REMARKS: --- !Analysis
// CHECK-REMARKS: pass: HAL:Serialization
// CHECK-REMARKS: name: ExecutableSerializationStarted
// CHECK-REMARKS: function: simple_mul_executable
// CHECK-REMARKS: args:
// CHECK-REMARKS:   - Remark: Starting executable serialization
// CHECK-REMARKS:   - Target: rocm
// CHECK-REMARKS:   - DebugLevel:

// CHECK-REMARKS: --- !Passed
// CHECK-REMARKS: pass: Serialization:ROCM
// CHECK-REMARKS: name: SerializationSuccess
// CHECK-REMARKS: function: simple_mul_executable
// CHECK-REMARKS: args:
// CHECK-REMARKS:   - Remark: Successfully serialized binary
// CHECK-REMARKS:   - BinarySize:
// CHECK-REMARKS:   - TargetArch:
// CHECK-REMARKS:   - WavefrontSize:

// CHECK-REMARKS: --- !Analysis
// CHECK-REMARKS: pass: OptimizationDecisions:ROCM
// CHECK-REMARKS: name: OptimizationPipeline
// CHECK-REMARKS: function: simple_mul_executable
// CHECK-REMARKS: args:
// CHECK-REMARKS:   - Remark: Ran optimization pipeline
// CHECK-REMARKS:   - OptLevel: O2
// CHECK-REMARKS:   - PassPipeline:
// CHECK-REMARKS:   - SLPVectorization:

// CHECK-REMARKS: --- !Passed
// CHECK-REMARKS: pass: HAL:Serialization
// CHECK-REMARKS: name: ExecutableSerializationCompleted
// CHECK-REMARKS: function: simple_mul_executable
// CHECK-REMARKS: args:
// CHECK-REMARKS:   - Remark: Completed executable serialization
// CHECK-REMARKS:   - Target: rocm
// CHECK-REMARKS:   - VariantsProcessed:

// CHECK-IR: hal.executable.binary
module attributes {hal.device.targets = [#hal.device.target<"hip", [#hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942"}>]>, #hal.device.target<"llvm-cpu", [#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}>]>]} {
  hal.executable private @simple_mul_executable {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx942"}>) {
      hal.executable.export public @simple_mul ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {workgroup_size = [64 : index, 1 : index, 1 : index]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @simple_mul() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<4xf32>
          linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0, %1 : memref<4xf32>, memref<4xf32>) outs(%2 : memref<4xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %3 = arith.mulf %in, %in_0 : f32
            linalg.yield %3 : f32
          }
          return
        }
      }
    }
  }
}

// -----

// Test for register spilling remark emission

// CHECK-REMARKS: --- !Analysis
// CHECK-REMARKS: pass: ResourceUsage:ROCM
// CHECK-REMARKS: name: RegisterUsage
// CHECK-REMARKS: args:
// CHECK-REMARKS:   - Remark: Kernel resource usage
// CHECK-REMARKS:   - VGPRCount:
// CHECK-REMARKS:   - SGPRCount:
// CHECK-REMARKS:   - VGPRSpillCount:
// CHECK-REMARKS:   - SGPRSpillCount:
// CHECK-REMARKS:   - SharedMemoryBytes:

