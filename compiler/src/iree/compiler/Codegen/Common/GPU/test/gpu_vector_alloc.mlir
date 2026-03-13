// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-vector-alloc))" | FileCheck %s

// Inner dim = 1*1*1*16*1 = 16 f16 elements = 32 bytes.
// 32 / 4 = 8 banks per row stride. gcd(8, 32) = 8 -> conflict.
// rowWidthElems = min(64, 16) = 16, accessWidth = 1.
#layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 16],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [0, 0]
>

func.func @test(%vector: vector<16x16xf16>) -> vector<16x16xf16> {
  %out = iree_vector_ext.to_layout %vector to layout(#layout) {shared_memory_conversion} : vector<16x16xf16>
  return %out : vector<16x16xf16>
}

//    CHECK-LABEL: func.func @test
//         CHECK:    gpu.barrier memfence [#gpu.address_space<workgroup>]
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<256xf16, #gpu.address_space<workgroup>>
//         CHECK:    %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.xor_shuffle<16, 1>]
//         CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[SWIZZLE]] {{\[}}[0, 1]{{\]}}
//         CHECK:    %[[WRITE:.+]] = vector.transfer_write %{{.*}}, %[[EXPAND]]
//         CHECK:    %[[BAR:.+]]   = iree_gpu.value_barrier %[[WRITE]]
//         CHECK:    %[[READ:.+]]  = vector.transfer_read %[[BAR]]
//         CHECK:    %[[OUT:.+]]   = iree_vector_ext.to_layout %[[READ]]

// -----

// Inner dim = 1*1*1*32*4 = 128 f16 elements = 256 bytes.
// 256 / 4 = 64 banks per row stride. 64 % 32 = 0 -> conflict.
// rowWidthElems = min(64, 128) = 64, accessWidth = 4.
#conflict_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [16, 32],
  element_tile = [1, 4],

  subgroup_strides = [0, 0],
  thread_strides = [1, 16]
>

func.func @conflict(%v: vector<16x128xf16>) -> vector<16x128xf16> {
  %out = iree_vector_ext.to_layout %v to layout(#conflict_layout)
      {shared_memory_conversion} : vector<16x128xf16>
  return %out : vector<16x128xf16>
}

//    CHECK-LABEL: func.func @conflict
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<2048xf16, #gpu.address_space<workgroup>>
//         CHECK:    %[[SWIZZLE:.+]] = iree_codegen.swizzle_hint %[[ALLOC]][#iree_codegen.xor_shuffle<64, 4>]
//         CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[SWIZZLE]] {{\[}}[0, 1]{{\]}}
//         CHECK:    vector.transfer_write %{{.*}}, %[[EXPAND]]

// -----

// Inner dim = 1*1*1*17*1 = 17 f32 elements = 68 bytes.
// 68 / 4 = 17 banks per row stride. gcd(17, 32) = 1 -> no conflict.
#no_conflict_layout = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [4, 17],
  element_tile = [4, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [0, 0]
>

func.func @no_conflict(%vector: vector<16x17xf32>) -> vector<16x17xf32> {
  %out = iree_vector_ext.to_layout %vector to layout(#no_conflict_layout)
      {shared_memory_conversion} : vector<16x17xf32>
  return %out : vector<16x17xf32>
}

//    CHECK-LABEL: func.func @no_conflict
//     CHECK-NOT:    iree_codegen.swizzle_hint
//         CHECK:    %[[ALLOC:.+]] = bufferization.alloc_tensor() {memory_space = #gpu.address_space<workgroup>} : tensor<16x17xf32, #gpu.address_space<workgroup>>
//         CHECK:    %[[WRITE:.+]] = vector.transfer_write %{{.*}}, %[[ALLOC]]
