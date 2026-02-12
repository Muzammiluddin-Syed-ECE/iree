// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-unroll-to-intrinsics))' --split-input-file | FileCheck %s

// Test: Standard (non-scaled) MMA outer-dimension unrolling.
// A 2x2x4 vector is unrolled across the M and K outer dimensions, producing
// 8 base intrinsic ops (4 parallel M/N positions x 2 K steps) plus
// insert_strided_slice reassembly.

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @unroll_mfma_f16(
    %lhs: vector<2x2x4xf16>,
    %rhs: vector<2x2x4xf16>,
    %acc: vector<2x2x4xf32>) -> vector<2x2x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2x2x4xf16>, vector<2x2x4xf16> into vector<2x2x4xf32>
  return %0 : vector<2x2x4xf32>
}

// Verify: 8 base inner_tiled ops (2M x 2N x 2K), each operating on vector<4x...>
// After unit dim folding, the inner tile is vector<4xf16>/vector<4xf32>.

// CHECK-LABEL: func @unroll_mfma_f16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<2x2x4xf32>

//       CHECK:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x2x4xf32>

// First K=0 pass: 4 intrinsics at M/N positions (0,0), (0,1), (1,0), (1,1).
//       CHECK:   vector.extract %[[LHS]][0, 0]
//       CHECK:   vector.extract %[[RHS]][0, 0]
//       CHECK:   vector.extract %[[ACC]][0, 0]
//       CHECK:   %[[MMA_00_K0:.*]] = iree_codegen.inner_tiled {{.*}} kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][0, 0]
//       CHECK:   vector.extract %[[RHS]][0, 1]
//       CHECK:   vector.extract %[[ACC]][0, 1]
//       CHECK:   %[[MMA_01_K0:.*]] = iree_codegen.inner_tiled {{.*}} kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][1, 0]
//       CHECK:   vector.extract %[[RHS]][0, 0]
//       CHECK:   vector.extract %[[ACC]][1, 0]
//       CHECK:   %[[MMA_10_K0:.*]] = iree_codegen.inner_tiled {{.*}} kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][1, 0]
//       CHECK:   vector.extract %[[RHS]][0, 1]
//       CHECK:   vector.extract %[[ACC]][1, 1]
//       CHECK:   %[[MMA_11_K0:.*]] = iree_codegen.inner_tiled {{.*}} kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

// Second K=1 pass: chain through ACC from K=0.
//       CHECK:   vector.extract %[[LHS]][0, 1]
//       CHECK:   vector.extract %[[RHS]][1, 0]
//       CHECK:   %[[MMA_00:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA_00_K0]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][0, 1]
//       CHECK:   vector.extract %[[RHS]][1, 1]
//       CHECK:   %[[MMA_01:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA_01_K0]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][1, 1]
//       CHECK:   vector.extract %[[RHS]][1, 0]
//       CHECK:   %[[MMA_10:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA_10_K0]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

//       CHECK:   vector.extract %[[LHS]][1, 1]
//       CHECK:   vector.extract %[[RHS]][1, 1]
//       CHECK:   %[[MMA_11:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA_11_K0]])
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

// Reassemble the 2x2x4 result.
//       CHECK:   vector.insert_strided_slice {{.*}} {offsets = [0, 0, 0]{{.*}} : vector<1x1x4xf32> into vector<2x2x4xf32>
//       CHECK:   vector.insert_strided_slice {{.*}} {offsets = [0, 1, 0]{{.*}} : vector<1x1x4xf32> into vector<2x2x4xf32>
//       CHECK:   vector.insert_strided_slice {{.*}} {offsets = [1, 0, 0]{{.*}} : vector<1x1x4xf32> into vector<2x2x4xf32>
//       CHECK:   %[[RES:.*]] = vector.insert_strided_slice {{.*}} {offsets = [1, 1, 0]{{.*}} : vector<1x1x4xf32> into vector<2x2x4xf32>
//       CHECK:   return %[[RES]]

// -----
// Test: Scaled MMA outer-dimension unrolling (no repeats).
// Outer dims k=2, b=2 are reduction dims that get unrolled, chaining
// through the accumulator. After unit dim folding the inner tile is 1-D.

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]
func.func @unroll_scaled_mma_outer(
    %lhs: vector<1x2x2x32xf4E2M1FN>,
    %rhs: vector<2x2x1x32xf8E4M3FN>,
    %lhs_sc: vector<1x2x1xf8E8M0FNU>,
    %rhs_sc: vector<2x1x1xf8E8M0FNU>,
    %acc: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_sc, %rhs_sc) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<1x2x2x32xf4E2M1FN>, vector<2x2x1x32xf8E4M3FN>, vector<1x2x1xf8E8M0FNU>, vector<2x1x1xf8E8M0FNU> into vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

// Verify: 4 base inner_tiled ops chained (k=2 x b=2 reduction steps),
// each with 1-D operands after unit dim folding.

// CHECK-LABEL: func @unroll_scaled_mma_outer
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x2x2x32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2x2x1x32xf8E4M3FN>
//  CHECK-SAME:   %[[LHS_SC:[A-Za-z0-9]+]]: vector<1x2x1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SC:[A-Za-z0-9]+]]: vector<2x1x1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1x1x4xf32>

// Step 1: (k=0, b=0)
//       CHECK:   vector.extract %[[LHS]][0, 0, 0]
//       CHECK:   vector.extract %[[RHS]][0, 0, 0]
//       CHECK:   vector.extract %[[LHS_SC]][0, 0]
//       CHECK:   vector.extract %[[RHS_SC]][0, 0]
//       CHECK:   vector.extract %[[ACC]][0, 0]
//       CHECK:   %[[MMA0:.*]] = iree_codegen.inner_tiled {{.*}} kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>

// Step 2: (k=0, b=1) chains through MMA0
//       CHECK:   vector.extract %[[LHS]][0, 0, 1]
//       CHECK:   vector.extract %[[RHS]][0, 1, 0]
//       CHECK:   vector.extract %[[LHS_SC]][0, 0]
//       CHECK:   vector.extract %[[RHS_SC]][0, 0]
//       CHECK:   %[[MMA1:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA0]])
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>

// Step 3: (k=1, b=0) chains through MMA1
//       CHECK:   vector.extract %[[LHS]][0, 1, 0]
//       CHECK:   vector.extract %[[RHS]][1, 0, 0]
//       CHECK:   vector.extract %[[LHS_SC]][0, 1]
//       CHECK:   vector.extract %[[RHS_SC]][1, 0]
//       CHECK:   %[[MMA2:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA1]])
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>

// Step 4: (k=1, b=1) chains through MMA2
//       CHECK:   vector.extract %[[LHS]][0, 1, 1]
//       CHECK:   vector.extract %[[RHS]][1, 1, 0]
//       CHECK:   vector.extract %[[LHS_SC]][0, 1]
//       CHECK:   vector.extract %[[RHS_SC]][1, 0]
//       CHECK:   %[[MMA3:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[MMA2]])
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>

// After unit dim folding the final result is broadcast back.
//       CHECK:   vector.broadcast %[[MMA3]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return

// -----
// Test: No-op case. An inner_tiled op that is already at base intrinsic
// size (no outer dims, no repeats) should pass through unchanged.

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @already_base_intrinsic(
    %lhs: vector<4xf16>,
    %rhs: vector<4xf16>,
    %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

// Verify: the op passes through unchanged -- a single inner_tiled op, no slicing.
// CHECK-LABEL: func @already_base_intrinsic
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   %[[RES:.*]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>
//   CHECK-NOT:   vector.extract
//   CHECK-NOT:   vector.insert
//       CHECK:   return %[[RES]]
