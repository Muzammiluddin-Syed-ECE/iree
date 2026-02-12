// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-decompose-repeats))' --split-input-file | FileCheck %s

// Test: Scaled MMA with K-only repeats=[1,1,4,1].
// No outer dimensions. The pattern slices the grouped inner tile (kScale=16)
// into 4 base intrinsic ops (kScale=4) that are chained through the
// accumulator using tensor.extract_slice / tensor.insert_slice.

#map0 = affine_map<() -> ()>
func.func @decompose_k_repeats(
    %lhs: tensor<16x16x32xf4E2M1FN>,
    %rhs: tensor<16x32x16xf4E2M1FN>,
    %lhs_sc: tensor<16x16xf8E8M0FNU>,
    %rhs_sc: tensor<16x16xf8E8M0FNU>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_sc, %rhs_sc) outs(%acc) {
    indexing_maps = [#map0, #map0, #map0, #map0, #map0],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      repeats = [1, 1, 4, 1]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<16x16x32xf4E2M1FN>, tensor<16x32x16xf4E2M1FN>, tensor<16x16xf8E8M0FNU>, tensor<16x16xf8E8M0FNU> into tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

// Verify: 4 chained base inner_tiled ops. Each uses kScale=4 slices.
// The base kind has no repeats attribute.
// K is a reduction dim, so the accumulator is threaded through all 4 steps.

// CHECK-LABEL: func @decompose_k_repeats
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x16x32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x32x16xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SC:[A-Za-z0-9]+]]: tensor<16x16xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SC:[A-Za-z0-9]+]]: tensor<16x16xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf32>

// K step 0: extract at K offset 0
//       CHECK:   tensor.extract_slice %[[LHS]][0, 0, 0] [16, 4, 32] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[RHS]][0, 0, 0] [4, 32, 16] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[LHS_SC]][0, 0] [16, 4] [1, 1]
//       CHECK:   tensor.extract_slice %[[RHS_SC]][0, 0] [4, 16] [1, 1]
//       CHECK:   %[[K0:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[ACC]])
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32
//  CHECK-NOT:      repeats
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>

// K step 1: extract at K offset 4
//       CHECK:   tensor.extract_slice %[[LHS]][0, 4, 0] [16, 4, 32] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[RHS]][4, 0, 0] [4, 32, 16] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[LHS_SC]][0, 4] [16, 4] [1, 1]
//       CHECK:   tensor.extract_slice %[[RHS_SC]][4, 0] [4, 16] [1, 1]
//       CHECK:   %[[K1:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[K0]])

// K step 2: extract at K offset 8
//       CHECK:   tensor.extract_slice %[[LHS]][0, 8, 0] [16, 4, 32] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[RHS]][8, 0, 0] [4, 32, 16] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[LHS_SC]][0, 8] [16, 4] [1, 1]
//       CHECK:   tensor.extract_slice %[[RHS_SC]][8, 0] [4, 16] [1, 1]
//       CHECK:   %[[K2:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[K1]])

// K step 3: extract at K offset 12
//       CHECK:   tensor.extract_slice %[[LHS]][0, 12, 0] [16, 4, 32] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[RHS]][12, 0, 0] [4, 32, 16] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[LHS_SC]][0, 12] [16, 4] [1, 1]
//       CHECK:   tensor.extract_slice %[[RHS_SC]][12, 0] [4, 16] [1, 1]
//       CHECK:   %[[K3:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[K2]])

//       CHECK:   return %[[K3]] : tensor<16x16xf32>

// -----
// Test: Scaled MMA with M/N-only repeats=[2,2,1,1].
// The inner tile for ACC is 32x32 (2x16 in M, 2x16 in N). The pass
// decomposes into 4 independent base intrinsic ops at parallel positions
// (0,0), (0,1), (1,0), (1,1), each with 16x16 accumulator tiles.
// Results are assembled using tensor.insert_slice.

#map0 = affine_map<() -> ()>
func.func @decompose_mn_repeats(
    %lhs: tensor<32x4x32xf4E2M1FN>,
    %rhs: tensor<4x32x32xf4E2M1FN>,
    %lhs_sc: tensor<32x4xf8E8M0FNU>,
    %rhs_sc: tensor<4x32xf8E8M0FNU>,
    %acc: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_sc, %rhs_sc) outs(%acc) {
    indexing_maps = [#map0, #map0, #map0, #map0, #map0],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      repeats = [2, 2, 1, 1]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<32x4x32xf4E2M1FN>, tensor<4x32x32xf4E2M1FN>, tensor<32x4xf8E8M0FNU>, tensor<4x32xf8E8M0FNU> into tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// Verify: 4 parallel base ops, no reduction chaining.
// The 32x32 result is assembled from four 16x16 sub-tiles.

// CHECK-LABEL: func @decompose_mn_repeats
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<32x4x32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<4x32x32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SC:[A-Za-z0-9]+]]: tensor<32x4xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SC:[A-Za-z0-9]+]]: tensor<4x32xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<32x32xf32>

// Position (M=0, N=0)
//       CHECK:   tensor.extract_slice %[[ACC]][0, 0] [16, 16] [1, 1]
//       CHECK:   tensor.extract_slice %[[LHS]][0, 0, 0] [16, 4, 32] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[RHS]][0, 0, 0] [4, 32, 16] [1, 1, 1]
//       CHECK:   tensor.extract_slice %[[LHS_SC]][0, 0] [16, 4] [1, 1]
//       CHECK:   tensor.extract_slice %[[RHS_SC]][0, 0] [4, 16] [1, 1]
//       CHECK:   %[[P00:.*]] = iree_codegen.inner_tiled
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   tensor.insert_slice %[[P00]] into %[[ACC]][0, 0] [16, 16] [1, 1]

// Position (M=0, N=1)
//       CHECK:   tensor.extract_slice {{.*}}[0, 16] [16, 16] [1, 1]
//       CHECK:   %[[P01:.*]] = iree_codegen.inner_tiled
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   tensor.insert_slice %[[P01]] into {{.*}}[0, 16] [16, 16] [1, 1]

// Position (M=1, N=0)
//       CHECK:   tensor.extract_slice {{.*}}[16, 0] [16, 16] [1, 1]
//       CHECK:   %[[P10:.*]] = iree_codegen.inner_tiled
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   tensor.insert_slice %[[P10]] into {{.*}}[16, 0] [16, 16] [1, 1]

// Position (M=1, N=1)
//       CHECK:   tensor.extract_slice {{.*}}[16, 16] [16, 16] [1, 1]
//       CHECK:   %[[P11:.*]] = iree_codegen.inner_tiled
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   %[[RES:.*]] = tensor.insert_slice %[[P11]] into {{.*}}[16, 16] [16, 16] [1, 1]

//       CHECK:   return %[[RES]] : tensor<32x32xf32>

// -----
// Test: Scaled MMA with mixed repeats=[2,2,2,1].
// 4 parallel positions (2M x 2N) and 2 reduction steps (K). This produces
// 8 total base intrinsic ops: for each parallel position, 2 ops are chained
// along the K dimension.

#map0 = affine_map<() -> ()>
func.func @decompose_mnk_repeats(
    %lhs: tensor<32x8x32xf4E2M1FN>,
    %rhs: tensor<8x32x32xf4E2M1FN>,
    %lhs_sc: tensor<32x8xf8E8M0FNU>,
    %rhs_sc: tensor<8x32xf8E8M0FNU>,
    %acc: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %result = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_sc, %rhs_sc) outs(%acc) {
    indexing_maps = [#map0, #map0, #map0, #map0, #map0],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      repeats = [2, 2, 2, 1]>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<32x8x32xf4E2M1FN>, tensor<8x32x32xf4E2M1FN>, tensor<32x8xf8E8M0FNU>, tensor<8x32xf8E8M0FNU> into tensor<32x32xf32>
  return %result : tensor<32x32xf32>
}

// Verify: 8 total inner_tiled ops. 4 parallel positions x 2 K steps each.
// All 8 use the base kind (no repeats). Each reduction pair chains ACC.

// CHECK-LABEL: func @decompose_mnk_repeats
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<32x8x32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x32x32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SC:[A-Za-z0-9]+]]: tensor<32x8xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SC:[A-Za-z0-9]+]]: tensor<8x32xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<32x32xf32>

// Position (M=0, N=0), K=0 then K=1
//       CHECK:   tensor.extract_slice %[[ACC]][0, 0] [16, 16] [1, 1]
//       CHECK:   %[[P00_K0:.*]] = iree_codegen.inner_tiled
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   %[[P00:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[P00_K0]])
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//       CHECK:   tensor.insert_slice %[[P00]] into %[[ACC]][0, 0] [16, 16] [1, 1]

// Position (M=0, N=1), K=0 then K=1
//       CHECK:   tensor.extract_slice {{.*}}[0, 16] [16, 16] [1, 1]
//       CHECK:   %[[P01_K0:.*]] = iree_codegen.inner_tiled
//       CHECK:   %[[P01:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[P01_K0]])
//       CHECK:   tensor.insert_slice %[[P01]] into {{.*}}[0, 16] [16, 16] [1, 1]

// Position (M=1, N=0), K=0 then K=1
//       CHECK:   tensor.extract_slice {{.*}}[16, 0] [16, 16] [1, 1]
//       CHECK:   %[[P10_K0:.*]] = iree_codegen.inner_tiled
//       CHECK:   %[[P10:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[P10_K0]])
//       CHECK:   tensor.insert_slice %[[P10]] into {{.*}}[16, 0] [16, 16] [1, 1]

// Position (M=1, N=1), K=0 then K=1
//       CHECK:   tensor.extract_slice {{.*}}[16, 16] [16, 16] [1, 1]
//       CHECK:   %[[P11_K0:.*]] = iree_codegen.inner_tiled
//       CHECK:   %[[P11:.*]] = iree_codegen.inner_tiled ins({{.*}}) outs(%[[P11_K0]])
//       CHECK:   %[[RES:.*]] = tensor.insert_slice %[[P11]] into {{.*}}[16, 16] [16, 16] [1, 1]

//       CHECK:   return %[[RES]] : tensor<32x32xf32>

// -----
// Test: No-op case. An inner_tiled op with no repeats should pass through
// unchanged.

#map0 = affine_map<() -> ()>
func.func @no_repeats_noop(
    %lhs: tensor<16x4x32xf4E2M1FN>,
    %rhs: tensor<4x32x16xf4E2M1FN>,
    %lhs_sc: tensor<16x4xf8E8M0FNU>,
    %rhs_sc: tensor<4x16xf8E8M0FNU>,
    %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %result = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_sc, %rhs_sc) outs(%acc) {
    indexing_maps = [#map0, #map0, #map0, #map0, #map0],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
  return %result : tensor<16x16xf32>
}

// Verify: the op passes through unchanged -- a single inner_tiled op, no slicing.
// CHECK-LABEL: func @no_repeats_noop
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x4x32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<4x32x16xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SC:[A-Za-z0-9]+]]: tensor<16x4xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SC:[A-Za-z0-9]+]]: tensor<4x16xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<16x16xf32>
//       CHECK:   %[[RES:.*]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]], %[[LHS_SC]], %[[RHS_SC]]) outs(%[[ACC]])
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32
//  CHECK-SAME:     : tensor<16x4x32xf4E2M1FN>, tensor<4x32x16xf4E2M1FN>, tensor<16x4xf8E8M0FNU>, tensor<4x16xf8E8M0FNU> into tensor<16x16xf32>
//   CHECK-NOT:   tensor.extract_slice
//   CHECK-NOT:   tensor.insert_slice
//       CHECK:   return %[[RES]]
