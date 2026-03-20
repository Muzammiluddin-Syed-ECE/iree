// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

// Verify that ScaledMMAAttr with repeats reindexes LHS scale reads through the
// preshuffled LDS layout. The scale input comes through the standard chain:
//   flat 2D LDS memref -> expand_shape -> transfer_read -> shape_cast
// When repeats are present, buildUnderlyingOperations walks this chain, finds
// the flat memref, and emits a new transfer_read at reindexed coordinates.

// Test 1: Constant indices, fully foldable.
// Original 4D position: [2, 5, 1, 2] → flat (m=37, k=6)
// reindex(37, 6, R_m=2, R_k=2) → (43, 1)

#map5 = affine_map<() -> ()>

func.func @reindex_scale_constant_indices(
    %lhs: vector<32xf4E2M1FN>,
    %rhs: vector<32xf4E2M1FN>,
    %acc: vector<4xf32>,
    %lds_lhs_scale: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>,
    %lds_rhs_scale: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
  ) -> vector<4xf32> {
  %pad = arith.constant 0.0 : f8E8M0FNU
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index

  %lhs_exp = memref.expand_shape %lds_lhs_scale [[0, 1], [2, 3]]
    output_shape [16, 16, 2, 4]
    : memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
      into memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>
  %rhs_exp = memref.expand_shape %lds_rhs_scale [[0, 1], [2, 3]]
    output_shape [16, 16, 2, 4]
    : memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
      into memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>

  // transfer_read at [outer_m=2, thread_m=5, outer_k=1, thread_kb=2]
  %lhs_vec = vector.transfer_read %lhs_exp[%c2, %c5, %c1, %c2], %pad
    {in_bounds = [true, true, true, true]}
    : memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>,
      vector<1x1x1x1xf8E8M0FNU>
  %lhs_scale = vector.shape_cast %lhs_vec
    : vector<1x1x1x1xf8E8M0FNU> to vector<1xf8E8M0FNU>

  %rhs_vec = vector.transfer_read %rhs_exp[%c2, %c5, %c1, %c2], %pad
    {in_bounds = [true, true, true, true]}
    : memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>,
      vector<1x1x1x1xf8E8M0FNU>
  %rhs_scale = vector.shape_cast %rhs_vec
    : vector<1x1x1x1xf8E8M0FNU> to vector<1xf8E8M0FNU>

  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_scale, %rhs_scale) outs(%acc) {
    indexing_maps = [#map5, #map5, #map5, #map5, #map5],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      repeats = [2, 2]>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reindex_scale_constant_indices
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//  CHECK-SAME:   %[[LDS_LHS_SCALE:[A-Za-z0-9]+]]: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
//  CHECK-SAME:   %[[LDS_RHS_SCALE:[A-Za-z0-9]+]]: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
//
//       The reindexed read goes to the flat 2D LDS memref at [43, 1].
//   CHECK-DAG:   %[[C43:.+]] = arith.constant 43 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   %[[REINDEXED:.+]] = vector.transfer_read %[[LDS_LHS_SCALE]][%[[C43]], %[[C1]]]
//  CHECK-SAME:     memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
//  CHECK-SAME:     vector<1xf8E8M0FNU>
//       CHECK:   %[[SCALE_BYTE:.+]] = vector.extract %[[REINDEXED]][0]
//       CHECK:   %[[PADDED:.+]] = vector.insert %[[SCALE_BYTE]], %{{.+}} [0]
//       CHECK:   amdgpu.scaled_mfma 16x16x128 (%[[PADDED]][0] * %[[LHS]])

// -----

// Test 2: Dynamic indices produce arith ops for the reindex formula.
// The key check is that the read goes to the flat 2D memref (not the expanded
// 4D one) with computed index expressions.

#map5d = affine_map<() -> ()>

func.func @reindex_scale_dynamic_indices(
    %lhs: vector<32xf4E2M1FN>,
    %rhs: vector<32xf4E2M1FN>,
    %acc: vector<4xf32>,
    %lds_lhs_scale: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>,
    %lds_rhs_scale: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>,
    %outer_m: index, %thread_m: index, %thread_kb: index
  ) -> vector<4xf32> {
  %pad = arith.constant 0.0 : f8E8M0FNU
  %c0 = arith.constant 0 : index

  %lhs_exp = memref.expand_shape %lds_lhs_scale [[0, 1], [2, 3]]
    output_shape [16, 16, 2, 4]
    : memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
      into memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>
  %rhs_exp = memref.expand_shape %lds_rhs_scale [[0, 1], [2, 3]]
    output_shape [16, 16, 2, 4]
    : memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
      into memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>

  %lhs_vec = vector.transfer_read %lhs_exp[%outer_m, %thread_m, %c0, %thread_kb], %pad
    {in_bounds = [true, true, true, true]}
    : memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>,
      vector<1x1x1x1xf8E8M0FNU>
  %lhs_scale = vector.shape_cast %lhs_vec
    : vector<1x1x1x1xf8E8M0FNU> to vector<1xf8E8M0FNU>

  %rhs_vec = vector.transfer_read %rhs_exp[%c0, %thread_m, %c0, %thread_kb], %pad
    {in_bounds = [true, true, true, true]}
    : memref<16x16x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>,
      vector<1x1x1x1xf8E8M0FNU>
  %rhs_scale = vector.shape_cast %rhs_vec
    : vector<1x1x1x1xf8E8M0FNU> to vector<1xf8E8M0FNU>

  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhs_scale, %rhs_scale) outs(%acc) {
    indexing_maps = [#map5d, #map5d, #map5d, #map5d, #map5d],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32,
      repeats = [2, 2]>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @reindex_scale_dynamic_indices
//  CHECK-SAME:   %[[LDS_LHS:[A-Za-z0-9]+]]: memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
//  CHECK-SAME:   %[[OUTER_M:[A-Za-z0-9]+]]: index, %[[THREAD_M:[A-Za-z0-9]+]]: index, %[[THREAD_KB:[A-Za-z0-9]+]]: index
//
//       All constants defined up-front.
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//
//       Linearize m = outer_m * 16 + thread_m
//       CHECK:   %[[FLAT_M_PART:.+]] = arith.muli %[[OUTER_M]], %[[C16]]
//       CHECK:   %[[FLAT_M:.+]] = arith.addi %[[FLAT_M_PART]], %[[THREAD_M]]
//
//       Reindex formula: divs and mods for T_m, T_k, IntrinsicM, IntrinsicK
//       CHECK:   arith.divui %[[FLAT_M]], %[[C16]]
//       CHECK:   arith.divui %[[THREAD_KB]], %[[C4]]
//       CHECK:   arith.remui %[[FLAT_M]], %[[C16]]
//       CHECK:   arith.remui %[[THREAD_KB]], %[[C4]]
//
//       The reindexed read comes from the flat 2D LDS memref, not the expanded 4D one.
//       CHECK:   vector.transfer_read %[[LDS_LHS]][%{{.+}}, %{{.+}}]
//  CHECK-SAME:     memref<256x8xf8E8M0FNU, #gpu.address_space<workgroup>>
//  CHECK-SAME:     vector<1xf8E8M0FNU>
//       CHECK:   amdgpu.scaled_mfma

// -----

// Test 3: Without repeats, the standard path is used (no reindexing).
// Scale values are extracted from the per-thread scale input as-is.

#map5nr = affine_map<() -> ()>

func.func @no_reindex_without_repeats(
    %lhs: vector<32xf4E2M1FN>,
    %rhs: vector<32xf4E2M1FN>,
    %lhsScale: vector<1xf8E8M0FNU>,
    %rhsScale: vector<1xf8E8M0FNU>,
    %acc: vector<4xf32>
  ) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = [#map5nr, #map5nr, #map5nr, #map5nr, #map5nr],
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<32xf4E2M1FN>, vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>, vector<1xf8E8M0FNU> into vector<4xf32>
  return %0 : vector<4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @no_reindex_without_repeats
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: vector<1xf8E8M0FNU>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<4xf32>
//       CHECK:   %[[S0:.+]] = vector.extract %[[LHS_SCALE]][0]
//       CHECK:   %[[PADDED_LHS:.+]] = vector.insert %[[S0]], %{{.+}} [0]
//       CHECK:   %[[S1:.+]] = vector.extract %[[RHS_SCALE]][0]
//       CHECK:   %[[PADDED_RHS:.+]] = vector.insert %[[S1]], %{{.+}} [0]
//       CHECK:   amdgpu.scaled_mfma 16x16x128 (%[[PADDED_LHS]][0] * %[[LHS]]) * (%[[PADDED_RHS]][0] * %[[RHS]]) + %[[ACC]]
//   CHECK-NOT:   memref
