// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx950 --test-iree-data-tiling-alternate-heuristic \
// RUN:   --split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// 1. Scaled MFMA
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 0 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_LHS_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<255x127x32xf4E2M1FN>) -> tensor<255x127x32xf4E2M1FN, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x127x32xf4E2M1FN> -> tensor<255x127x32xf4E2M1FN, #encoding>
  return %0 : tensor<255x127x32xf4E2M1FN, #encoding>
}

// CHECK-LABEL:   func.func @set_encoding_LHS_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f4E2M1FN)
// CHECK-SAME:      outer_dims_perm = [0, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 1, 2]
// CHECK-SAME:      inner_tiles = [16, 8, 32]
// CHECK-SAME:      : tensor<255x127x32xf4E2M1FN> -> tensor<16x16x1x16x8x32xf4E2M1FN>
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<16x16x1x16x8x32xf4E2M1FN> into tensor<16x16x1x16x2x4x32xf4E2M1FN>
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[EXPANDED]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 1 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_RHS_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<513x127x32xf4E2M1FN>) -> tensor<513x127x32xf4E2M1FN, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<513x127x32xf4E2M1FN> -> tensor<513x127x32xf4E2M1FN, #encoding>
  return %0 : tensor<513x127x32xf4E2M1FN, #encoding>
}

// CHECK-LABEL:   func.func @set_encoding_RHS_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f4E2M1FN)
// CHECK-SAME:      outer_dims_perm = [0, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 1, 2]
// CHECK-SAME:      inner_tiles = [16, 8, 32]
// CHECK-SAME:      : tensor<513x127x32xf4E2M1FN> -> tensor<33x16x1x16x8x32xf4E2M1FN>
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<33x16x1x16x8x32xf4E2M1FN> into tensor<33x16x1x16x2x4x32xf4E2M1FN>
// CHECK-NOT:     linalg.transpose
// CHECK:         return %[[EXPANDED]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 2 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_LHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<255x127xf8E8M0FNU>) -> tensor<255x127xf8E8M0FNU, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x127xf8E8M0FNU> -> tensor<255x127xf8E8M0FNU, #encoding>
  return %0 : tensor<255x127xf8E8M0FNU, #encoding>
}

// CHECK-LABEL:   func.func @set_encoding_LHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f8E8M0FNU)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 8]
// CHECK-SAME:      : tensor<255x127xf8E8M0FNU> -> tensor<16x16x16x8xf8E8M0FNU>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<16x16x16x8xf8E8M0FNU> into tensor<16x16x16x2x4xf8E8M0FNU>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<16x16x16x2x4xf8E8M0FNU>)
// CHECK-SAME:       outs({{.*}} : tensor<16x16x4x16x2xf8E8M0FNU>)
// CHECK-SAME:       permutation = [0, 1, 4, 2, 3]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 3 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_RHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<513x127xf8E8M0FNU>) -> tensor<513x127xf8E8M0FNU, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<513x127xf8E8M0FNU> -> tensor<513x127xf8E8M0FNU, #encoding>
  return %0 : tensor<513x127xf8E8M0FNU, #encoding>
}

// CHECK-LABEL:   func.func @set_encoding_RHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f8E8M0FNU)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 8]
// CHECK-SAME:      : tensor<513x127xf8E8M0FNU> -> tensor<33x16x16x8xf8E8M0FNU>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<33x16x16x8xf8E8M0FNU> into tensor<33x16x16x2x4xf8E8M0FNU>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<33x16x16x2x4xf8E8M0FNU>)
// CHECK-SAME:       outs({{.*}} : tensor<33x16x4x16x2xf8E8M0FNU>)
// CHECK-SAME:       permutation = [0, 1, 4, 2, 3]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [1024, 2048, 128, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [1024, 2048, 128, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [1024, 2048, 128, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [1024, 2048, 128, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [1024, 2048, 128, 32]>

func.func @scaled_matmul_lowering_large_f4_f4_f8_f8_f32(
    %arg0: tensor<1024x128x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<2048x128x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<1024x128xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<2048x128xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<1024x2048xf32, #encoding_result>
) -> tensor<1024x2048xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<1024x128x32xf4E2M1FN, #encoding_lhs>, tensor<2048x128x32xf4E2M1FN, #encoding_rhs>,
             tensor<1024x128xf8E8M0FNU, #encoding_lhs_scales>, tensor<2048x128xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<1024x2048xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<1024x2048xf32, #encoding_result>
  return %0 : tensor<1024x2048xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_large_f4_f4_f8_f8_f32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<4x32x1x4x4x16x4x32xf4E2M1FN>, %[[RHS:.+]]: tensor<8x32x1x2x8x16x4x32xf4E2M1FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<4x32x4x4x16x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<8x32x2x4x16x8xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<4x8x4x2x4x8x4x16x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:    kind = {{.*}}unswizzled_operands = [0, 1]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32(
    %arg0: tensor<?x?x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf4E2M1FN, #encoding_lhs>, tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x4x16x4x4x32xf4E2M1FN>, %[[RHS:.+]]: tensor<?x?x1x4x2x16x4x4x32xf4E2M1FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x4x16x4x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x4x4x16x2x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x4x4x2x4x16x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    kind = {{.*}}unswizzled_operands = [0, 1]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

func.func @scaled_matmul_lowering_f8_f8_f8_f8_f32(
    %arg0: tensor<?x?x32xf8E4M3FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf8E4M3FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf8E4M3FN, #encoding_lhs>, tensor<?x?x32xf8E4M3FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f8E4M3FN, %in_0: f8E4M3FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f8E4M3FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f8E4M3FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f8_f8_f8_f8_f32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x2x2x16x4x4x32xf8E4M3FN>, %[[RHS:.+]]: tensor<?x?x1x2x2x16x4x4x32xf8E4M3FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x2x4x16x2x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x2x4x16x2x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x2x2x2x2x4x16x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    kind = {{.*}}unswizzled_operands = [0, 1]>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx950", features = "",
    wgp = <compute = fp16, storage =  b16,
           scaled_mma = [
             <intrinsic = MFMA_SCALE_F32_32x32x64_B32,
              lhs_elem_type = f4E2M1FN,
              rhs_elem_type = f4E2M1FN,
              acc_elem_type = f32>],
           subgroup =  none, subgroup_size_choices = [64],
           max_workgroup_sizes = [1024, 1024, 1024],
           max_thread_count_per_workgroup = 1024,
           max_workgroup_memory_bytes = 163840,
           max_workgroup_counts = [2147483647, 2147483647, 2147483647],
           max_load_instruction_bits = 128,
           simds_per_wgp = 4,
           vgpr_space_bits = 16384>>,
   iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>}>
func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32_MFMA_SCALE_F32_32x32x64_B32(
    %arg0: tensor<?x?x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result>
    attributes { hal.executable.target = #executable_target } {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf4E2M1FN, #encoding_lhs>, tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32_MFMA_SCALE_F32_32x32x64_B32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x2x2x32x4x2x32xf4E2M1FN>, %[[RHS:.+]]: tensor<?x?x1x2x2x32x4x2x32xf4E2M1FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x2x2x32x2x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x2x2x32x2x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x2x2x2x2x4x2x32x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    kind = {{.*}}unswizzled_operands = [0, 1]>
