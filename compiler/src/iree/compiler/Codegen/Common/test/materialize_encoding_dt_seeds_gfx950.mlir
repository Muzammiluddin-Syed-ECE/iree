// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx950 --iree-gpu-dt-use-seeds \
// RUN:   --split-input-file %s | FileCheck %s

// Verify that --iree-gpu-dt-use-seeds switches the data-tiled encoding path to
// use the seed-based heuristic (deduceMMASchedule) for schedule selection,
// producing a data_tiled_scaled_mma_layout with seed-derived parameters.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<
  operand_index = 0 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [1024, 1024, 512, 32]>

func.func @dt_seeds_set_encoding_LHS(
    %arg0: tensor<1024x512x32xf4E2M1FN>) -> tensor<1024x512x32xf4E2M1FN, #encoding_lhs> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<1024x512x32xf4E2M1FN> -> tensor<1024x512x32xf4E2M1FN, #encoding_lhs>
  return %0 : tensor<1024x512x32xf4E2M1FN, #encoding_lhs>
}

// CHECK-LABEL: func.func @dt_seeds_set_encoding_LHS(
// CHECK:         linalg.pack
// CHECK:         linalg.transpose
