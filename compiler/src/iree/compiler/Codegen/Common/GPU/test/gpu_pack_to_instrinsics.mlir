// RUN: iree-opt %s --mlir-print-local-scope --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-pack-to-intrinsics, canonicalize, cse))' --split-input-file | FileCheck %s

#config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>
module {
  func.func @matmul_32x32x8(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %mm = linalg.matmul {lowering_config = #config} ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }
}

// CHECK-LABEL: func.func @matmul_32x32x8
//  CHECK-SAME:   %[[A:[A-Za-z0-9]+]]: tensor<64x64xf16>
//  CHECK-SAME:   %[[B:[A-Za-z0-9]+]]: tensor<64x64xf16>
//  CHECK-SAME:   %[[C:[A-Za-z0-9]+]]: tensor<64x64xf32>
//   CHECK-DAG:   %[[A_PACK:.+]] = linalg.pack %[[A]] inner_dims_pos = [0, 1] inner_tiles = [32, 8]
//   CHECK-DAG:   %[[B_PACK:.+]] = linalg.pack %[[B]] inner_dims_pos = [1, 0] inner_tiles = [32, 8]
//   CHECK-DAG:   %[[C_PACK:.+]] = linalg.pack %[[C]] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
//       CHECK:   iree_codegen.inner_tiled ins(%[[A_PACK]], %[[B_PACK]]) outs(%[[C_PACK]])
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-SAME:       affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-SAME:     iterator_types = {{.*}}parallel{{.*}}parallel{{.*}}reduction
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>}>
//  CHECK-SAME:     permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @matmul_16x16x16(%a: tensor<?x?x?xf16>, %b: tensor<?x?x?x?xf16>, %c: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b : tensor<?x?x?xf16>, tensor<?x?x?x?xf16>)
    outs(%c : tensor<?x?x?xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
    } {
    ^bb0(%in: f16, %in_2: f16, %out: f32):
      %4 = arith.extf %in : f16 to f32
      %5 = arith.extf %in_2 : f16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<?x?x?xf32>
    return %mm : tensor<?x?x?xf32>
  }
}

// CHECK-LABEL: func.func @matmul_16x16x16
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d2, d0, d3, d4)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}>
//  CHECK-SAME:     : tensor<?x?x?x16x16xf16>, tensor<?x?x?x?x16x16xf16> into tensor<?x?x?x16x16xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @scaled_mfma_16x16x128(%a: tensor<?x?x?xf4E2M1FN>, %b: tensor<?x?x?xf4E2M1FN>, %a_scales: tensor<?x?xf8E8M0FNU>, %b_scales: tensor<?x?xf8E8M0FNU>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b, %a_scales, %b_scales : tensor<?x?x?xf4E2M1FN>, tensor<?x?x?xf4E2M1FN>, tensor<?x?xf8E8M0FNU>, tensor<?x?xf8E8M0FNU>)
    outs(%c : tensor<?x?xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>}>
    } {
    ^bb0(%in: f4E2M1FN, %in_4: f4E2M1FN, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: f32):
      %17 = arith.scaling_extf %in, %in_5 : f4E2M1FN, f8E8M0FNU to f32
      %18 = arith.scaling_extf %in_4, %in_6 : f4E2M1FN, f8E8M0FNU to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @scaled_mfma_16x16x128
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>}>
//  CHECK-SAME:     permutations = [array<i64: 0, 1, 2>, array<i64: 2, 0, 1>, array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
//  CHECK-SAME:     : tensor<?x?x?x16x4x32xf4E2M1FN>, tensor<?x?x?x16x4x32xf4E2M1FN>, tensor<?x?x16x4xf8E8M0FNU>, tensor<?x?x16x4xf8E8M0FNU> into tensor<?x?x16x16xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @scaled_mfma_32x32x64(%a: tensor<?x?x?xf8E8M0FNU>, %b: tensor<?x?x?xf8E8M0FNU>, %a_scales: tensor<?x?xf8E8M0FNU>, %b_scales: tensor<?x?xf8E8M0FNU>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b, %a_scales, %b_scales : tensor<?x?x?xf8E8M0FNU>, tensor<?x?x?xf8E8M0FNU>, tensor<?x?xf8E8M0FNU>, tensor<?x?xf8E8M0FNU>)
    outs(%c : tensor<?x?xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>}>
    } {
    ^bb0(%in: f8E8M0FNU, %in_4: f8E8M0FNU, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: f32):
      %17 = arith.scaling_extf %in, %in_5 : f8E8M0FNU, f8E8M0FNU to f32
      %18 = arith.scaling_extf %in_4, %in_6 : f8E8M0FNU, f8E8M0FNU to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @scaled_mfma_32x32x64
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f8E8M0FNU, rhs_elem_type = f8E8M0FNU, acc_elem_type = f32>}>
//  CHECK-SAME:     permutations = [array<i64: 0, 1, 2>, array<i64: 2, 0, 1>, array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
//  CHECK-SAME:     : tensor<?x?x?x32x2x32xf8E8M0FNU>, tensor<?x?x?x32x2x32xf8E8M0FNU>, tensor<?x?x32x2xf8E8M0FNU>, tensor<?x?x32x2xf8E8M0FNU> into tensor<?x?x32x32xf32>

// -----

// Verify that scaled_mma_layout with repeats produces inner_tiled ops with
// correct shapes and preserves the repeats attribute through the pass.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @scaled_mfma_16x16x128_with_k_repeats(%a: tensor<?x?x?xf4E2M1FN>, %b: tensor<?x?x?xf4E2M1FN>, %a_scales: tensor<?x?xf8E8M0FNU>, %b_scales: tensor<?x?xf8E8M0FNU>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b, %a_scales, %b_scales : tensor<?x?x?xf4E2M1FN>, tensor<?x?x?xf4E2M1FN>, tensor<?x?xf8E8M0FNU>, tensor<?x?xf8E8M0FNU>)
    outs(%c : tensor<?x?xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, repeats = [1, 1, 4, 1]>}>
    } {
    ^bb0(%in: f4E2M1FN, %in_4: f4E2M1FN, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: f32):
      %17 = arith.scaling_extf %in, %in_5 : f4E2M1FN, f8E8M0FNU to f32
      %18 = arith.scaling_extf %in_4, %in_6 : f4E2M1FN, f8E8M0FNU to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @scaled_mfma_16x16x128_with_k_repeats
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps =
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d1, d2)>
//  CHECK-SAME:       affine_map<(d0, d1, d2, d3) -> (d0, d1)>
//  CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, repeats = [1, 1, 4, 1]>}>
//  CHECK-SAME:     permutations = [array<i64: 0, 1, 2>, array<i64: 2, 0, 1>, array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
//  CHECK-SAME:     : tensor<?x?x?x16x16x32xf4E2M1FN>, tensor<?x?x?x16x16x32xf4E2M1FN>, tensor<?x?x16x16xf8E8M0FNU>, tensor<?x?x16x16xf8E8M0FNU> into tensor<?x?x16x16xf32>

// -----

// Baseline static-shape test WITHOUT repeats (implicitly [1, 1, 1, 1]).
// Same input shapes as scaled_mfma_16x16x128_k_repeats_static below.
//
// Base intrinsic MFMA_SCALE_F32_16x16x128_B32:
//   M=16, N=16, K_total=128 (kScale=4, blockSize=32)
//
// With kScale=4, the K=16 input dimension yields 4 outer K tiles (16/4=4).
// Compare with the repeats=[1,1,4,1] test below where the same K=16 yields
// only 1 outer K tile (16/16=1) because the grouped kScale is 4*4=16.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @scaled_mfma_16x16x128_no_repeats_static(
      %a: tensor<32x16x32xf4E2M1FN>, %b: tensor<48x16x32xf4E2M1FN>,
      %a_scales: tensor<32x16xf8E8M0FNU>, %b_scales: tensor<48x16xf8E8M0FNU>,
      %c: tensor<32x48xf32>) -> tensor<32x48xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b, %a_scales, %b_scales : tensor<32x16x32xf4E2M1FN>, tensor<48x16x32xf4E2M1FN>, tensor<32x16xf8E8M0FNU>, tensor<48x16xf8E8M0FNU>)
    outs(%c : tensor<32x48xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>}>
    } {
    ^bb0(%in: f4E2M1FN, %in_4: f4E2M1FN, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: f32):
      %17 = arith.scaling_extf %in, %in_5 : f4E2M1FN, f8E8M0FNU to f32
      %18 = arith.scaling_extf %in_4, %in_6 : f4E2M1FN, f8E8M0FNU to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<32x48xf32>
    return %mm : tensor<32x48xf32>
  }
}

// Without repeats: inner K tile = 4 (base kScale), so K=16 yields 4 outer
// K tiles (16/4=4). Inner tiles are [16, 4, 32] for data, [16, 4] for scales.
// CHECK-LABEL: func.func @scaled_mfma_16x16x128_no_repeats_static
//   CHECK-DAG:   %[[A_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1, 2] inner_tiles = [16, 4, 32] into %{{.+}} : tensor<32x16x32xf4E2M1FN> -> tensor<2x4x1x16x4x32xf4E2M1FN>
//   CHECK-DAG:   %[[B_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1, 2] inner_tiles = [16, 4, 32] into %{{.+}} : tensor<48x16x32xf4E2M1FN> -> tensor<3x4x1x16x4x32xf4E2M1FN>
//   CHECK-DAG:   %[[AS_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 4] into %{{.+}} : tensor<32x16xf8E8M0FNU> -> tensor<2x4x16x4xf8E8M0FNU>
//   CHECK-DAG:   %[[BS_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 4] into %{{.+}} : tensor<48x16xf8E8M0FNU> -> tensor<3x4x16x4xf8E8M0FNU>
//   CHECK-DAG:   %[[C_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %{{.+}} : tensor<32x48xf32> -> tensor<2x3x16x16xf32>
//       CHECK:   iree_codegen.inner_tiled ins(%[[A_PACK]], %[[B_PACK]], %[[AS_PACK]], %[[BS_PACK]]) outs(%[[C_PACK]])
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32>
//  CHECK-SAME:     : tensor<2x4x1x16x4x32xf4E2M1FN>, tensor<3x4x1x16x4x32xf4E2M1FN>, tensor<2x4x16x4xf8E8M0FNU>, tensor<3x4x16x4xf8E8M0FNU> into tensor<2x3x16x16xf32>

// -----

// Static-shape test: verify that repeats = [1, 1, 4, 1] scales the grouped
// intrinsic shape from 16x16x128 to 16x16x512.
//
// Base intrinsic MFMA_SCALE_F32_16x16x128_B32:
//   M=16, N=16, K_total=128 (kScale=4, blockSize=32, so 4*32=128)
// With repeats=[1,1,4,1]:
//   M=16, N=16, K_total=512 (kScale=4*4=16, blockSize=32, so 16*32=512)
//
// Inputs use kScale=16, KB=32 so K_total = 16*32 = 512, exactly one
// grouped tile along K. The pack inner tiles [16, 16, 32] confirm kScale=16.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
module {
  func.func @scaled_mfma_16x16x128_k_repeats_static(
      %a: tensor<32x16x32xf4E2M1FN>, %b: tensor<48x16x32xf4E2M1FN>,
      %a_scales: tensor<32x16xf8E8M0FNU>, %b_scales: tensor<48x16xf8E8M0FNU>,
      %c: tensor<32x48xf32>) -> tensor<32x48xf32> {
    %mm = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]
    } ins(%a, %b, %a_scales, %b_scales : tensor<32x16x32xf4E2M1FN>, tensor<48x16x32xf4E2M1FN>, tensor<32x16xf8E8M0FNU>, tensor<48x16xf8E8M0FNU>)
    outs(%c : tensor<32x48xf32>) attrs =  {
      lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, repeats = [1, 1, 4, 1]>}>
    } {
    ^bb0(%in: f4E2M1FN, %in_4: f4E2M1FN, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: f32):
      %17 = arith.scaling_extf %in, %in_5 : f4E2M1FN, f8E8M0FNU to f32
      %18 = arith.scaling_extf %in_4, %in_6 : f4E2M1FN, f8E8M0FNU to f32
      %19 = arith.mulf %17, %18 : f32
      %20 = arith.addf %out, %19 : f32
      linalg.yield %20 : f32
    } -> tensor<32x48xf32>
    return %mm : tensor<32x48xf32>
  }
}

// K_total = kScale(16) * KB(32) = 512, matching the expected 16x16x512
// grouped intrinsic. Outer K=1, outer KB=1 (exactly one grouped tile).
// Inner tiles [16, 16, 32] show kScale=16 (4x the base kScale=4).
// CHECK-LABEL: func.func @scaled_mfma_16x16x128_k_repeats_static
//   CHECK-DAG:   %[[A_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1, 2] inner_tiles = [16, 16, 32] into %{{.+}} : tensor<32x16x32xf4E2M1FN> -> tensor<2x1x1x16x16x32xf4E2M1FN>
//   CHECK-DAG:   %[[B_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1, 2] inner_tiles = [16, 16, 32] into %{{.+}} : tensor<48x16x32xf4E2M1FN> -> tensor<3x1x1x16x16x32xf4E2M1FN>
//   CHECK-DAG:   %[[AS_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %{{.+}} : tensor<32x16xf8E8M0FNU> -> tensor<2x1x16x16xf8E8M0FNU>
//   CHECK-DAG:   %[[BS_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %{{.+}} : tensor<48x16xf8E8M0FNU> -> tensor<3x1x16x16xf8E8M0FNU>
//   CHECK-DAG:   %[[C_PACK:.+]] = linalg.pack %{{.+}} inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %{{.+}} : tensor<32x48xf32> -> tensor<2x3x16x16xf32>
//       CHECK:   iree_codegen.inner_tiled ins(%[[A_PACK]], %[[B_PACK]], %[[AS_PACK]], %[[BS_PACK]]) outs(%[[C_PACK]])
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, repeats = [1, 1, 4, 1]>
//  CHECK-SAME:     : tensor<2x1x1x16x16x32xf4E2M1FN>, tensor<3x1x1x16x16x32xf4E2M1FN>, tensor<2x1x16x16xf8E8M0FNU>, tensor<3x1x16x16xf8E8M0FNU> into tensor<2x3x16x16xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @no_hoist_mismatched_pack_unpack(%arg0 : tensor<4x8x16x4xf32>, %arg1 : tensor<8x4x16x4xf32>, %arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>{
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %1 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
      %2 = tensor.empty() : tensor<4x4x16x16xf32>
      %pack = linalg.pack %arg4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x64xf32> -> tensor<4x4x16x16xf32>
      %3 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x4x16x4xf32> into tensor<4x4x16x16xf32>
      %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg4 : tensor<4x4x16x16xf32> -> tensor<64x64xf32>
      %empty = tensor.empty() : tensor<32x16x2x4xf32>
      %pack_2 = linalg.pack %unpack inner_dims_pos = [0, 1] inner_tiles = [2, 4] into %empty : tensor<64x64xf32> -> tensor<32x16x2x4xf32>
      %empty_2 = tensor.empty() : tensor<16x32x4x2xf32>
      %transpose = linalg.transpose ins(%pack_2 : tensor<32x16x2x4xf32>) outs(%empty_2 : tensor<16x32x4x2xf32>) permutation = [1, 0, 3, 2]
      %empty_3 = tensor.empty() : tensor<64x64xf32>
      %unpack_2 = linalg.unpack %transpose outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 2] into %empty_3 : tensor<16x32x4x2xf32> -> tensor<64x64xf32>
      scf.yield %unpack_2 : tensor<64x64xf32>
    }
  return %1 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @no_hoist_mismatched_pack_unpack
//       CHECK:   %[[FOR_RESULT:.+]] = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%[[ARG4:.+]] = %arg2) -> (tensor<64x64xf32>)
//       CHECK:     %[[PACK:.+]] = linalg.pack %[[ARG4]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %1 : tensor<64x64xf32> -> tensor<4x4x16x16xf32>
//       CHECK:     %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[PACK]])
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %transposed outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [4, 2] into %5 : tensor<16x32x4x2xf32> -> tensor<64x64xf32>
//       CHECK:     scf.yield %[[UNPACK]] : tensor<64x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @no_hoist_pack_unpack_otherusers(%arg1 : tensor<8x4x16x4xf32>, %arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>{
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %padding_value = arith.constant 0.000000e+00 : f32
  %1 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg2) -> (tensor<64x64xf32>) {
    %empty = tensor.empty() : tensor<4x8x16x4xf32>
    %slice = tensor.extract_slice %arg4[0, 0][64, 32][1, 1] : tensor<64x64xf32> to tensor<64x32xf32>
    %pack_1 = linalg.pack %slice inner_dims_pos = [0, 1] inner_tiles = [16, 4] into %empty : tensor<64x32xf32> -> tensor<4x8x16x4xf32>
    %2 = tensor.empty() : tensor<4x4x16x16xf32>
    %pack = linalg.pack %arg4 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x64xf32> -> tensor<4x4x16x16xf32>
    %3 = iree_codegen.inner_tiled ins(%pack_1, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x4x16x4xf32> into tensor<4x4x16x16xf32>
    %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg4 : tensor<4x4x16x16xf32> -> tensor<64x64xf32>
    scf.yield %unpack : tensor<64x64xf32>
    }
  return %1 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @no_hoist_pack_unpack_otherusers
//       CHECK:   %[[FOR_RESULT:.+]] = scf.for %arg2 = %c0 to %c512 step %c32 iter_args(%[[ARG3:.+]] = %arg1) -> (tensor<64x64xf32>)
//       CHECK:     %[[PACK:.+]] = linalg.pack %[[ARG3]] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x64xf32> -> tensor<4x4x16x16xf32>
//       CHECK:     %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%pack, %arg0) outs(%[[PACK]])
//       CHECK:     %[[UNPACK:.+]] = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[ARG3]] : tensor<4x4x16x16xf32> -> tensor<64x64xf32>
//       CHECK:     scf.yield %[[UNPACK]] : tensor<64x64xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @hoist_multiple_pack_unpack(%arg0 : tensor<4x8x16x4xf32>, %arg1 : tensor<8x8x16x4xf32>, %arg2 : tensor<64x127xf32>, %input : tensor<64x128xf32>) -> (tensor<64x127xf32>, tensor<64x128xf32>){
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %padding_value = arith.constant 0.000000e+00 : f32
  %1, %result0 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg2, %arg5 = %input) -> (tensor<64x127xf32>, tensor<64x128xf32>) {
      %2 = tensor.empty() : tensor<4x8x16x16xf32>
      %pack = linalg.pack %arg4 padding_value(%padding_value : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %2 : tensor<64x127xf32> -> tensor<4x8x16x16xf32>
      %empty = tensor.empty() : tensor<4x8x16x16xf32>
      %pack_1 = linalg.pack %arg5 padding_value(%padding_value : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %empty : tensor<64x128xf32> -> tensor<4x8x16x16xf32>
      %3 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %4 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack_1) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %unpack = linalg.unpack %3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg4 : tensor<4x8x16x16xf32> -> tensor<64x127xf32>
      %unpack_1 = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg5 : tensor<4x8x16x16xf32> -> tensor<64x128xf32>
      scf.yield %unpack, %unpack_1 : tensor<64x127xf32>, tensor<64x128xf32>
    }
  return %1, %result0 : tensor<64x127xf32>, tensor<64x128xf32>
}

// CHECK-LABEL: func.func @hoist_multiple_pack_unpack
//       CHECK:   %[[A_PACK:.+]] = linalg.pack
//  CHECK-SAME:     outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16]
//       CHECK:   %[[B_PACK:.+]] = linalg.pack %arg3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<64x128xf32> -> tensor<4x8x16x16xf32>
//       CHECK:   %[[FOR_RESULT:.+]]:2 = scf.for %arg4 = %c0 to %c512 step %c32 iter_args(%[[ARG5:.+]] = %[[A_PACK]], %[[ARG6:.+]] = %[[B_PACK]]) -> (tensor<4x8x16x16xf32>, tensor<4x8x16x16xf32>)
//       CHECK:     %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[ARG5]])
//       CHECK:     %[[INNER_TILED_RESULT_1:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[ARG6]])
//       CHECK:     scf.yield %[[INNER_TILED_RESULT]], %[[INNER_TILED_RESULT_1]] : tensor<4x8x16x16xf32>,  tensor<4x8x16x16xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<64x128xf32>
//       CHECK:   linalg.unpack %[[FOR_RESULT]]#1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[EMPTY]] : tensor<4x8x16x16xf32> -> tensor<64x128xf32>
//       CHECK:   %[[EMPTY_1:.+]] = tensor.empty() : tensor<64x127xf32>
//       CHECK:   linalg.unpack %[[FOR_RESULT]]#0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[EMPTY_1]] : tensor<4x8x16x16xf32> -> tensor<64x127xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @hoist_selective_pack_unpack(%lhs_input : tensor<4x8x16x4xf32>, %rhs_input : tensor<8x8x16x4xf32>, %arg0 : tensor<64x128xf32>, %arg1 : tensor<4x8x16x16xf32>, %arg2 : tensor<64x127xf32>) -> (tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<64x127xf32>){
  %c512 = arith.constant 512 : index
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %padding_value = arith.constant 0.000000e+00 : f32
  %result0, %result1, %result2 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2) -> (tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<64x127xf32>) {
      %empty = tensor.empty() : tensor<4x8x16x16xf32>
      %pack_1 = linalg.pack %arg4 padding_value(%padding_value : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %empty : tensor<64x128xf32> -> tensor<4x8x16x16xf32>
      %empty_1 = tensor.empty() : tensor<4x8x16x16xf32>
      %pack = linalg.pack %arg6 padding_value(%padding_value : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %empty_1 : tensor<64x127xf32> -> tensor<4x8x16x16xf32>
      %3 = iree_codegen.inner_tiled ins(%lhs_input, %rhs_input) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %4 = iree_codegen.inner_tiled ins(%lhs_input, %rhs_input) outs(%pack_1) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %5 = iree_codegen.inner_tiled ins(%lhs_input, %rhs_input) outs(%arg5) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<4x8x16x4xf32>, tensor<8x8x16x4xf32> into tensor<4x8x16x16xf32>
      %empty_2 = tensor.empty() : tensor<64x128xf32>
      %unpack_1 = linalg.unpack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %empty_2 : tensor<4x8x16x16xf32> -> tensor<64x128xf32>
      %empty_3 = tensor.empty() : tensor<32x16x2x8xf32>
      %pack_2 = linalg.pack %unpack_1 inner_dims_pos = [0, 1] inner_tiles = [2, 8] into %empty_3 : tensor<64x128xf32> -> tensor<32x16x2x8xf32>
      %empty_4 = tensor.empty() : tensor<16x32x8x2xf32>
      %transpose = linalg.transpose ins(%pack_2 : tensor<32x16x2x8xf32>) outs(%empty_4 : tensor<16x32x8x2xf32>) permutation = [1, 0, 3, 2]
      %unpack_2 = linalg.unpack %transpose outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 2] into %arg4 : tensor<16x32x8x2xf32> -> tensor<64x128xf32>
      %unpack = linalg.unpack %3 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %arg6 : tensor<4x8x16x16xf32> -> tensor<64x127xf32>
      scf.yield %unpack_2, %5, %unpack : tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<64x127xf32>
    }
  return %result0, %result1, %result2 : tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<64x127xf32>
}

// CHECK-LABEL: func.func @hoist_selective_pack_unpack
//       CHECK:   %[[A_PACK:.+]] = linalg.pack
//  CHECK-SAME:     outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<64x127xf32> -> tensor<4x8x16x16xf32>
//       CHECK:   %[[FOR_RESULT:.+]]:3 = scf.for %arg5 = %c0 to %c512 step %c32 iter_args(%[[ARG6:.+]] = %arg2, %[[ARG7:.+]] = %arg3, %[[ARG8:.+]] = %[[A_PACK]]) -> (tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<4x8x16x16xf32>)
//       CHECK:     %[[B_PACK:.+]] = linalg.pack %[[ARG6]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %0 : tensor<64x128xf32> -> tensor<4x8x16x16xf32>
//       CHECK:     %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[ARG8]])
//       CHECK:     %[[INNER_TILED_RESULT_1:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[B_PACK]])
//       CHECK:     %[[INNER_TILED_RESULT_2:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[ARG7]])
//       CHECK:     %[[UNPACK_3:.+]] = linalg.unpack %transposed outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [8, 2] into %[[ARG6]] : tensor<16x32x8x2xf32> -> tensor<64x128xf32>
//       CHECK:     scf.yield %[[UNPACK_3]], %[[INNER_TILED_RESULT_2]], %[[INNER_TILED_RESULT]] : tensor<64x128xf32>, tensor<4x8x16x16xf32>, tensor<4x8x16x16xf32>
//       CHECK:   %[[EMPTY_1:.+]] = tensor.empty() : tensor<64x127xf32>
//       CHECK:   linalg.unpack %[[FOR_RESULT]]#2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %[[EMPTY_1]] : tensor<4x8x16x16xf32> -> tensor<64x127xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
func.func @hoist_pack_unpack_multiple_loop(%arg0 : tensor<1x1x4x2x16x16xbf16>, %arg1 : tensor<2x1x1x2x16x16xbf16>, %arg2 : tensor<1x1x64x32xf32>) -> (tensor<1x1x64x32xf32>){
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%arg6 = %arg2) -> (tensor<1x1x64x32xf32>) {
    %1 = scf.for %arg7 = %c0 to %c3 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x1x64x32xf32>) {
      %empty = tensor.empty() : tensor<1x1x4x2x16x16xf32>
      %pack = linalg.pack %arg8 inner_dims_pos = [2, 3] inner_tiles = [16, 16] into %empty : tensor<1x1x64x32xf32> -> tensor<1x1x4x2x16x16xf32>
      %2 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%pack) {indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>], kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>, semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>} : tensor<1x1x4x2x16x16xbf16>, tensor<2x1x1x2x16x16xbf16> into tensor<1x1x4x2x16x16xf32>
      %unpack = linalg.unpack %2 inner_dims_pos = [2, 3] inner_tiles = [16, 16] into %arg8 : tensor<1x1x4x2x16x16xf32> -> tensor<1x1x64x32xf32>
      scf.yield %unpack : tensor<1x1x64x32xf32>
    }
    scf.yield %1 : tensor<1x1x64x32xf32>
  }
  return %0 : tensor<1x1x64x32xf32>
}

// CHECK-LABEL: func.func @hoist_pack_unpack_multiple_loop
//       CHECK:   %[[PACK:.+]] = linalg.pack %arg2 inner_dims_pos = [2, 3] inner_tiles = [16, 16] into %0 : tensor<1x1x64x32xf32> -> tensor<1x1x4x2x16x16xf32>
//       CHECK:   %[[OUTER_FOR_RESULT:.+]] = scf.for %arg3 = %c0 to %c3 step %c1 iter_args(%[[ARG4:.+]] = %[[PACK]]) -> (tensor<1x1x4x2x16x16xf32>) {
//       CHECK:     %[[INNER_FOR_RESULT:.+]] = scf.for %arg5 = %c0 to %c3 step %c1 iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x1x4x2x16x16xf32>)
//       CHECK:       %[[INNER_TILED_RESULT:.+]] = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%[[ARG6]])
//       CHECK:       scf.yield %[[INNER_TILED_RESULT:.+]] : tensor<1x1x4x2x16x16xf32>
//       CHECK:     scf.yield %[[INNER_FOR_RESULT]] : tensor<1x1x4x2x16x16xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x64x32xf32>
//       CHECK:   linalg.unpack %[[OUTER_FOR_RESULT]] inner_dims_pos = [2, 3] inner_tiles = [16, 16] into %[[EMPTY:.+]] : tensor<1x1x4x2x16x16xf32> -> tensor<1x1x64x32xf32>

// -----

func.func @propagate_extract_basic(%input : tensor<128xf32>, %arg0 : index, %arg1 : tensor<?xbf16>) -> tensor<?xbf16> {
  %empty = util.optimization_barrier %input : tensor<128xf32>
  %extracted_slice =  tensor.extract_slice %empty[%arg0] [%arg0] [1] : tensor<128xf32> to tensor<?xf32>
  %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<?xf32>) outs(%arg1 : tensor<?xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<?xbf16>
  return %generic : tensor<?xbf16>
}

// CHECK-LABEL: func.func @propagate_extract_basic
//       CHECK:  linalg.generic
//       CHECK:  tensor.extract_slice

// -----

func.func @no_propagate_extract_blockargument(%input : tensor<128xf32>, %arg0 : index, %arg1 : tensor<?xbf16>) -> tensor<?xbf16> {
  %extracted_slice =  tensor.extract_slice %input[%arg0] [%arg0] [1] : tensor<128xf32> to tensor<?xf32>
  %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<?xf32>) outs(%arg1 : tensor<?xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %1 = arith.truncf %in : f32 to bf16
    linalg.yield %1 : bf16
  } -> tensor<?xbf16>
  return %generic : tensor<?xbf16>
}

// CHECK-LABEL: func.func @no_propagate_extract_blockargument
//       CHECK:  tensor.extract_slice
//       CHECK:  linalg.generic

// -----

func.func @no_propagate_extract_differentblock_1(%input : tensor<128xf32>, %arg0 : index, %arg1 : tensor<?xbf16>) -> tensor<?xbf16> {
  %empty = util.optimization_barrier %input : tensor<128xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %for = scf.for %arg2 = %c0 to %c32 step %arg0 iter_args(%arg4 = %arg1) -> tensor<?xbf16> {
    %extracted_slice =  tensor.extract_slice %empty[%arg0] [%arg0] [1] : tensor<128xf32> to tensor<?xf32>
    %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<?xf32>) outs(%arg4 : tensor<?xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %1 = arith.truncf %in : f32 to bf16
      linalg.yield %1 : bf16
    } -> tensor<?xbf16>
    scf.yield %generic : tensor<?xbf16>
  }
  return %for : tensor<?xbf16>
}

// CHECK-LABEL: func.func @no_propagate_extract_differentblock_1
//       CHECK:  tensor.extract_slice
//       CHECK:  linalg.generic

// -----

func.func @no_propagate_extract_differentblock_2(%input : tensor<128xf32>, %arg0 : index, %arg1 : tensor<?xbf16>) -> tensor<?xbf16> {
  %empty = util.optimization_barrier %input : tensor<128xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %extracted_slice =  tensor.extract_slice %empty[%arg0] [%arg0] [1] : tensor<128xf32> to tensor<?xf32>
  %for = scf.for %arg2 = %c0 to %c32 step %arg0 iter_args(%arg4 = %arg1) -> tensor<?xbf16> {
    %generic = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%extracted_slice : tensor<?xf32>) outs(%arg4 : tensor<?xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %1 = arith.truncf %in : f32 to bf16
      linalg.yield %1 : bf16
    } -> tensor<?xbf16>
    scf.yield %generic : tensor<?xbf16>
  }
  return %for : tensor<?xbf16>
}

// CHECK-LABEL: func.func @no_propagate_extract_differentblock_2
//       CHECK:  tensor.extract_slice
//       CHECK:  linalg.generic
