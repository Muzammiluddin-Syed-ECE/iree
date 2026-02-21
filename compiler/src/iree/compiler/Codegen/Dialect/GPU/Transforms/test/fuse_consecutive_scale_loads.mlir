// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-fuse-consecutive-scale-loads))' --split-input-file | FileCheck %s

// Test: 4 chained scaled_mfma ops, each with a padded single-byte scale
// (scalesIdx=0). The pass should pack the 4 bytes into one shared
// vector<4xf8E8M0FNU> and assign scalesIdx=0,1,2,3.

func.func @fuse_four_scale_loads(
    %lhs0: vector<32xf4E2M1FN>, %rhs0: vector<32xf4E2M1FN>,
    %lhs1: vector<32xf4E2M1FN>, %rhs1: vector<32xf4E2M1FN>,
    %lhs2: vector<32xf4E2M1FN>, %rhs2: vector<32xf4E2M1FN>,
    %lhs3: vector<32xf4E2M1FN>, %rhs3: vector<32xf4E2M1FN>,
    %sa0: f8E8M0FNU, %sb0: f8E8M0FNU,
    %sa1: f8E8M0FNU, %sb1: f8E8M0FNU,
    %sa2: f8E8M0FNU, %sb2: f8E8M0FNU,
    %sa3: f8E8M0FNU, %sb3: f8E8M0FNU,
    %acc: vector<4xf32>) -> vector<4xf32> {

  %zeros = arith.constant dense<1.0> : vector<4xf8E8M0FNU>

  %pa0 = vector.insert %sa0, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb0 = vector.insert %sb0, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r0 = amdgpu.scaled_mfma 16x16x128 (%pa0[0] * %lhs0) * (%pb0[0] * %rhs0) + %acc
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  %pa1 = vector.insert %sa1, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb1 = vector.insert %sb1, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r1 = amdgpu.scaled_mfma 16x16x128 (%pa1[0] * %lhs1) * (%pb1[0] * %rhs1) + %r0
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  %pa2 = vector.insert %sa2, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb2 = vector.insert %sb2, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r2 = amdgpu.scaled_mfma 16x16x128 (%pa2[0] * %lhs2) * (%pb2[0] * %rhs2) + %r1
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  %pa3 = vector.insert %sa3, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb3 = vector.insert %sb3, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r3 = amdgpu.scaled_mfma 16x16x128 (%pa3[0] * %lhs3) * (%pb3[0] * %rhs3) + %r2
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  return %r3 : vector<4xf32>
}

// All 4 ops should share a packed scale vector with ascending scalesIdx.

// CHECK-LABEL: func @fuse_four_scale_loads
//  CHECK-SAME:   %[[SA0:[a-z0-9]+]]: f8E8M0FNU, %[[SB0:[a-z0-9]+]]: f8E8M0FNU,
//  CHECK-SAME:   %[[SA1:[a-z0-9]+]]: f8E8M0FNU, %[[SB1:[a-z0-9]+]]: f8E8M0FNU,
//  CHECK-SAME:   %[[SA2:[a-z0-9]+]]: f8E8M0FNU, %[[SB2:[a-z0-9]+]]: f8E8M0FNU,
//  CHECK-SAME:   %[[SA3:[a-z0-9]+]]: f8E8M0FNU, %[[SB3:[a-z0-9]+]]: f8E8M0FNU,

//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[0] * {{.*}}) * ({{.*}}[0] * {{.*}})
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[1] * {{.*}}) * ({{.*}}[1] * {{.*}})
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[2] * {{.*}}) * ({{.*}}[2] * {{.*}})
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[3] * {{.*}}) * ({{.*}}[3] * {{.*}})

// -----

// Test: chain of 2 (should also be fused).

func.func @fuse_two_scale_loads(
    %lhs0: vector<32xf4E2M1FN>, %rhs0: vector<32xf4E2M1FN>,
    %lhs1: vector<32xf4E2M1FN>, %rhs1: vector<32xf4E2M1FN>,
    %sa0: f8E8M0FNU, %sb0: f8E8M0FNU,
    %sa1: f8E8M0FNU, %sb1: f8E8M0FNU,
    %acc: vector<4xf32>) -> vector<4xf32> {

  %zeros = arith.constant dense<1.0> : vector<4xf8E8M0FNU>

  %pa0 = vector.insert %sa0, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb0 = vector.insert %sb0, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r0 = amdgpu.scaled_mfma 16x16x128 (%pa0[0] * %lhs0) * (%pb0[0] * %rhs0) + %acc
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  %pa1 = vector.insert %sa1, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb1 = vector.insert %sb1, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r1 = amdgpu.scaled_mfma 16x16x128 (%pa1[0] * %lhs1) * (%pb1[0] * %rhs1) + %r0
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  return %r1 : vector<4xf32>
}

// CHECK-LABEL: func @fuse_two_scale_loads
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[0] * {{.*}}) * ({{.*}}[0] * {{.*}})
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[1] * {{.*}}) * ({{.*}}[1] * {{.*}})

// -----

// Test: single op (no chain, should not be modified).

func.func @single_op_noop(
    %lhs: vector<32xf4E2M1FN>, %rhs: vector<32xf4E2M1FN>,
    %sa: f8E8M0FNU, %sb: f8E8M0FNU,
    %acc: vector<4xf32>) -> vector<4xf32> {

  %zeros = arith.constant dense<1.0> : vector<4xf8E8M0FNU>

  %pa = vector.insert %sa, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %pb = vector.insert %sb, %zeros [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %r = amdgpu.scaled_mfma 16x16x128 (%pa[0] * %lhs) * (%pb[0] * %rhs) + %acc
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  return %r : vector<4xf32>
}

// CHECK-LABEL: func @single_op_noop
//       CHECK:   amdgpu.scaled_mfma 16x16x128 ({{.*}}[0] * {{.*}}) * ({{.*}}[0] * {{.*}})
//   CHECK-NOT:   [1]
//   CHECK-NOT:   [2]
//   CHECK-NOT:   [3]
