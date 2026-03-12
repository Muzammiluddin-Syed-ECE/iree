// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdgpu-coalesce-scale-reads))" %s | FileCheck %s

// Test that the pass replaces the padScales pattern (vector.extract single
// byte + vector.insert into zeros) with a contiguous dword read
// (vector.transfer_read of vector<1x1x1x4xf8E8M0FNU> + vector.shape_cast)
// and sets scalesIdx=0 on the amdgpu.scaled_mfma op.

// CHECK-LABEL: func.func @coalesce_scale_reads_basic
func.func @coalesce_scale_reads_basic(
    %data_a: vector<32xf4E2M1FN>,
    %data_b: vector<32xf4E2M1FN>,
    %acc: vector<4xf32>) -> vector<4xf32> {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %padding = ub.poison : f8E8M0FNU

  // Workgroup scale memrefs with last dim = 4 (K tiles).
  %lhs_scales_mem = memref.alloc() : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>
  %rhs_scales_mem = memref.alloc() : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>

  // Read scale vectors from LDS (mimics post-LowerIREEGPUOps IR).
  %lhs_scales = vector.transfer_read %lhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<4x1x2x1xf8E8M0FNU>
  %rhs_scales = vector.transfer_read %rhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8x1x2x1xf8E8M0FNU>

  // padScales pattern: extract single byte, insert into zeros.
  %lhs_byte = vector.extract %lhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<4x1x2x1xf8E8M0FNU>
  %lhs_padded = vector.insert %lhs_byte, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %rhs_byte = vector.extract %rhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<8x1x2x1xf8E8M0FNU>
  %rhs_padded = vector.insert %rhs_byte, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>

  // The MFMA op using padded single-byte scales.
  %result = amdgpu.scaled_mfma 16x16x128 (%lhs_padded[0] * %data_a) * (%rhs_padded[0] * %data_b) + %acc
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  // Verify: new dword reads (vector<1x1x1x4>) are created for both LHS and RHS.
  // CHECK-DAG: %[[LHS_DWORD:.*]] = vector.transfer_read {{.*}} vector<1x1x1x4xf8E8M0FNU>
  // CHECK-DAG: %[[LHS_FLAT:.*]] = vector.shape_cast %[[LHS_DWORD]] : vector<1x1x1x4xf8E8M0FNU> to vector<4xf8E8M0FNU>
  // CHECK-DAG: %[[RHS_DWORD:.*]] = vector.transfer_read {{.*}} vector<1x1x1x4xf8E8M0FNU>
  // CHECK-DAG: %[[RHS_FLAT:.*]] = vector.shape_cast %[[RHS_DWORD]] : vector<1x1x1x4xf8E8M0FNU> to vector<4xf8E8M0FNU>

  // Verify: MFMA uses the dword-read scale vectors with scalesIdx=0.
  // CHECK: amdgpu.scaled_mfma 16x16x128 (%[[LHS_FLAT]][0] * %{{.*}}) * (%[[RHS_FLAT]][0] * %{{.*}})

  return %result : vector<4xf32>
}

// -----

// Test that dword reads are cached and reused across multiple MFMAs
// that extract from the same transfer_read at the same dim0 index.

// CHECK-LABEL: func.func @coalesce_scale_reads_reuse
func.func @coalesce_scale_reads_reuse(
    %data_a: vector<32xf4E2M1FN>,
    %data_b0: vector<32xf4E2M1FN>,
    %data_b1: vector<32xf4E2M1FN>,
    %acc0: vector<4xf32>,
    %acc1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {

  %c0 = arith.constant 0 : index
  %pad = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %padding = ub.poison : f8E8M0FNU

  %lhs_scales_mem = memref.alloc() : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>
  %rhs_scales_mem = memref.alloc() : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>

  %lhs_scales = vector.transfer_read %lhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<4x1x2x1xf8E8M0FNU>
  %rhs_scales = vector.transfer_read %rhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<8x1x2x1xf8E8M0FNU>

  // Two MFMAs extracting from the same dim0=0 position (same LHS scale row).
  %lhs_byte0 = vector.extract %lhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<4x1x2x1xf8E8M0FNU>
  %lhs_padded0 = vector.insert %lhs_byte0, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %rhs_byte0 = vector.extract %rhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<8x1x2x1xf8E8M0FNU>
  %rhs_padded0 = vector.insert %rhs_byte0, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %res0 = amdgpu.scaled_mfma 16x16x128 (%lhs_padded0[0] * %data_a) * (%rhs_padded0[0] * %data_b0) + %acc0
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  // Same LHS dim0=0 (should reuse the dword read), different RHS dim0=1.
  %lhs_byte1 = vector.extract %lhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<4x1x2x1xf8E8M0FNU>
  %lhs_padded1 = vector.insert %lhs_byte1, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %rhs_byte1 = vector.extract %rhs_scales[1, 0, 0, 0] : f8E8M0FNU from vector<8x1x2x1xf8E8M0FNU>
  %rhs_padded1 = vector.insert %rhs_byte1, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %res1 = amdgpu.scaled_mfma 16x16x128 (%lhs_padded1[0] * %data_a) * (%rhs_padded1[0] * %data_b1) + %acc1
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  // LHS dword read created once (dim0=0), reused by both MFMAs.
  // CHECK: %[[LHS_DWORD:.*]] = vector.transfer_read %{{.*}} : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1x1x1x4xf8E8M0FNU>
  // CHECK: %[[LHS_FLAT:.*]] = vector.shape_cast %[[LHS_DWORD]] : vector<1x1x1x4xf8E8M0FNU> to vector<4xf8E8M0FNU>

  // Two distinct RHS dword reads: one at dim0=0, one at dim0+1.
  // CHECK-DAG: vector.transfer_read {{.*}} : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1x1x1x4xf8E8M0FNU>
  // CHECK-DAG: vector.transfer_read {{.*}} : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<workgroup>>, vector<1x1x1x4xf8E8M0FNU>

  // Both MFMAs use the same LHS dword.
  // CHECK: amdgpu.scaled_mfma {{.*}}(%[[LHS_FLAT]][0]
  // CHECK: amdgpu.scaled_mfma {{.*}}(%[[LHS_FLAT]][0]

  return %res0, %res1 : vector<4xf32>, vector<4xf32>
}

// -----

// Test that non-workgroup-memory scale operands are not transformed.

// CHECK-LABEL: func.func @no_transform_private_memory
func.func @no_transform_private_memory(
    %data_a: vector<32xf4E2M1FN>,
    %data_b: vector<32xf4E2M1FN>,
    %acc: vector<4xf32>) -> vector<4xf32> {

  %c0 = arith.constant 0 : index
  %pad = arith.constant dense<5.877470e-39> : vector<4xf8E8M0FNU>
  %padding = ub.poison : f8E8M0FNU

  // Private memory (not workgroup) -- should NOT be transformed.
  %lhs_scales_mem = memref.alloc() : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<private>>
  %rhs_scales_mem = memref.alloc() : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<private>>

  %lhs_scales = vector.transfer_read %lhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<4x2x2x4xf8E8M0FNU, #gpu.address_space<private>>, vector<4x1x2x1xf8E8M0FNU>
  %rhs_scales = vector.transfer_read %rhs_scales_mem[%c0, %c0, %c0, %c0], %padding
      {in_bounds = [true, true, true, true]}
      : memref<8x2x2x4xf8E8M0FNU, #gpu.address_space<private>>, vector<8x1x2x1xf8E8M0FNU>

  %lhs_byte = vector.extract %lhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<4x1x2x1xf8E8M0FNU>
  %lhs_padded = vector.insert %lhs_byte, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>
  %rhs_byte = vector.extract %rhs_scales[0, 0, 0, 0] : f8E8M0FNU from vector<8x1x2x1xf8E8M0FNU>
  %rhs_padded = vector.insert %rhs_byte, %pad [0] : f8E8M0FNU into vector<4xf8E8M0FNU>

  // No vector<1x1x1x4> reads should be created for private memory.
  // CHECK-NOT: vector<1x1x1x4xf8E8M0FNU>

  // The original padScales pattern should remain: insert into the MFMA.
  // CHECK: %[[LHS_INS:.*]] = vector.insert {{.*}} : f8E8M0FNU into vector<4xf8E8M0FNU>
  // CHECK: %[[RHS_INS:.*]] = vector.insert {{.*}} : f8E8M0FNU into vector<4xf8E8M0FNU>
  // CHECK: amdgpu.scaled_mfma {{.*}}(%[[LHS_INS]][0] * %{{.*}}) * (%[[RHS_INS]][0] * %{{.*}})
  %result = amdgpu.scaled_mfma 16x16x128 (%lhs_padded[0] * %data_a) * (%rhs_padded[0] * %data_b) + %acc
      : vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf8E8M0FNU>, vector<32xf4E2M1FN>, vector<4xf32>

  return %result : vector<4xf32>
}
