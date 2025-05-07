!matA = tensor<32000x4096xf16>
!matB = tensor<2x4096xf16>
!matCF16 = tensor<32000x2xf16>
!matCF32 = tensor<32000x2xf32>

func.func @matvec(%arg0: !matA, %arg1: !matB) -> !matCF16 {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %5 = tensor.empty() : !matCF32
  %6 = linalg.fill ins(%cst : f16) outs(%5 : !matCF32) -> !matCF32
  %7 = linalg.matmul_transpose_b ins(%arg0, %arg1 : !matA, !matB) outs(%6 : !matCF32) -> !matCF32
  %8 = arith.truncf %7 : !matCF32 to !matCF16
  return %8 : !matCF16
}