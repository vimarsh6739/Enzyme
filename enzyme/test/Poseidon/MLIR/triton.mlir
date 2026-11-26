// REQUIRES: poseidon-triton
// RUN: %poseidon_mlir_opt %s --poseidon-rewrite="remark-fpcore only-function=missing" -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=NONE
// NONE-NOT: FPCore
// RUN: %poseidon_mlir_opt %s --pass-pipeline='builtin.module(tt.func(poseidon-rewrite{remark-fpcore}))' -o /dev/null 2>&1 | FileCheck %s

tt.func @log_ratio(%x: tensor<32xf32>) -> tensor<32xf32> {
  %one = arith.constant dense<1.0> : tensor<32xf32>
  %sum = arith.addf %x, %one : tensor<32xf32>
  // CHECK: remark: (FPCore (x0) :name "log_ratio/0" :precision binary32 (/ x0 (+ x0 1)))
  %ratio = arith.divf %x, %sum : tensor<32xf32>
  tt.return %ratio : tensor<32xf32>
}
