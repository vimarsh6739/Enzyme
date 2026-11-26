// REQUIRES: poseidon-triton
// RUN: %poseidon_mlir_opt %s --pass-pipeline='builtin.module(tt.func(poseidon-rewrite{remark-fpcore}))' --verify-diagnostics -o /dev/null

tt.func @log_ratio(%x: tensor<32xf32>) -> tensor<32xf32> {
  %one = arith.constant dense<1.0> : tensor<32xf32>
  %sum = arith.addf %x, %one : tensor<32xf32>
  // expected-remark@below {{(FPCore (x0) :name "log_ratio/0" :precision binary32 (/ x0 (+ x0 1)))}}
  %ratio = arith.divf %x, %sum : tensor<32xf32>
  tt.return %ratio : tensor<32xf32>
}
