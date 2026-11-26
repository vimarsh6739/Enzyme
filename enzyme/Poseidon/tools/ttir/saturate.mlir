tt.func private @saturate(%x: tensor<32xf32>) -> tensor<32xf32> {
  %one = arith.constant dense<1.0> : tensor<32xf32>
  %d = arith.addf %one, %x : tensor<32xf32>
  %y = arith.divf %x, %d : tensor<32xf32>
  tt.return %y : tensor<32xf32>
}
