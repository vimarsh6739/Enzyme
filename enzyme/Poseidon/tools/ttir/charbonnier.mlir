tt.func private @charbonnier(%x: tensor<32xf32>) -> tensor<32xf32> {
  %one = arith.constant dense<1.0> : tensor<32xf32>
  %sq = arith.mulf %x, %x : tensor<32xf32>
  %d = arith.addf %one, %sq : tensor<32xf32>
  %s = math.sqrt %d : tensor<32xf32>
  %y = arith.subf %s, %one : tensor<32xf32>
  tt.return %y : tensor<32xf32>
}
