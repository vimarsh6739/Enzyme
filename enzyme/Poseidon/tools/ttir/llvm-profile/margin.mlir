tt.func private @response(%x: f64) -> f64 attributes {noinline = true} {
  %one = arith.constant 1.0 : f64
  %r = arith.addf %x, %one : f64
  %rr = arith.mulf %r, %r : f64
  %xx = arith.mulf %x, %x : f64
  %gap = arith.subf %rr, %xx : f64
  tt.return %gap : f64
}
