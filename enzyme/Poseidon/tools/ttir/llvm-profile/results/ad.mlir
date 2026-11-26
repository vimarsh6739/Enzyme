module {
  tt.func private @response(%arg0: f64) -> f64 attributes {noinline = true} {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.addf %arg0, %cst : f64
    %1 = arith.mulf %0, %0 : f64
    %2 = arith.mulf %arg0, %arg0 : f64
    %3 = arith.subf %1, %2 : f64
    tt.return %3 : f64
  }
  tt.func private @response_jvp(%arg0: f64, %arg1: f64) -> f64 attributes {noinline = true} {
    %cst = arith.constant 1.000000e+00 : f64
    %0 = arith.addf %arg0, %cst : f64
    %1 = arith.mulf %arg1, %0 fastmath<fast> : f64
    %2 = arith.addf %1, %1 fastmath<fast> : f64
    %3 = arith.mulf %arg1, %arg0 fastmath<fast> : f64
    %4 = arith.addf %3, %3 fastmath<fast> : f64
    %5 = arith.subf %2, %4 fastmath<fast> : f64
    tt.return %5 : f64
  }
}
