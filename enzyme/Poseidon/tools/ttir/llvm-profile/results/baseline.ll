source_filename = "LLVMDialectModule"
target datalayout = "e-p3:32:32-p4:32:32-p5:32:32-p6:32:32-p7:32:32-p101:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none)
define internal fastcc double @response(double %0) unnamed_addr #0 {
  %2 = fadd double %0, 1.000000e+00
  %3 = fmul double %2, %2
  %4 = fmul double %0, %0
  %5 = fsub double %3, %4
  ret double %5
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none)
define internal fastcc double @response_jvp(double %0, double %1) unnamed_addr #0 {
  %3 = fadd double %0, 1.000000e+00
  %4 = fmul double %1, %3
  %5 = fadd double %4, %4
  %6 = fmul double %0, %1
  %7 = fadd double %6, %6
  %8 = fsub double %5, %7
  ret double %8
}

; Function Attrs: nounwind
define ptx_kernel void @entry(double %0, double %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) nofree readnone captures(none) %4, ptr addrspace(1) nofree readnone captures(none) %5) local_unnamed_addr #1 {
  %7 = tail call fastcc double @response(double %0)
  %8 = tail call fastcc double @response_jvp(double %0, double %1)
  %9 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %10 = and i32 %9, 31
  %11 = icmp eq i32 %10, 0
  %12 = bitcast double %7 to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %12, ptr addrspace(1) %2, i1 %11) #3
  %13 = bitcast double %8 to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %13, ptr addrspace(1) %3, i1 %11) #3
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) }
attributes #1 = { nounwind "nvvm.reqntid"="32" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
