source_filename = "LLVMDialectModule"
target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: nounwind
define ptx_kernel void @entry(double %0, double %1, ptr addrspace(1) %2, ptr addrspace(1) %3, ptr addrspace(1) nofree readnone captures(none) %4, ptr addrspace(1) nofree readnone captures(none) %5) local_unnamed_addr #0 {
  %herbie.sub.i = fadd double %0, %0
  %herbie.sub.i1 = fadd double %1, %1
  %7 = tail call range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %8 = icmp eq i32 %7, 0
  %9 = bitcast double %herbie.sub.i to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %9, ptr addrspace(1) %2, i1 %8) #2
  %10 = bitcast double %herbie.sub.i1 to i64
  tail call void asm sideeffect "@$2 st.global.b64 [ $1 + 0 ], { $0 };", "l,l,b"(i64 %10, ptr addrspace(1) %3, i1 %8) #2
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { nounwind "nvvm.reqntid"="32" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 4, !"nvvm-reflect-ftz", i32 1}
