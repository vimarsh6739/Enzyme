"""Lower the margin and its Enzyme JVP to NVPTX LLVM and mark both sites."""

import argparse
from pathlib import Path
import re
import subprocess

import triton
from triton.backends.compiler import GPUTarget


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}")
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", default="enzymexlamlir-opt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("margin.mlir")
    primal_jvp = args.out / "ad.mlir"
    wrap = ("--enzyme-wrap=infn=response outfn=response_jvp argTys=enzyme_dup "
            "retTys=enzyme_dupnoneed mode=ForwardMode")
    subprocess.run([args.opt, source, wrap, "--cse", "--canonicalize", "-o", primal_jvp], check=True)
    module = primal_jvp.read_text().rstrip()
    if not module.endswith("}"):
        raise ValueError("expected a TTIR module")
    entry = """
      tt.func public @entry(%x: f64, %dx: f64, %yp: !tt.ptr<f64>, %gp: !tt.ptr<f64>) {
        %v = tt.call @response(%x) : (f64) -> f64
        %j = tt.call @response_jvp(%x, %dx) : (f64, f64) -> f64
        tt.store %yp, %v : !tt.ptr<f64>
        tt.store %gp, %j : !tt.ptr<f64>
        tt.return
      }
    """
    ttir = args.out / "margin.ttir"
    ttir.write_text(module[:-1] + entry + "}\n")
    kernel = triton.compile(str(ttir), target=GPUTarget("cuda", 120, 32),
                            options={"num_warps": 1, "enable_fp_fusion": False})
    device = kernel.asm["llir"]
    (args.out / "baseline.ll").write_text(device)

    # These two private scalar calls are the complete set of call sites. The
    # marker calls its operand by the C calling convention; leaving Triton's
    # internal fastcc on the callee would make the generated call undefined.
    marked = device
    for name in ("response", "response_jvp"):
        marked = replace_once(marked, f"define internal fastcc double @{name}(",
                              f"define internal double @{name}(")
        call = rf"tail call fastcc double @{name}\(([^)]*)\)"
        marked, count = re.subn(call, rf"tail call double (ptr, ...) @__poseidon_fp_optimize(ptr @{name}, \1)", marked)
        if count != 1:
            raise ValueError(f"expected one call to {name}")
    # The pass checks the marked function's target-cpu against its cost CSV.
    definition = re.search(r"define internal double @response\([^\n]*? #(\d+) ", marked)
    if not definition:
        raise ValueError("missing target attributes on response")
    attrs = definition[1]
    if not re.search(rf"define internal double @response_jvp\([^\n]*? #{attrs} ", marked):
        raise ValueError("the marked functions must share the same target attributes")
    marked = replace_once(marked, f"attributes #{attrs} = {{", f'attributes #{attrs} = {{ "target-cpu"="sm_120"')
    (args.out / "marked.ll").write_text(marked + "\ndeclare double @__poseidon_fp_optimize(ptr, ...)\n")


if __name__ == "__main__":
    main()
