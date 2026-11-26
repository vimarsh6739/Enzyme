"""Run each generated primal and JVP on CUDA and compare to a Decimal oracle."""

import argparse
import csv
from decimal import Decimal, localcontext
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import triton


def entry_point(name):
    return f"""
  tt.func public @kernel(%x: !tt.ptr<f32>, %dx: !tt.ptr<f32>, %y: !tt.ptr<f32>, %dy: !tt.ptr<f32>, %ddy: !tt.ptr<f32>) {{
    %pid = tt.get_program_id x : i32
    %size = arith.constant 32 : i32
    %base = arith.muli %pid, %size : i32
    %row = tt.splat %base : i32 -> tensor<32xi32>
    %off = tt.make_range {{start = 0 : i32, end = 32 : i32}} : tensor<32xi32>
    %i = arith.addi %row, %off : tensor<32xi32>
    %px = tt.splat %x : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %tx = tt.splat %dx : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %py = tt.splat %y : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %ty = tt.splat %dy : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %hy = tt.splat %ddy : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %h = tt.addptr %hy, %i : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %p = tt.addptr %px, %i : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %q = tt.addptr %tx, %i : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %r = tt.addptr %py, %i : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %s = tt.addptr %ty, %i : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %value = tt.load %p : tensor<32x!tt.ptr<f32>>
    %tangent = tt.load %q : tensor<32x!tt.ptr<f32>>
    %loss = tt.call @{name}(%value) : (tensor<32xf32>) -> tensor<32xf32>
    %jvp = tt.call @jvp(%value, %tangent) : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %one = arith.constant dense<1.0> : tensor<32xf32>
    %hvp = tt.call @hvp(%value, %one, %tangent) : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
    tt.store %h, %hvp : tensor<32x!tt.ptr<f32>>
    tt.store %r, %loss : tensor<32x!tt.ptr<f32>>
    tt.store %s, %jvp : tensor<32x!tt.ptr<f32>>
    tt.return
  }}
"""


def kernel(args, path, name):
    ir = path.read_text().rstrip()
    if not ir.endswith("}"):
        raise ValueError(f"expected MLIR module in {path}")
    ttir = path.with_suffix(".ttir")
    subprocess.run([args.enzyme_opt, "--inline", "--symbol-dce", "--canonicalize", "--cse", "-o", ttir],
                   input=ir[:-1] + entry_point(name) + "}", text=True, check=True)
    compiled = triton.compile(str(ttir), options={"num_warps": 4, "enable_fp_fusion": False})
    path.with_suffix(".ptx").write_text(compiled.asm["ptx"])
    return compiled


def references(name, xs, tangents):
    values, derivatives, second = [], [], []
    with localcontext() as ctx:
        ctx.prec = 100
        for value, tangent in zip(xs, tangents):
            x, dx = Decimal.from_float(float(value)), Decimal.from_float(float(tangent))
            if name == "saturate":
                y, dy, ddy = x / (1 + x), dx / (1 + x)**2, -2*dx / (1 + x)**3
            else:
                norm = (1 + x*x).sqrt()
                y, dy, ddy = x*x / (1 + norm), dx*x / norm, dx / norm**3
            values.append(float(y))
            derivatives.append(float(dy))
            second.append(float(ddy))
    return np.array(values), np.array(derivatives), np.array(second)


def ulps(actual, expected):
    # Map finite IEEE words to increasing integers, merging signed zero.
    def ordered(v):
        bits = np.asarray(v, dtype=np.float32).view(np.uint32).astype(np.int64)
        return np.where(bits >= 2**31, 2**31 - (bits % 2**31), 2**31 + bits)
    return np.abs(ordered(actual) - ordered(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enzyme-opt", default="enzymexlamlir-opt")
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    metadata = {"gpu": torch.cuda.get_device_name(), "triton": triton.__version__,
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "fp_fusion": False,
                "compiler": subprocess.check_output([args.enzyme_opt, "--version"], text=True).strip()}
    (args.run / "device.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.run / "accuracy.csv").open("w", newline="") as raw, (args.run / "metrics.csv").open("w", newline="") as summary:
        rows = csv.writer(raw, lineterminator="\n")
        metrics = csv.writer(summary, lineterminator="\n")
        rows.writerow(["case", "schedule", "output", "x", "dx", "observed", "reference", "ulp_distance"])
        metrics.writerow(["case", "schedule", "output", "samples", "wrong_zero", "median_ulp", "p95_ulp", "max_ulp"])
        for name in ("saturate", "charbonnier"):
            xs = np.geomspace(2.0**-27, 2.0**28, 256).astype(np.float32)
            tangents = np.resize(np.array([1, -0.75, 0.5, 1.25], dtype=np.float32), xs.size)
            if name == "charbonnier":
                xs[1::2] *= -1
            refs = references(name, xs, tangents)
            x, dx = (torch.as_tensor(v, device="cuda") for v in (xs, tangents))
            for schedule in ("base", "pre", "post", "both"):
                fn = kernel(args, args.run / name / (schedule + "-hvp.mlir"), name)
                ys = [torch.full_like(x, float("nan")) for _ in range(3)]
                fn[(xs.size // 32, 1, 1)](x, dx, *ys)
                for output, y, ref in zip(("primal", "jvp", "hvp"), ys, refs):
                    obs = y.cpu().numpy()
                    if not np.all(np.isfinite(obs)):
                        raise ValueError(f"non-finite {name} {schedule} {output}")
                    errors = ulps(obs, ref)
                    for i in range(xs.size):
                        rows.writerow([name, schedule, output, float(xs[i]), float(tangents[i]),
                                       float(obs[i]), ref[i], errors[i]])
                    metrics.writerow([name, schedule, output, xs.size, np.sum((obs == 0) & (ref != 0)),
                                      np.median(errors), np.percentile(errors, 95), np.max(errors)])


if __name__ == "__main__":
    main()
