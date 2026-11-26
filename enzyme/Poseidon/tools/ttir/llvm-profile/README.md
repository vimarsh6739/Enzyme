# Poseidon profiles a Triton loss and its gradient on CUDA

A difference of squared distances has an exact value of 2x + 1 and a JVP of
2dx. We write the distances separately in TTIR, generate the JVP with Enzyme,
then ask the LLVM Poseidon solver to optimize each lowered device function.
Both stages see the same RTX 5090, calibration inputs and target cost model.

| GPU kernel | Primal chosen by solver | Estimated relative error | Worst observed relative error | Exact JVPs |
|---|---|---:|---:|---:|
| Untuned | (x + 1)² − x² | — | 2.71 × 10⁻⁹ | 32/32 |
| Target 10⁻¹⁷ | 1 − (−x − x) | 0 | 0 | 32/32 |
| Target 10⁻¹² | 2x | 3.10 × 10⁻¹⁶ | 26.7% | 32/32 |
| Target 10⁻³ | 1 | 7.18 × 10⁻⁷ | ≈100% | 32/32 |

For x = 1.375, the control and the 10⁻¹⁷ target both returned the exact value,
3.75. The looser targets returned 2.75 (10⁻¹²) and 1.0 (10⁻³). All tuned
JVP kernels emitted dx + dx. The solver models error over the profiled
workload. Its estimated error and the worst error on independent points are
different statistics. Mean observed primal relative errors were 1.85% and
98.15% at targets 10⁻¹² and 10⁻³, respectively. At 10⁻¹⁷, the solver
selected the exact real-algebra expression already on its frontier. All 32
CUDA outputs then matched the reference. Running `z3 margin.smt2` checks
that this candidate and the selected JVP preserve the real-arithmetic
identities for arbitrary inputs. This is a reason to validate value and gradient
outputs individually before accepting a candidate.

![Held-out margin error on the RTX 5090](results/margin-error.png)

Poseidon's CUDA runtime collected gradient-weighted values from real Enzyme
reverse probes at every arithmetic instruction. Each of the 32 launches used
one warp. The 1,024 records at the JVP's x+1 intermediate gave an accumulated
adjoint of 1,008, exactly 32 × Σ 2dx. The script checks the corresponding
primal adjoint Σ 2(x+1) too. Calibration used x = 1.125 × 2ⁱ for i from 0
to 31. Each held-out x differs from every training x and lies within their
range. The first 31 are 1.375 × 2ⁱ and the last is 1.0625 × 2³¹. The tangent
cycles through 0.25, 0.5 and 0.75.

## Where this fits

| Prior result | Connection |
|---|---|
| ADAPT estimates how local rounding affects the output using AD. Pareto Herbie searches for fast, accurate expressions by combining rewrites and precision changes. | Both techniques underlie the search used here. [ADAPT, SC 2018](https://w3.cs.jmu.edu/lam2mo/papers/2018-Menon-SC.pdf) · [Pareto Herbie, ARITH 2021](https://fpbench.cs.washington.edu/arith21-paper.pdf) |
| Poseidon tuned a quaternion Jacobian. It profiled 1,000 inputs and checked a million more against MPFR. | Applying Poseidon to derivative code and validating it on held-out data both have direct precedent. [Poseidon, CGO 2026, §IV-B](https://ece.is/assets/pdf/poseidon-cgo26.pdf) |
| Chiluveru and colleagues fitted piecewise polynomials to sigmoid, tanh and their first derivatives. Hückelheim and colleagues measured how cancellation can give a much larger error in an AD result than in the function value. | There are prior designs for accurate value and gradient implementations, and prior measurements of the gap between their errors. [Activation approximation, 2021](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cds2.12058) · [AD pitfalls, Fig. 4](https://differentiable.xyz/papers-2023/paper_22.pdf) |
| Park and colleagues constructed floating-point networks that can realize specified outputs and AD gradients under their assumptions. | Matching values alone can leave considerable freedom in AD gradients, even at machine-representable inputs. [Park et al., ICML 2026](https://proceedings.mlr.press/v306/park26r.html) |

Our controlled comparison asks where to place the rewrite in a Triton AD
pipeline. On the same 256 inputs for x/(1+x), tuning the JVP alone and tuning
both stages gave the same worst observed value and JVP errors (2 and 3 ULP).
When we differentiated those JVPs again, the second schedule produced 14
false zeros in curvature. The first produced none. The source rewrite selected
a small-input branch that returns x, whose curvature is zero. This observation
is specific to the expression, search and input range tested. The two compiler
stages still choose their candidates independently. [TTIR pass schedules and raw results](../README.md)

## Rerun

Use the LLVM 21.1.8 release-compatible toolchain, CUDA 12.9 nvcc and the
Poseidon Explorer archive identified in `results/metadata.json`. Its plugins
and Herbie executable come from the same release. This experiment requires
an RTX 5090 with the shipped cost model.

```sh
export LLVM_BIN=/path/to/llvm-21/bin
export POSEIDON_RELEASE=/path/to/poseidon-explorer-llvm21
export NVCC=/path/to/cuda-12.9/bin/nvcc
export TRITON_PTXAS_PATH="$(dirname "$NVCC")/ptxas"
python prepare.py --opt /path/to/enzymexlamlir-opt --out run
./run.sh run
python report.py run
```

`prepare.py` keeps the two scalar TTIR functions visible at the LLVM
boundary and wraps their calls with the Poseidon marker. Triton emits them
with LLVM's internal fast calling convention. At the marker boundary both
sites need the C calling convention used by the profiling plugin. It also
sets their target CPU to match the sm_120 cost model. The four-argument
TTIR entry acquires two extra scratch pointers in Triton's lowering.

`run.sh` links the instrumented device code against Poseidon's CUDA runtime,
then uses the collected profiles for three solves. A native CUDA
kernel wrapper registers the device function for launch. The control uses
Triton's original LLVM and the same launch wrapper. Every reported value
comes from copying both CUDA outputs back to the host. The independent
algebraic references are evaluated with 100 decimal digits. The measured
costs in the explanation files are solver cost units.

The raw samples, instruction profiles, solver explanations, exact accepted
Herbie expressions and emitted LLVM bodies are in `results/`. We repeated the
10⁻¹² and 10⁻³ runs, including the optimizer and Herbie search, in separate
output directories and obtained the same held-out results.
