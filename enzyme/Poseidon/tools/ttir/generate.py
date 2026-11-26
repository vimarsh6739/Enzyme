"""Generate TTIR for tuning the source, its Enzyme JVP, or both."""

import argparse
from pathlib import Path
import subprocess


def run(command, source, dest):
    with dest.with_suffix(".log").open("w") as log:
        subprocess.run([*command, source, "-o", dest], check=True, stderr=log)


def tune(args, source, dest, function):
    options = ["least-error", "remark-fpcore"]
    if args.answers:
        saved = args.answers / dest.parent.name / (dest.stem + ".json")
        options.append(f"herbie-results={saved.resolve()}")
    else:
        options.extend(["run-herbie", "herbie-threads=2", f"herbie-binary={args.herbie.resolve()}",
                        f"herbie-cache={args.out.resolve() / 'cache'}"])
    options += [f"only-function={function}", "exclude-ops=math.copysign"]
    if args.platform:
        options.append(f"herbie-platform={args.platform}")
    run([args.poseidon_opt, "--poseidon-rewrite=" + " ".join(options), "--canonicalize",
         "--cse"], source, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enzyme-opt", default="enzymexlamlir-opt")
    parser.add_argument("--poseidon-opt", default="poseidon-mlir-opt")
    how = parser.add_mutually_exclusive_group(required=True)
    how.add_argument("--herbie", type=Path, help="run a fresh search with this Herbie executable")
    how.add_argument("--answers", type=Path, help="replay saved per-core candidates")
    parser.add_argument("--platform", help="Herbie platform name or cost-model .rkt")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for name in ("saturate", "charbonnier"):
        dest = args.out / name
        dest.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name(name + ".mlir")
        pre = dest / "tuned-source.mlir"
        tune(args, source, pre, name)
        for source_schedule, src, after_schedule in (("base", source, "post"), ("pre", pre, "both")):
            ad = (f"--enzyme-wrap=infn={name} outfn=jvp argTys=enzyme_dup "
                  "retTys=enzyme_dupnoneed mode=ForwardMode")
            generated = dest / f"{source_schedule}.mlir"
            run([args.enzyme_opt, ad, "--cse", "--canonicalize"], src, generated)
            tuned = dest / f"{after_schedule}.mlir"
            tune(args, generated, tuned, "jvp")
            # Differentiate the actual chosen JVP, holding the first direction
            # fixed. This tests the curvature of a numerically tuned expression.
            second = ("--enzyme-wrap=infn=jvp outfn=hvp argTys=enzyme_dup,enzyme_const "
                      "retTys=enzyme_dupnoneed mode=ForwardMode")
            for path in (generated, tuned):
                run([args.enzyme_opt, second, "--canonicalize", "--cse"], path,
                    dest / f"{path.stem}-hvp.mlir")


if __name__ == "__main__":
    main()
