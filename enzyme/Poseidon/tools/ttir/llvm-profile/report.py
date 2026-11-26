"""Check recorded probe adjoints and plot independent held-out errors."""

import argparse
import csv
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def read_profile(path):
    payload = path.read_text()
    slot = re.search(r"^0\n(.*?)(?=^\d+\n|\Z)", payload, re.S | re.M)[1]
    return int(re.search(r"Exec = (\d+)", slot)[1]), float(re.search(r"SumGrad = ([^\n]+)", slot)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run
    training = [1.125 * 2**i for i in range(32)]
    tangents = [(i % 3 + 1) * .25 for i in range(32)]
    for fn, seed in (("response", [2*(x+1) for x in training]),
                     ("response_jvp", [2*dx for dx in tangents])):
        records, grad = read_profile(root / "profiles" / f"preprocess_{fn}.fpprofile")
        if records != 1024 or grad != 32 * sum(seed):
            raise ValueError(f"{fn}: unexpected probe count {records} or SumGrad {grad}")

    fig, ax = plt.subplots(figsize=(7.2, 3.7), layout="constrained")
    colors = ("#414b5a", "#4c9269", "#2675a2", "#b84a2b")
    markers = ("o", "D", "s", "^")
    variants = ("baseline", "1e-17", "1e-12", "1e-3")
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["variant", "output", "samples", "formula_exact_count", "max_relative_error",
                         "mean_relative_error", "poseidon_estimated_relative_error", "candidate"])
        for variant, color, marker in zip(variants, colors, markers):
            with (root / f"{variant}.csv").open() as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 32:
                raise ValueError(f"{variant}: expected 32 held-out rows")
            xs = [float(row["x"]) for row in rows]
            if set(xs) & set(training) or min(xs) < min(training) or max(xs) > max(training):
                raise ValueError("the held-out x values must be distinct and within the training range")
            for column, fn in (("value", "response"), ("jvp", "response_jvp")):
                errors = []
                with localcontext() as ctx:
                    ctx.prec = 100
                    for row in rows:
                        x = Decimal.from_float(float(row["x"]))
                        dx = Decimal.from_float(float(row["dx"]))
                        expected = 2*x + 1 if column == "value" else 2*dx
                        actual = Decimal.from_float(float(row[column]))
                        errors.append(float(abs((actual - expected) / expected)))
                estimate, selected = "", ""
                if variant != "baseline":
                    result = json.loads((root / f"explain-{variant}" / f"preprocess_{fn}.explain.json").read_text())["result"]
                    estimate, selected = result["modelled_rel_error"], ",".join(result["applied"])
                writer.writerow([variant, column, len(errors), errors.count(0), max(errors),
                                 sum(errors)/len(errors), estimate, selected])
                if column == "value":
                    label = "Untuned" if variant == "baseline" else f"Target {variant}"
                    ax.plot(xs, errors, color=color, marker=marker, markersize=3, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("Held-out input x")
    ax.set_ylabel("Relative error in squared-distance margin")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "margin-error.png", dpi=180)


if __name__ == "__main__":
    main()
