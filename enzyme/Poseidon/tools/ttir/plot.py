"""Plot the number of nonzero reference results rounded to zero by each kernel."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    with args.metrics.open() as stream:
        rows = {(r["case"], r["schedule"], r["output"]): r for r in csv.DictReader(stream)}
    schedules = (("base", "Untuned"), ("pre", "Primal"), ("post", "JVP"), ("both", "Primal + JVP"))
    outputs = ("primal", "jvp", "hvp")
    colors = ("#414b5a", "#2675a2", "#e89229", "#529b78")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True, layout="constrained")
    for ax, case, title in zip(axes, ("saturate", "charbonnier"), ("Saturating response", "Charbonnier loss")):
        for j, ((key, label), color) in enumerate(zip(schedules, colors)):
            offsets = np.arange(len(outputs)) + .19 * (j - 1.5)
            counts = [int(rows[case, key, o]["wrong_zero"]) for o in outputs]
            ax.bar(offsets, counts, width=.17, label=label, color=color)
            for x, n in zip(offsets, counts):
                if n:
                    ax.text(x, n + 1, str(n), ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xticks(range(len(outputs)), ["Value", "JVP", "HVP"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("False zeros (of 256 inputs)")
    axes[0].set_ylim(0, 84)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4,
               frameon=False, title="Expressions tuned with Poseidon")
    fig.savefig(args.figure, dpi=180)


if __name__ == "__main__":
    main()
