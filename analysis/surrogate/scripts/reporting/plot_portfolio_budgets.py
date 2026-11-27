"""Plot independent-pass tuning success differences by workload and query budget."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

FAMILIES = {"gemm": "GEMM (48)", "conv2d": "Conv2D (24)", "attention": "Attention (8)",
            "softmax": "Softmax (12)", "cross_entropy": "Cross-entropy (12)",
            "gather": "Gather (20)", "residual_layernorm": "Residual LayerNorm (8)"}
BUDGETS = [1, 2, 4, 8]


def plot(source, target):
    with source.open(newline="") as stream:
        family_rows = [row for row in csv.DictReader(stream) if row["scope"] == "family"]
    lookup = {(r["mode"], r["group"], int(r["budget"])): float(r["gaussian_minus_local_portfolio_pp"])
              for r in family_rows}
    if len(family_rows) != 56 or len(lookup) != 56:
        raise ValueError("Expected a unique result for each family, mode and budget")
    limit = max(abs(v) for v in lookup.values())
    colors = LinearSegmentedColormap.from_list("advantage", ["#9d471a", "#faf9f5", "#1d5c80"])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.23), sharey=True)
    for ax, (mode, label) in zip(axes, [("graph", "(a) Graph replay"), ("individual", "(b) Individual launch")]):
        values = np.array([[lookup[mode, family, b] for b in BUDGETS] for family in FAMILIES])
        ax.imshow(values, cmap=colors, vmin=-limit, vmax=limit, aspect="auto")
        for (y, x), v in np.ndenumerate(values):
            value = f"{v:+.1f}" if abs(v) >= 0.05 else "0.0"
            ax.text(x, y, value, ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.66 * limit else "#1a2024")
        ax.set_xticks(range(len(BUDGETS)), BUDGETS)
        ax.set_yticks(range(len(FAMILIES)), FAMILIES.values())
        ax.tick_params(length=0, labelsize=8)
        ax.set_title(label, fontsize=9, loc="left", pad=4)
        ax.set_xlabel("Configurations measured", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=.20, right=.99, top=.87, bottom=.20, wspace=.13)
    fig.savefig(target, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    plot(args.summary, args.output)
