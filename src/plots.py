"""
Visualization utilities for DOSA Layer 1 results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import GROUPS


def plot_heatmap(df, output_dir):
    """Heatmap of overall accuracy (model × language group)."""
    pivot = df.groupby(["model", "group"])["correct_any"].mean().unstack()
    for g in GROUPS:
        if g not in pivot.columns:
            pivot[g] = float("nan")
    pivot = pivot[GROUPS]
    _elementwise = getattr(pivot, "map", None) or pivot.applymap
    annot = _elementwise(lambda v: f"{v:.2f}" if pd.notna(v) else "")

    plt.figure(figsize=(9, 4))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1)
    plt.title("Layer 1 Overall Accuracy by Model and Language Group\n(canonical runs)")
    plt.tight_layout()
    path = os.path.join(output_dir, "layer1_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_guess_comparison(agg, n_runs, output_dir):
    """Bar chart comparing accuracy@GUESS1 vs overall accuracy per model."""
    models  = agg.index.tolist()
    x       = range(len(models))
    g1_mean = agg["accuracy_guess1_mean"].tolist()
    g1_std  = [v if pd.notna(v) else 0 for v in agg["accuracy_guess1_std"].tolist()]
    oa_mean = agg["overall_accuracy_mean"].tolist()
    oa_std  = [v if pd.notna(v) else 0 for v in agg["overall_accuracy_std"].tolist()]

    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.2 for i in x], g1_mean, width=0.4, yerr=g1_std,
            label="accuracy@GUESS1", color="steelblue", capsize=4)
    plt.bar([i + 0.2 for i in x], oa_mean, width=0.4, yerr=oa_std,
            label="overall accuracy", color="coral", capsize=4)
    plt.xticks(list(x), models, rotation=15)
    plt.ylabel("Accuracy")
    plt.title(f"GUESS1 vs Overall Accuracy per Model (mean ± std, {n_runs} runs)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "layer1_guess_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")
