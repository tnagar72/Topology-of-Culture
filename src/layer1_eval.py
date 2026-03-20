"""
layer1_eval.py — Layer 1 DOSA Evaluation Metrics
Topology of Culture Project

Computes accuracy scores on DOSA results:
  - Overall accuracy per model
  - Accuracy per model x language group
  - Accuracy per model x state
  - Bar charts and heatmaps

Usage:
    python src/layer1_eval.py \
        --results_path results/dosa/results.csv \
        --output_dir results/layer1
"""

import argparse
import os
import csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

MODEL_COLORS = {
    "tiny-aya-fire":   "#E74C3C",
    "tiny-aya-global": "#3498DB",
    "tiny-aya-earth":  "#2ECC71",
    "tiny-aya-water":  "#9B59B6",
}

MODEL_LABELS = {
    "tiny-aya-fire":   "Fire",
    "tiny-aya-global": "Global",
    "tiny-aya-earth":  "Earth",
    "tiny-aya-water":  "Water",
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_results(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["correct_any"] = row["correct_any"].strip().lower() == "true"
            rows.append(row)
    return rows


def accuracy(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["correct_any"]) / len(rows)


# ── compute metrics ───────────────────────────────────────────────────────────

def compute_metrics(rows):
    models  = sorted(set(r["model"] for r in rows))
    groups  = sorted(set(r["group"] for r in rows))
    states  = sorted(set(r["state"] for r in rows))

    # overall per model
    overall = {}
    for m in models:
        subset = [r for r in rows if r["model"] == m]
        overall[m] = accuracy(subset)

    # model x group
    by_group = defaultdict(dict)
    for m in models:
        for g in groups:
            subset = [r for r in rows if r["model"] == m and r["group"] == g]
            by_group[m][g] = accuracy(subset)

    # model x state
    by_state = defaultdict(dict)
    for m in models:
        for s in states:
            subset = [r for r in rows if r["model"] == m and r["state"] == s]
            by_state[m][s] = accuracy(subset)

    return overall, by_group, by_state, models, groups, states


# ── save CSVs ─────────────────────────────────────────────────────────────────

def save_overall_csv(overall, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy"])
        for model, acc in sorted(overall.items(), key=lambda x: -x[1]):
            w.writerow([model, round(acc, 4)])
    print(f"  Saved {path}")


def save_group_csv(by_group, groups, path):
    models = sorted(by_group.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + groups)
        for m in models:
            w.writerow([m] + [round(by_group[m].get(g, 0), 4) for g in groups])
    print(f"  Saved {path}")


def save_state_csv(by_state, states, path):
    models = sorted(by_state.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model"] + states)
        for m in models:
            w.writerow([m] + [round(by_state[m].get(s, 0), 4) for s in states])
    print(f"  Saved {path}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_overall(overall, out_path):
    models  = sorted(overall, key=lambda m: -overall[m])
    accs    = [overall[m] for m in models]
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    colors  = [MODEL_COLORS.get(m, "#999") for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, min(1.0, max(accs) * 1.3))
    ax.set_ylabel("Accuracy (correct_any)", fontsize=11)
    ax.set_title("Layer 1 DOSA — Overall Accuracy by Model", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_group_bars(by_group, groups, models, out_path):
    x       = np.arange(len(groups))
    n       = len(models)
    width   = 0.18
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(sorted(models)):
        vals = [by_group[m].get(g, 0) for g in groups]
        ax.bar(x + offsets[i], vals, width,
               label=MODEL_LABELS.get(m, m),
               color=MODEL_COLORS.get(m, "#999"),
               edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Accuracy (correct_any)", fontsize=11)
    ax.set_title("Layer 1 DOSA — Accuracy by Model x Language Group", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_heatmap(by_group, groups, models, out_path):
    sorted_models = sorted(models)
    data = np.array([[by_group[m].get(g, 0) for g in groups] for m in sorted_models])
    labels_y = [MODEL_LABELS.get(m, m) for m in sorted_models]

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.5)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(labels_y, fontsize=10)

    for i in range(len(sorted_models)):
        for j in range(len(groups)):
            ax.text(j, i, f"{data[i, j]:.1%}",
                    ha="center", va="center", fontsize=9,
                    color="black" if data[i, j] > 0.05 else "gray")

    plt.colorbar(im, ax=ax, format="{x:.0%}")
    ax.set_title("Layer 1 DOSA — Accuracy Heatmap (Model x Language Group)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


def plot_state_heatmap(by_state, states, models, out_path):
    sorted_models = sorted(models)
    data = np.array([[by_state[m].get(s, 0) for s in states] for m in sorted_models])
    labels_y = [MODEL_LABELS.get(m, m) for m in sorted_models]
    labels_x = [s.replace("_", " ").title() for s in states]

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.5)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(labels_x, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(sorted_models)))
    ax.set_yticklabels(labels_y, fontsize=10)

    for i in range(len(sorted_models)):
        for j in range(len(states)):
            ax.text(j, i, f"{data[i, j]:.1%}",
                    ha="center", va="center", fontsize=8,
                    color="black" if data[i, j] > 0.05 else "gray")

    plt.colorbar(im, ax=ax, format="{x:.0%}")
    ax.set_title("Layer 1 DOSA — Accuracy Heatmap (Model x State)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path}")


# ── print summary ─────────────────────────────────────────────────────────────

def print_summary(overall, by_group, groups, models):
    print("\n" + "=" * 60)
    print("  LAYER 1 DOSA EVALUATION RESULTS")
    print("=" * 60)

    print("\n>> Overall accuracy by model (higher = better)")
    for m, acc in sorted(overall.items(), key=lambda x: -x[1]):
        print(f"   {MODEL_LABELS.get(m, m):10s}  {acc:.1%}")

    print("\n>> Accuracy by model x language group")
    header = f"{'Model':12s}" + "".join(f"{g:14s}" for g in groups)
    print("   " + header)
    for m in sorted(models):
        row = f"{MODEL_LABELS.get(m, m):12s}"
        row += "".join(f"{by_group[m].get(g, 0):.1%}{'':8s}" for g in groups)
        print("   " + row)
    print("=" * 60)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="results/dosa/results.csv")
    parser.add_argument("--output_dir",   default="results/layer1")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading DOSA results from {args.results_path}")
    rows = load_results(args.results_path)
    print(f"Loaded {len(rows)} rows\n")

    overall, by_group, by_state, models, groups, states = compute_metrics(rows)

    print("Saving CSVs ...")
    save_overall_csv(overall, os.path.join(args.output_dir, "accuracy_overall.csv"))
    save_group_csv(by_group, groups, os.path.join(args.output_dir, "accuracy_by_group.csv"))
    save_state_csv(by_state, states, os.path.join(args.output_dir, "accuracy_by_state.csv"))

    print_summary(overall, by_group, groups, models)

    print("\nGenerating plots ...")
    plot_overall(overall,   os.path.join(plots_dir, "accuracy_overall.png"))
    plot_group_bars(by_group, groups, models, os.path.join(plots_dir, "accuracy_by_group.png"))
    plot_heatmap(by_group, groups, models,    os.path.join(plots_dir, "accuracy_heatmap_group.png"))
    plot_state_heatmap(by_state, states, models, os.path.join(plots_dir, "accuracy_heatmap_state.png"))

    print("\nLayer 1 evaluation complete.")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
