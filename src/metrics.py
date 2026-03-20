"""
Multi-run metrics: mean ± std, median run identification, canonical JSONL.
"""

import os
import json
from pathlib import Path
import pandas as pd

from .io import save_run_jsonl
from .scoring import parse_answer, score_run
from .plots import plot_heatmap, plot_guess_comparison


def compute_multi_run_metrics(run_dfs, output_dir, parse_fn=None, correct_fn=None):
    """
    Score each run, compute mean ± std across runs, identify the median run
    per model, build the canonical DataFrame, and save all outputs.

    Parameters
    ----------
    run_dfs : list of DataFrame
        One DataFrame per run (guess1_raw/guess2_raw or guess1_samples columns).
        If a run is already scored (has correct1 column), scoring is skipped.
    output_dir : str or Path
        Directory where CSVs, JSONL, and plots are written.
    parse_fn, correct_fn : callable, optional
        Pluggable scoring functions; default to paper implementations.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_runs        = len(run_dfs)
    parse_fn_name = getattr(parse_fn or parse_answer, "__name__", "parse_answer")

    scored_dfs    = _score_all_runs(run_dfs, parse_fn, correct_fn)
    agg, all_runs = _aggregate_runs(scored_dfs, n_runs, output_dir)
    median_runs   = _find_median_runs(agg, all_runs)
    canonical_df  = _build_canonical(scored_dfs, median_runs, output_dir)

    _save_group_summary(canonical_df, output_dir)
    _save_eval_metadata(canonical_df, parse_fn_name, n_runs, output_dir)
    plot_heatmap(canonical_df, output_dir)
    plot_guess_comparison(agg, n_runs, output_dir)

    return agg, canonical_df, median_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_all_runs(run_dfs, parse_fn, correct_fn):
    """Score any unscored runs and return the full list of scored DataFrames."""
    scored = []
    for df in run_dfs:
        if "correct1" not in df.columns:
            df = score_run(df, parse_fn=parse_fn, correct_fn=correct_fn)
        scored.append(df)
    return scored


def _aggregate_runs(scored_dfs, n_runs, output_dir=None):
    """Build per-model per-run summary and aggregate to mean ± std."""
    run_summaries = []
    for run_idx, df in enumerate(scored_dfs):
        rows = []
        for model, grp in df.groupby("model"):
            wrong1 = grp[grp["correct1"] == False]
            has_guess2 = "correct2" in grp.columns and len(wrong1) > 0
            rows.append({
                "model":            model,
                "accuracy_guess1":  grp["correct1"].mean(),
                "accuracy_guess2":  wrong1["correct2"].mean() if has_guess2 else float("nan"),
                "overall_accuracy": grp["correct_any"].mean(),
                "n":                len(grp),
            })
        s = pd.DataFrame(rows).set_index("model").round(4)
        s["run"] = run_idx + 1
        run_summaries.append(s)

    if not run_summaries:
        raise ValueError("No valid runs to process")
    all_runs = pd.concat(run_summaries)

    ddof = 1 if n_runs > 1 else 0
    agg  = all_runs.groupby("model").agg(
        accuracy_guess1_mean =("accuracy_guess1",  "mean"),
        accuracy_guess1_std  =("accuracy_guess1",  lambda x: x.std(ddof=ddof)),
        accuracy_guess2_mean =("accuracy_guess2",  "mean"),
        accuracy_guess2_std  =("accuracy_guess2",  lambda x: x.std(ddof=ddof)),
        overall_accuracy_mean=("overall_accuracy", "mean"),
        overall_accuracy_std =("overall_accuracy", lambda x: x.std(ddof=ddof)),
        n                    =("n",                "first"),
    ).round(3)

    print("\n=== Multi-run results (mean ± std) ===")
    for model in agg.index:
        r = agg.loc[model]
        print(f"  {model}: "
              f"guess1={r.accuracy_guess1_mean:.3f}±{r.accuracy_guess1_std:.3f}  "
              f"overall={r.overall_accuracy_mean:.3f}±{r.overall_accuracy_std:.3f}")

    if output_dir is not None:
        agg.to_csv(os.path.join(output_dir, "layer1_multirun_summary.csv"))

    return agg, all_runs


def _find_median_runs(agg, all_runs):
    """For each model, find the run whose overall accuracy is closest to the mean."""
    median_runs = {}
    for model in agg.index:
        mean_acc   = agg.loc[model, "overall_accuracy_mean"]
        model_rows = all_runs[all_runs.index == model].copy()
        model_rows["dist"] = (model_rows["overall_accuracy"] - mean_acc).abs()
        median_runs[model] = int(model_rows.sort_values("dist").iloc[0]["run"])

    print("\n=== Median run per model (used for fine-grained analysis) ===")
    for model, run_idx in median_runs.items():
        print(f"  {model}: run {run_idx}")

    return median_runs


def _build_canonical(scored_dfs, median_runs, output_dir):
    """Concatenate median-run rows per model and save as canonical JSONL."""
    canonical_rows = []
    for model, run_idx in median_runs.items():
        if run_idx > len(scored_dfs):
            raise ValueError(
                f"Median run {run_idx} for '{model}' exceeds number of loaded runs "
                f"({len(scored_dfs)}). Re-run after completing all runs."
            )
        model_rows = scored_dfs[run_idx - 1][scored_dfs[run_idx - 1]["model"] == model]
        canonical_rows.append(model_rows)
    canonical_df = pd.concat(canonical_rows, ignore_index=True)

    pd.DataFrame([
        {"model": m, "median_run": r} for m, r in median_runs.items()
    ]).to_csv(os.path.join(output_dir, "layer1_median_runs.csv"), index=False)

    canonical_path = os.path.join(output_dir, "layer1_results_canonical.jsonl")
    save_run_jsonl(canonical_df.to_dict("records"), canonical_path)
    print(f"\nCanonical results (median runs) → {canonical_path}")
    return canonical_df


def _save_group_summary(canonical_df, output_dir):
    """Save per-group accuracy breakdown from the canonical DataFrame."""
    group_summary = canonical_df.groupby(["model", "group"]).agg(
        accuracy_guess1 =("correct1",    "mean"),
        overall_accuracy=("correct_any", "mean"),
        n               =("artifact",    "count"),
    ).round(3)
    print("\n=== Per-group breakdown (canonical runs) ===")
    print(group_summary.to_string())
    group_summary.to_csv(os.path.join(output_dir, "layer1_group_summary.csv"))


def _save_eval_metadata(canonical_df, parse_fn_name, n_runs, output_dir):
    """Write a sidecar JSON recording the eval configuration."""
    metadata = {
        "parse_fn":  parse_fn_name,
        "n_runs":    n_runs,
        "decoders":  sorted(canonical_df["decoder"].dropna().unique().tolist())
                     if "decoder" in canonical_df.columns else [],
    }
    path = os.path.join(output_dir, "layer1_eval_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Eval metadata → {path}")
