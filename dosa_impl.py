"""
Layer 1 Surface Benchmark — DOSA Evaluation (replication of paper methodology)
See layer1_methodology.md for methodology rationale.

Task: Taboo-style guessing game. Given clues describing a cultural artifact,
the model must name the artifact. Two attempts are allowed.

Metrics (per DOSA paper):
  - accuracy@GUESS1  : correct on first attempt
  - accuracy@GUESS2  : correct on second attempt (given first was wrong)
  - overall_accuracy : correct on either attempt

Reference: https://arxiv.org/abs/2403.14651

Models (via Cohere Chat API):
  tiny-aya-global, tiny-aya-fire, tiny-aya-earth, tiny-aya-water

Usage:
    # Full run
    export COHERE_API_KEY=your_key_here
    python layer1_benchmark.py \
        --models tiny-aya-global tiny-aya-fire tiny-aya-earth tiny-aya-water \
        --output_dir results/layer1

    # Only build the eval set (no inference)
    python layer1_benchmark.py --build_only

    # Only recompute metrics from existing results CSV
    python layer1_benchmark.py --metrics_only --output_dir results/layer1
"""

import os
import re
import json
import random
import argparse
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from pathlib import Path
import cohere

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROUPS = ["Telugu", "Punjabi", "Bengali", "Marathi", "Hindi-Urdu"]

STATE_TO_GROUP = {
    "andhra_pradesh": "Telugu",
    "telangana":      "Telugu",
    "punjab":         "Punjabi",
    "west_bengal":    "Bengali",
    "maharashtra":    "Marathi",
    "uttar_pradesh":  "Hindi",
    "bihar":          "Hindi",
    "delhi":          "Hindi",
    "rajasthan":      "Hindi",
}

DOSA_BASE_URL = (
    "https://raw.githubusercontent.com/microsoft/DOSA/main/data/{state}/original_artifacts.csv"
)

# Prompts replicating the DOSA paper exactly
SYSTEM_PROMPT = (
    "You are an agent who is well-versed in the cultures of the world. "
    "You are playing a game of Taboo where you have to guess the name of a social artifact "
    "based on clues given to you. Social artifacts are objects that help us connect and stay "
    "associated with the culture."
)

INSTRUCTION_PROMPT_1 = (
    "Clues: {clues}\n\n"
    "Name the object based on the above clues from {state}. "
    "Just tell me the answer and nothing else."
)

INSTRUCTION_PROMPT_2 = (
    "Your first guess is not correct. "
    "While making your second guess, please stick to the format as ANSWER: your_answer_here."
)

# ---------------------------------------------------------------------------
# Step 1: Fetch and pool DOSA data
# ---------------------------------------------------------------------------

def fetch_dosa(cache_dir="dosa_cache"):
    """Download DOSA CSVs from GitHub and pool into a single DataFrame."""
    Path(cache_dir).mkdir(exist_ok=True)
    dfs = []

    for state, group in STATE_TO_GROUP.items():
        cache_path = os.path.join(cache_dir, f"{state}.csv")

        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        else:
            url = DOSA_BASE_URL.format(state=state)
            print(f"Downloading {state}...")
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"  Warning: could not fetch {state} (status {resp.status_code}), skipping")
                continue
            df = pd.read_csv(StringIO(resp.text))
            df.to_csv(cache_path, index=False)

        df["group"] = group
        df["state"] = state
        dfs.append(df)

    dosa = pd.concat(dfs, ignore_index=True)

    if "artifact" not in dosa.columns or "clues" not in dosa.columns:
        raise ValueError(f"Unexpected columns in DOSA data: {dosa.columns.tolist()}")

    dosa = dosa.dropna(subset=["artifact", "clues"])

    print("\nArtifact counts per group:")
    print(dosa["group"].value_counts().to_string())
    return dosa


# ---------------------------------------------------------------------------
# Step 2: Build eval set
# ---------------------------------------------------------------------------

def build_eval_set(dosa, output_path):
    """
    Build eval set matching the DOSA paper setup.
    Each item includes the two-turn prompt pair and the ground truth artifact.
    State name is formatted as title case for readability in prompts.
    """
    eval_set = []
    for _, row in dosa.iterrows():
        state_display = row["state"].replace("_", " ").title()
        clues = row["clues"].replace("\n", " ").strip()

        prompt1 = INSTRUCTION_PROMPT_1.format(clues=clues, state=state_display)
        prompt2 = INSTRUCTION_PROMPT_2  # sent only if guess 1 is wrong

        eval_set.append({
            "id":            f"{row['state']}_{row.name}",
            "state":         row["state"],
            "state_display": state_display,
            "group":         row["group"],
            "artifact":      row["artifact"],
            "clues":         clues,
            "prompt1":       prompt1,
            "prompt2":       prompt2,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(eval_set)} items → {output_path}")
    return eval_set


# ---------------------------------------------------------------------------
# Step 3: Model inference via Cohere Chat API
# ---------------------------------------------------------------------------

def make_cohere_client():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY environment variable not set")
    return cohere.ClientV2(api_key=api_key)


def generate_response(co, model_name, messages):
    """
    Generate a response via the Cohere Chat API.
    Temperature=0 (greedy) to match DOSA paper.
    """
    response = co.chat(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=30,
    )
    return response.message.content[0].text.strip()


def is_correct(prediction, ground_truth):
    """
    Check if prediction matches ground truth.
    Normalizes both to lowercase, strips punctuation.
    Also checks if ground truth appears as substring of prediction
    (handles cases where model adds extra words).
    """
    def normalize(s):
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        return s

    pred = normalize(prediction)
    gt   = normalize(ground_truth)

    # Exact match
    if pred == gt:
        return True

    # Ground truth contained in prediction (e.g., "ANSWER: kondapalli toys")
    if gt in pred:
        return True

    # Prediction contained in ground truth (handles abbreviated answers)
    if pred in gt and len(pred) > 3:
        return True

    return False


def parse_guess2(response):
    """Extract answer from second-guess format: 'ANSWER: ...'"""
    match = re.search(r"ANSWER[:\s]+(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Step 4: Run inference
# ---------------------------------------------------------------------------

def run_inference(eval_set, models, output_path):
    """
    Two-turn inference per the DOSA paper:
      Turn 1: system prompt + clues → guess 1
      Turn 2 (if wrong): "your first guess was incorrect" → guess 2
    Saves incrementally after each model for crash recovery.
    """
    co = make_cohere_client()

    # Load existing results to support resuming
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        completed_models = set(existing["model"].unique())
        print(f"Found existing results for: {completed_models}")
        all_results = existing.to_dict("records")
    else:
        completed_models = set()
        all_results = []

    for model_name in models:
        if model_name in completed_models:
            print(f"\nSkipping {model_name} (already complete)")
            continue

        print(f"\n=== {model_name} ===")
        model_results = []

        for i, item in enumerate(eval_set):

            # --- Turn 1 ---
            messages1 = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": item["prompt1"]},
            ]
            guess1 = generate_response(co, model_name, messages1)
            correct1 = is_correct(guess1, item["artifact"])

            # --- Turn 2 (only if guess 1 was wrong) ---
            if correct1:
                guess2    = None
                correct2  = False
            else:
                messages2 = messages1 + [
                    {"role": "assistant", "content": guess1},
                    {"role": "user",      "content": item["prompt2"]},
                ]
                raw_guess2 = generate_response(co, model_name, messages2)
                guess2     = parse_guess2(raw_guess2)
                correct2   = is_correct(guess2, item["artifact"])

            model_results.append({
                "id":           item["id"],
                "state":        item["state"],
                "group":        item["group"],
                "artifact":     item["artifact"],
                "model":        model_name,
                "guess1":       guess1,
                "correct1":     correct1,
                "guess2":       guess2,
                "correct2":     correct2,
                "correct_any":  correct1 or correct2,
            })

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(eval_set)} done")

        # Save incrementally
        all_results.extend(model_results)
        pd.DataFrame(all_results).to_csv(output_path, index=False)
        print(f"  Saved → {output_path}")

    df = pd.DataFrame(all_results)
    print(f"\nInference complete. Total rows: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Step 5: Metrics and plots
# ---------------------------------------------------------------------------

def compute_metrics(df, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Per the DOSA paper: accuracy@GUESS1, accuracy@GUESS2, overall
    summary = df.groupby("model").agg(
        accuracy_guess1 =("correct1",    "mean"),
        accuracy_guess2 =("correct2",    "mean"),
        overall_accuracy=("correct_any", "mean"),
        n               =("artifact",    "count"),
    ).round(3)

    print("\n=== Results (replicating DOSA paper metrics) ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, "layer1_summary.csv"))

    # Per-group breakdown
    group_summary = df.groupby(["model", "group"]).agg(
        accuracy_guess1 =("correct1",    "mean"),
        overall_accuracy=("correct_any", "mean"),
        n               =("artifact",    "count"),
    ).round(3)
    print("\n=== Per-group breakdown ===")
    print(group_summary.to_string())
    group_summary.to_csv(os.path.join(output_dir, "layer1_group_summary.csv"))

    # --- Plot 1: Overall accuracy heatmap (model × group) ---
    pivot = df.groupby(["model", "group"])["correct_any"].mean().unstack()
    # Ensure all groups appear even if some missing
    for g in GROUPS:
        if g not in pivot.columns:
            pivot[g] = float("nan")
    pivot = pivot[GROUPS]

    plt.figure(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title("Layer 1 Overall Accuracy by Model and Language Group\n(accuracy@GUESS1 + accuracy@GUESS2)")
    plt.tight_layout()
    path = os.path.join(output_dir, "layer1_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved {path}")

    # --- Plot 2: Guess 1 vs Guess 2 improvement per model ---
    models = df["model"].unique()
    x = range(len(models))
    g1 = [df[df["model"] == m]["correct1"].mean()    for m in models]
    g2 = [df[df["model"] == m]["correct_any"].mean() for m in models]

    plt.figure(figsize=(7, 4))
    plt.bar([i - 0.2 for i in x], g1, width=0.4, label="accuracy@GUESS1", color="steelblue")
    plt.bar([i + 0.2 for i in x], g2, width=0.4, label="overall accuracy", color="coral")
    plt.xticks(list(x), models, rotation=15)
    plt.ylabel("Accuracy")
    plt.title("GUESS1 vs Overall Accuracy per Model")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "layer1_guess_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer 1 DOSA benchmark — replication of paper methodology"
    )
    parser.add_argument(
        "--models", nargs="+", default=[],
        metavar="MODEL_ID",
        help="Cohere model IDs to evaluate, e.g. tiny-aya-global tiny-aya-fire"
    )
    parser.add_argument("--output_dir",  default="results/layer1")
    parser.add_argument("--dosa_cache",  default="dosa_cache")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--build_only",  action="store_true",
                        help="Only build the eval set, skip inference")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Recompute metrics from existing results CSV")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eval_set_path = os.path.join(args.output_dir, "layer1_eval_set.json")
    results_path  = os.path.join(args.output_dir, "layer1_results.csv")

    if args.metrics_only:
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"No results found at {results_path}")
        df = pd.read_csv(results_path)
        compute_metrics(df, args.output_dir)

    else:
        if os.path.exists(eval_set_path):
            print(f"Loading existing eval set from {eval_set_path}")
            with open(eval_set_path) as f:
                eval_set = json.load(f)
        else:
            dosa = fetch_dosa(cache_dir=args.dosa_cache)
            eval_set = build_eval_set(dosa, output_path=eval_set_path)

        if args.build_only:
            print("Build complete. Exiting (--build_only).")
        else:
            if not args.models:
                raise ValueError("Provide at least one model via --models MODEL_ID")
            df = run_inference(eval_set, args.models, results_path)
            compute_metrics(df, args.output_dir)
