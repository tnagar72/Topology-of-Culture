"""
Human Inglehart Cultural Map
=============================
Plots where real WVS Wave 7 India language groups sit on the
Inglehart-Welzel cultural map using human ground truth data only.

USAGE
-----
    python src/human_inglehart.py \
        --wvs_human_path data/wvs/wvs_wave7_india_distributions.json \
        --output_dir     results/layer2
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Inglehart config (same as layer2_eval.py)
# ---------------------------------------------------------------------------

WVS_SAMPLE_SIZES = {
    "hindi": 930, "telugu": 186, "marathi": 183, "bengali": 141, "punjabi": 126,
}

LANGUAGES = ["hindi", "telugu", "marathi", "bengali", "punjabi"]

LANG_COLORS = {
    "hindi":   "#E63946",
    "telugu":  "#457B9D",
    "marathi": "#2D6A4F",
    "bengali": "#F4A261",
    "punjabi": "#9B2226",
}

INGLEHART_QUESTIONS = {
    "Q164": {"axis": "trad_secular", "scale_min": 1, "scale_max": 10, "flip": True,  "loading": 0.87},
    "Q254": {"axis": "trad_secular", "scale_min": 1, "scale_max": 4,  "flip": True,  "loading": 0.61},
    "Q45":  {"axis": "trad_secular", "scale_min": 1, "scale_max": 3,  "flip": True,  "loading": 0.57},
    "Q46":  {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": True,  "loading": 0.74},
    "Q49":  {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 10, "flip": False, "loading": 0.60},
    "Q57":  {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 2,  "flip": True,  "loading": 0.65},
    "Q209": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 3,  "flip": True,  "loading": 0.67},
    "Q152": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [3, 4]},
    "Q153": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [3, 4]},
    "Q154": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [2, 4]},
    "Q155": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [2, 4]},
    "Q156": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [2, 3]},
    "Q157": {"axis": "surv_selfexp", "scale_min": 1, "scale_max": 4,  "flip": None,  "loading": 0.60, "postmat_codes": [2, 3]},
}


def compute_human_positions(human: dict) -> pd.DataFrame:
    means = {}  # lang -> qid -> float

    for qid, meta in INGLEHART_QUESTIONS.items():
        if qid not in human:
            continue
        s_min, s_max = meta["scale_min"], meta["scale_max"]
        bins = np.arange(s_min, s_max + 1, dtype=float)

        for lang in LANGUAGES:
            if lang not in human[qid]:
                continue
            probs = np.array([
                float(human[qid][lang]["distribution"].get(str(int(b)), 0.0))
                for b in bins
            ])
            if probs.sum() == 0:
                continue
            probs /= probs.sum()

            if "postmat_codes" in meta:
                score = float(sum(
                    probs[int(c) - s_min] for c in meta["postmat_codes"]
                    if s_min <= int(c) <= s_max
                ))
            else:
                raw = float(np.dot(bins, probs))
                score = (s_min + s_max - raw) if meta["flip"] else raw

            means.setdefault(lang, {})[qid] = score

    # Pooled standardisation (weighted by sample size)
    global_stats = {}
    for qid in INGLEHART_QUESTIONS:
        vals = []
        for lang in LANGUAGES:
            if lang in means and qid in means[lang]:
                n = WVS_SAMPLE_SIZES.get(lang, 1)
                vals.extend([means[lang][qid]] * n)
        global_stats[qid] = (float(np.mean(vals)), float(np.std(vals)) or 1.0) if len(vals) >= 2 else (0.0, 1.0)

    rows = []
    for lang, qid_means in means.items():
        scores  = {"trad_secular": 0.0, "surv_selfexp": 0.0}
        weights = {"trad_secular": 0.0, "surv_selfexp": 0.0}
        for qid, meta in INGLEHART_QUESTIONS.items():
            if qid not in qid_means:
                continue
            mu, sd = global_stats[qid]
            z = (qid_means[qid] - mu) / (sd or 1.0)
            scores[meta["axis"]]  += meta["loading"] * z
            weights[meta["axis"]] += meta["loading"]
        rows.append({
            "language":           lang,
            "trad_secular_score": round(scores["trad_secular"] / (weights["trad_secular"] or 1.0), 4),
            "surv_selfexp_score": round(scores["surv_selfexp"] / (weights["surv_selfexp"] or 1.0), 4),
            "n":                  WVS_SAMPLE_SIZES[lang],
        })

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    for _, row in df.iterrows():
        lang = row["language"]
        ax.scatter(
            row["surv_selfexp_score"], row["trad_secular_score"],
            s=row["n"] / 4,   # bubble size ∝ sample size
            color=LANG_COLORS[lang], alpha=0.85,
            edgecolors="white", linewidths=1.5, zorder=5,
        )
        ax.annotate(
            lang.capitalize(),
            (row["surv_selfexp_score"], row["trad_secular_score"]),
            fontsize=10, ha="center", va="bottom", fontweight="bold",
            color=LANG_COLORS[lang],
        )

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("<-- Survival                          Self-Expression -->", fontsize=11)
    ax.set_ylabel("<-- Traditional                    Secular-Rational -->", fontsize=11)
    ax.set_title(
        "Inglehart Cultural Map — WVS Wave 7 India\n"
        "Real human positions by interview language (bubble size = n)",
        fontsize=12,
    )

    legend = [mpatches.Patch(color=LANG_COLORS[l], label=l.capitalize()) for l in LANGUAGES]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    plt.tight_layout()

    out = Path(output_dir) / "plots" / "human_inglehart_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return str(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wvs_human_path", default="data/wvs/wvs_wave7_india_distributions.json")
    p.add_argument("--output_dir",     default="results/layer2")
    args = p.parse_args()

    with open(args.wvs_human_path, encoding="utf-8") as f:
        human = json.load(f)

    df = compute_human_positions(human)
    print("\nInglehart positions (human WVS):")
    print(df.to_string(index=False))

    plot(df, args.output_dir)


if __name__ == "__main__":
    main()
