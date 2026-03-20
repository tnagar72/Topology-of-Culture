"""
WVS Wave 7 India Data Preparation
===================================
Converts the WVS Wave 7 India-specific CSV into the JSON format expected by layer2_eval.py.

FILE
----
F00017093-WVS_Wave_7_India_Csv_v6.0.zip  (or the extracted .csv)
Downloaded from worldvaluessurvey.org → Wave 7 → India

USAGE
-----
    python src/prepare_wvs_data.py \\
        --wvs_csv  path/to/F00017093-WVS_Wave_7_India_Csv_v6.0.csv \\
        --output   data/wvs/wvs_wave7_india_distributions.json

    # Or pass the zip directly:
    python src/prepare_wvs_data.py \\
        --wvs_csv  path/to/F00017093-WVS_Wave_7_India_Csv_v6.0.zip \\
        --output   data/wvs/wvs_wave7_india_distributions.json

OUTPUT FORMAT
-------------
    {
        "Q164": {
            "hindi":   { "distribution": {"1": 0.02, ..., "10": 0.45}, "n": 930 },
            "telugu":  { "distribution": {...}, "n": 186 },
            "marathi": { ... },
            "bengali": { ... },
            "punjabi": { ... }
        },
        ...
    }

LANGUAGE COLUMN
---------------
Q272 = "What language do you normally speak at home?"
Codes used in this dataset:
    1740 → Hindi   (n=930)
    4220 → Telugu  (n=186)
    2940 → Marathi (n=183)
     490 → Bengali (n=141)
    3540 → Punjabi (n=126)
    9000 → Other   (excluded)
      -1 → Unknown (excluded)

MISSING VALUE CODES
-------------------
WVS encodes missing / refused / don't know as negative integers (-1, -2, -4, -5).
Q254 also has code 5 = "I am not [nationality]" which is treated as missing.
All of these are excluded before computing distributions.
"""

import io
import json
import argparse
import warnings
import zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration — verified against F00017093-WVS_Wave_7_India_Csv_v6.0.csv
# ---------------------------------------------------------------------------

LANG_COL = "Q272"          # Home language column
CSV_SEP  = ";"             # The India CSV uses semicolons

LANG_MAP = {
    1740: "hindi",
    4220: "telugu",
    2940: "marathi",
     490: "bengali",
    3540: "punjabi",
}

# Questions for Layer 2 evaluation
TARGET_QUESTIONS = [
    "Q45",   # Respect for authority      (1–3)
    "Q46",   # Happiness                  (1–4)
    "Q49",   # Life satisfaction          (1–10)
    "Q57",   # Social trust               (1–2)
    "Q152",  # Country aims: 1st choice   (1–4)
    "Q153",  # Country aims: 2nd choice   (1–4)
    "Q154",  # Respondent aims: 1st       (1–4)
    "Q155",  # Respondent aims: 2nd       (1–4)
    "Q156",  # Society goals: 1st         (1–4)
    "Q157",  # Society goals: 2nd         (1–4)
    "Q164",  # Importance of God          (1–10)
    "Q209",  # Signed a petition          (1–3)
    "Q254",  # National pride             (1–4, code 5 = not citizen → excluded)
]

# Q254 has a valid-looking code 5 ("I am not [nationality]") that is NOT a real response
Q254_EXTRA_INVALID = {5}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load the India WVS CSV (plain or zipped)."""
    p = Path(path)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV found inside zip.")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8-sig"), sep=CSV_SEP, low_memory=False)
    return pd.read_csv(p, sep=CSV_SEP, low_memory=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

def compute_distribution(series: pd.Series, scale_min: int, scale_max: int,
                          extra_invalid: set = None) -> dict:
    """
    Compute normalised response distribution.
    Drops NaN, negative values (WVS missing codes), and any extra_invalid codes.
    Returns {"1": 0.02, ..., "10": 0.45}.
    """
    valid = series.dropna()
    valid = valid[valid >= 0]
    if extra_invalid:
        valid = valid[~valid.isin(extra_invalid)]
    valid = valid[valid.between(scale_min, scale_max)]

    if len(valid) == 0:
        return {}

    counts = valid.value_counts().reindex(range(scale_min, scale_max + 1), fill_value=0)
    probs  = counts / counts.sum()
    return {str(int(k)): round(float(v), 6) for k, v in probs.items()}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def prepare_distributions(df: pd.DataFrame, questions: list) -> dict:
    # Map home-language codes → language names; drop unmapped rows
    df = df.copy()
    df["_lang"] = df[LANG_COL].map(LANG_MAP)

    n_unmapped = df["_lang"].isna().sum()
    if n_unmapped > 0:
        raw = df.loc[df["_lang"].isna(), LANG_COL].value_counts().to_dict()
        print(f"  Excluded {n_unmapped} rows with unmapped language codes: {raw}")
    df = df[df["_lang"].notna()]

    print(f"\nLanguage group sizes:")
    for lang, count in df["_lang"].value_counts().sort_index().items():
        print(f"  {lang:10s}: {count}")

    result = {}
    langs  = ["hindi", "telugu", "marathi", "bengali", "punjabi"]

    for qid in questions:
        if qid not in df.columns:
            warnings.warn(f"Column {qid} not found in CSV — skipping")
            continue

        # Determine valid scale from actual data
        valid_all = df[qid].dropna()
        valid_all = valid_all[valid_all >= 0]
        if qid == "Q254":
            valid_all = valid_all[~valid_all.isin(Q254_EXTRA_INVALID)]
        if valid_all.empty:
            warnings.warn(f"No valid responses for {qid} — skipping")
            continue

        scale_min = int(valid_all.min())
        scale_max = int(valid_all.max())

        result[qid] = {}
        for lang in langs:
            subset = df[df["_lang"] == lang][qid]
            extra  = Q254_EXTRA_INVALID if qid == "Q254" else None
            dist   = compute_distribution(subset, scale_min, scale_max, extra)
            if not dist:
                warnings.warn(f"No valid responses for {qid}/{lang} — skipping")
                continue
            n_valid = int(subset[subset >= 0].count() if extra is None
                          else subset[(subset >= 0) & (~subset.isin(extra))].count())
            result[qid][lang] = {"distribution": dist, "n": n_valid}

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prepare WVS Wave 7 India distributions for layer2_eval.py")
    p.add_argument("--wvs_csv", required=True,
                   help="Path to F00017093-WVS_Wave_7_India_Csv_v6.0.csv (or .zip)")
    p.add_argument("--output",  default="data/wvs/wvs_wave7_india_distributions.json",
                   help="Output JSON path")
    p.add_argument("--questions", nargs="+", default=None,
                   help="Question IDs to process (default: all 13 Layer 2 questions)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.wvs_csv} ...")
    df = load_csv(args.wvs_csv)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    questions = args.questions or TARGET_QUESTIONS
    print(f"\nProcessing {len(questions)} questions: {questions}")

    distributions = prepare_distributions(df, questions)

    n_q = len(distributions)
    n_e = sum(len(v) for v in distributions.values())
    print(f"\nBuilt distributions: {n_q} questions, {n_e} (question × language) entries")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(distributions, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
