"""
Layer 2 Evaluation Module — WVS Deep-Value Calibration
=======================================================

Evaluates how well Tiny Aya (Fire / Global / Earth) captures real human value
distributions from WVS Wave 7 India, disaggregated by interview language.

KEY METRICS
-----------
1. JSD per question × language × model        (lower = better calibrated)
2. Differentiation: pairwise JSD between language distributions
   (model vs. human) — does the model collapse all languages into one?
3. Fire vs. Global vs. Earth comparison on JSD scores
4. Inglehart cultural map positions (Traditional/Secular-Rational ×
   Survival/Self-Expression) using 10 standard WVS questions + factor loadings

INPUTS
------
Model inference JSONs (Ananya's pipeline), one record per inference call.
Each file may contain a single record or a list of records:

    {
        "model":       "fire",     # fire | global | earth
        "language":    "telugu",   # hindi | telugu | marathi | bengali | punjabi
        "question_id": "Q164",
        "responses":   [8, 9, 7, ...]   # N stochastic samples from the model
    }

WVS human distributions (Tanay's preprocessed output) — see data/wvs/:

    {
        "Q164": {
            "hindi":   { "distribution": {"1": 0.02, ..., "10": 0.45}, "n": 930 },
            "telugu":  { "distribution": {...}, "n": 186 },
            ...
        },
        ...
    }

OUTPUTS (in results/layer2/)
----------------------------
    jsd_scores.csv           — JSD per question × language × model (+ bootstrap CIs)
    jsd_summary.csv          — Mean JSD per model × language × topic cluster
    differentiation.csv      — Pairwise language JSD (model vs. human) per question
    inglehart_positions.csv  — Inglehart axis scores per source × language
    plots/jsd_heatmap.png    — Heatmap of mean JSD per model × language × cluster
    plots/model_comparison.png — Fire / Global / Earth bar chart by language
    plots/differentiation.png  — Model vs. human pairwise JSD scatter
    plots/inglehart_map.png    — Inglehart cultural map positioning

USAGE
-----
    python src/layer2_eval.py \\
        --inference_dir  results/layer2/inference \\
        --wvs_human_path data/wvs/wvs_wave7_india_distributions.json \\
        --output_dir     results/layer2 \\
        --n_bootstrap    1000

    # Skip JSD recomputation, only redo Inglehart positions + plots
    python src/layer2_eval.py --inglehart_only \\
        --inference_dir  results/layer2/inference \\
        --wvs_human_path data/wvs/wvs_wave7_india_distributions.json \\
        --output_dir     results/layer2

REFERENCES
----------
- Tao et al. (PNAS Nexus, 2024)   — WVS-based cultural bias in LLMs (Inglehart method)
- Wang et al. (ACL, 2024)         — WVS as LLM evaluation instrument
- Inglehart & Welzel (2005)       — Cultural Map methodology and factor loadings
- WVS Wave 7 (2017–2022)          — worldvaluessurvey.org
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import entropy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LANGUAGES = ["hindi", "telugu", "marathi", "bengali", "punjabi"]
MODELS    = ["fire", "global", "earth"]

# WVS Wave 7 India — sample sizes by interview language (from paper)
WVS_SAMPLE_SIZES = {
    "hindi":   930,
    "telugu":  186,
    "marathi": 183,
    "bengali": 141,
    "punjabi": 126,
}

# ---------------------------------------------------------------------------
# WVS question metadata
# ---------------------------------------------------------------------------

# Topic clusters as defined in the research design
TOPIC_CLUSTERS = {
    "family_authority":  [f"Q{i}" for i in list(range(1, 7)) + list(range(36, 46))],
    "child_rearing":     [f"Q{i}" for i in range(7, 18)],
    "gender_roles":      [f"Q{i}" for i in range(29, 36)],
    "religion_morality": [f"Q{i}" for i in range(163, 199)],
}

QUESTION_TO_CLUSTER = {}
for _cluster, _qids in TOPIC_CLUSTERS.items():
    for _qid in _qids:
        QUESTION_TO_CLUSTER[_qid] = _cluster

# ---------------------------------------------------------------------------
# Inglehart-Welzel cultural map — 13 WVS Wave 7 questions
# ---------------------------------------------------------------------------
#
# Axis conventions (after recoding):
#   trad_secular : positive → more secular-rational (away from traditional)
#   surv_selfexp : positive → more self-expression  (away from survival)
#
# "flip": True  → raw response runs in the traditional/survival direction,
#                 so we apply (scale_min + scale_max − response) before
#                 standardising so that HIGH recoded value = secular/selfexp.
#
# Factor loadings from Inglehart & Welzel (2005), as used in
# Tao et al. (PNAS Nexus, 2024).
#
# WVS Wave 7 question IDs used here:
#
#   Traditional / Secular-Rational axis:
#     Q164 — Importance of God            (1–10; 10=very important=traditional)
#     Q254 — Respect for human rights     (1–4;  1=great deal=secular, 4=none=traditional)
#     Q45  — Respect for authority        (1–3;  1=good thing=traditional)
#
#   Survival / Self-Expression axis:
#     Q46  — Happiness                    (1–4;  1=very happy=self-expression)
#     Q49  — Life satisfaction            (1–10; 10=satisfied=self-expression)
#     Q57  — Social trust                 (1–2;  1=trust=self-expression)
#     Q209 — Signed a petition            (1–3;  1=have done=self-expression)
#     Q152 — National aims: 1st choice    (1–4;  3,4=post-materialist)
#     Q153 — National aims: 2nd choice    (1–4;  3,4=post-materialist)
#     Q154 — Respondent aims: 1st choice  (1–4;  2,4=post-materialist)
#     Q155 — Respondent aims: 2nd choice  (1–4;  2,4=post-materialist)
#     Q156 — Society aims: 1st choice     (1–4;  2,3=post-materialist)
#     Q157 — Society aims: 2nd choice     (1–4;  2,3=post-materialist)
#
# NOTE: Q152–Q157 use "postmat_codes" instead of "flip".
#   Each is a 4-option question. The post-materialist options differ per question
#   (they are not linearly ordered), so we recode each response to 0/1
#   (1 = chose a post-materialist option) and use the mean proportion as the score.

INGLEHART_QUESTIONS = {
    # --- Traditional / Secular-Rational axis ---
    "Q164": {
        "axis": "trad_secular", "label": "Importance of God",
        "scale_min": 1, "scale_max": 10,
        "flip": True,    # 10 = very important = traditional → flip so HIGH = secular
        "loading": 0.87,
    },
    "Q254": {
        "axis": "trad_secular", "label": "Respect for individual human rights",
        "scale_min": 1, "scale_max": 4,
        "flip": True,    # 1 = great deal of respect = secular → flip so HIGH = secular
        "loading": 0.61,
    },
    "Q45": {
        "axis": "trad_secular", "label": "Respect for authority",
        "scale_min": 1, "scale_max": 3,
        "flip": True,    # 1 = good thing = traditional → flip so HIGH = secular
        "loading": 0.57,
    },
    # --- Survival / Self-Expression axis ---
    "Q46": {
        "axis": "surv_selfexp", "label": "Happiness",
        "scale_min": 1, "scale_max": 4,
        "flip": True,    # 1 = very happy = self-expression → flip so HIGH = selfexp
        "loading": 0.74,
    },
    "Q49": {
        "axis": "surv_selfexp", "label": "Life satisfaction",
        "scale_min": 1, "scale_max": 10,
        "flip": False,   # 10 = satisfied = self-expression → no flip needed
        "loading": 0.60,
    },
    "Q57": {
        "axis": "surv_selfexp", "label": "Social trust",
        "scale_min": 1, "scale_max": 2,
        "flip": True,    # 1 = trust = self-expression → flip so HIGH = selfexp
        "loading": 0.65,
    },
    "Q209": {
        "axis": "surv_selfexp", "label": "Signed a petition",
        "scale_min": 1, "scale_max": 3,
        "flip": True,    # 1 = have done = self-expression → flip so HIGH = selfexp
        "loading": 0.67,
    },
    # Post-materialist questions (WVS Wave 7, Q152–Q157)
    # Each has 4 options (scale 1–4). Post-materialist options are NOT linearly
    # ordered, so we use "postmat_codes" to recode each response to 0/1.
    # The Inglehart score for each question = proportion of post-materialist responses.
    #
    # Q152/Q153 — National aims (1st and 2nd choice):
    #   1=Economic growth, 2=Strong defence, 3=More say, 4=Beautiful cities
    #   Post-materialist = options 3 or 4
    #
    # Q154/Q155 — Respondent aims (1st and 2nd choice):
    #   1=Maintaining order, 2=More say in govt, 3=Fighting prices, 4=Freedom of speech
    #   Post-materialist = options 2 or 4
    #
    # Q156/Q157 — Society aims (1st and 2nd choice):
    #   1=Stable economy, 2=More humane society, 3=Ideas over money, 4=Fight crime
    #   Post-materialist = options 2 or 3
    "Q152": {
        "axis": "surv_selfexp", "label": "National aims: 1st choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [3, 4],
        "loading": 0.60,
    },
    "Q153": {
        "axis": "surv_selfexp", "label": "National aims: 2nd choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [3, 4],
        "loading": 0.60,
    },
    "Q154": {
        "axis": "surv_selfexp", "label": "Respondent aims: 1st choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [2, 4],
        "loading": 0.60,
    },
    "Q155": {
        "axis": "surv_selfexp", "label": "Respondent aims: 2nd choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [2, 4],
        "loading": 0.60,
    },
    "Q156": {
        "axis": "surv_selfexp", "label": "Society aims: 1st choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [2, 3],
        "loading": 0.60,
    },
    "Q157": {
        "axis": "surv_selfexp", "label": "Society aims: 2nd choice",
        "scale_min": 1, "scale_max": 4,
        "flip": None,
        "postmat_codes": [2, 3],
        "loading": 0.60,
    },
}

INGLEHART_Q_IDS = list(INGLEHART_QUESTIONS.keys())

# Plotting aesthetics
MODEL_COLORS = {"fire": "#E63946", "global": "#457B9D", "earth": "#2D6A4F"}
LANG_MARKERS  = {
    "hindi": "o", "telugu": "s", "marathi": "^", "bengali": "D", "punjabi": "P",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_human_distributions(path: str) -> dict:
    """
    Load pre-processed WVS Wave 7 human distributions.

    Expected JSON schema:
        {
            "Q164": {
                "hindi":   { "distribution": {"1": 0.02, ..., "10": 0.45}, "n": 930 },
                "telugu":  { "distribution": {...}, "n": 186 },
                ...
            },
            ...
        }

    Returns dict: question_id → language → {"distribution": dict[str,float], "n": int}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for qid, lang_data in data.items():
        for lang, info in lang_data.items():
            if "distribution" not in info:
                raise ValueError(f"Missing 'distribution' key for {qid}/{lang}")
            if "n" not in info:
                warnings.warn(
                    f"Missing 'n' for {qid}/{lang}; using WVS_SAMPLE_SIZES default"
                )
                info["n"] = WVS_SAMPLE_SIZES.get(lang, 0)

    return data


def load_model_outputs(inference_dir: str) -> list:
    """
    Load all model inference JSON files from a directory (searched recursively).

    Each file must contain a dict or list of dicts with keys:
        model, language, question_id, responses

    Returns list of dicts.
    """
    records = []
    json_files = list(Path(inference_dir).glob("**/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {inference_dir}")

    for fpath in json_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)

    required = {"model", "language", "question_id", "responses"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} in {inference_dir} missing fields: {missing}")
        if rec["model"] not in MODELS:
            warnings.warn(f"Unknown model '{rec['model']}' in record {i}")
        if rec["language"] not in LANGUAGES:
            warnings.warn(f"Unknown language '{rec['language']}' in record {i}")

    print(f"Loaded {len(records)} inference records from {inference_dir}")
    return records


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------

def compute_distribution(responses: list, scale_min: int, scale_max: int) -> np.ndarray:
    """
    Convert a list of integer responses to a probability distribution
    over [scale_min, scale_max].  Returns array of length
    (scale_max − scale_min + 1) summing to 1.0.
    """
    bins   = list(range(scale_min, scale_max + 1))
    counts = Counter(int(r) for r in responses if scale_min <= int(r) <= scale_max)
    dist   = np.array([counts.get(b, 0) for b in bins], dtype=float)
    if dist.sum() == 0:
        warnings.warn("All responses out of scale range; returning uniform distribution")
        dist = np.ones(len(bins))
    return dist / dist.sum()


def human_dist_to_array(dist_dict: dict, scale_min: int, scale_max: int) -> np.ndarray:
    """
    Convert human distribution dict {"1": 0.02, ..., "10": 0.45}
    into a numpy probability array over [scale_min, scale_max].
    """
    bins = list(range(scale_min, scale_max + 1))
    dist = np.array([float(dist_dict.get(str(b), 0.0)) for b in bins])
    total = dist.sum()
    if total == 0:
        warnings.warn("Human distribution sums to 0; returning uniform")
        return np.ones(len(bins)) / len(bins)
    return dist / total


def get_scale(question_id: str, human_distributions: dict) -> tuple:
    """
    Return (scale_min, scale_max) for a question.
    Priority: INGLEHART_QUESTIONS metadata → infer from human dist keys → default 1–10.
    """
    if question_id in INGLEHART_QUESTIONS:
        q = INGLEHART_QUESTIONS[question_id]
        return q["scale_min"], q["scale_max"]
    if question_id in human_distributions:
        all_keys = set()
        for lang_info in human_distributions[question_id].values():
            all_keys.update(int(k) for k in lang_info["distribution"])
        if all_keys:
            return min(all_keys), max(all_keys)
    warnings.warn(f"Cannot determine scale for {question_id}; defaulting to 1–10")
    return 1, 10


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence
# ---------------------------------------------------------------------------

def compute_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence (log base 2, bounded [0, 1]).

    JSD(P ‖ Q) = 0.5 · KL(P ‖ M) + 0.5 · KL(Q ‖ M),  M = (P+Q)/2

    Lower JSD → more similar distributions → better-calibrated model.
    """
    eps = 1e-12
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * float(entropy(p, m, base=2)) + 0.5 * float(entropy(q, m, base=2))


def bootstrap_jsd_ci(
    model_responses: list,
    human_dist:      np.ndarray,
    scale_min:       int,
    scale_max:       int,
    n_bootstrap:     int = 1000,
    alpha:           float = 0.05,
    rng:             np.random.Generator = None,
) -> tuple:
    """
    Bootstrapped confidence interval for JSD(model ‖ human).

    For each replicate: resample model_responses with replacement,
    recompute JSD against the fixed human distribution.
    Returns (ci_lower, ci_upper) at (alpha/2, 1−alpha/2) quantiles.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.array([int(r) for r in model_responses])
    samples = []
    for _ in range(n_bootstrap):
        boot = rng.choice(arr, size=len(arr), replace=True)
        dist = compute_distribution(boot.tolist(), scale_min, scale_max)
        samples.append(compute_jsd(dist, human_dist))
    samples = np.array(samples)
    return float(np.percentile(samples, 100 * alpha / 2)), \
           float(np.percentile(samples, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Core metric 1 — JSD per question × language × model
# ---------------------------------------------------------------------------

def compute_all_jsd(
    model_outputs:       list,
    human_distributions: dict,
    n_bootstrap:         int   = 1000,
    alpha:               float = 0.05,
    seed:                int   = 42,
) -> pd.DataFrame:
    """
    Compute JSD scores (+ bootstrapped CIs) for every
    question × language × model combination in model_outputs.

    Returns DataFrame with columns:
        model, language, question_id, topic_cluster,
        n_samples, jsd, ci_lower, ci_upper
    """
    rng = np.random.default_rng(seed)

    # Aggregate responses across multiple records for the same (model, lang, qid)
    grouped: dict = defaultdict(list)
    for rec in model_outputs:
        key = (rec["model"], rec["language"], rec["question_id"])
        grouped[key].extend(rec["responses"])

    total = len(grouped)
    rows  = []
    for i, ((model, language, qid), responses) in enumerate(grouped.items()):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  JSD {i+1}/{total}  ({model}, {language}, {qid})")

        if qid not in human_distributions:
            warnings.warn(f"No human distribution for {qid}; skipping")
            continue
        if language not in human_distributions[qid]:
            warnings.warn(f"No human distribution for {qid}/{language}; skipping")
            continue

        scale_min, scale_max = get_scale(qid, human_distributions)
        human_dist = human_dist_to_array(
            human_distributions[qid][language]["distribution"], scale_min, scale_max
        )
        model_dist = compute_distribution(responses, scale_min, scale_max)
        jsd_val    = compute_jsd(model_dist, human_dist)
        ci_lo, ci_hi = bootstrap_jsd_ci(
            responses, human_dist, scale_min, scale_max,
            n_bootstrap=n_bootstrap, alpha=alpha, rng=rng,
        )
        rows.append({
            "model":         model,
            "language":      language,
            "question_id":   qid,
            "topic_cluster": QUESTION_TO_CLUSTER.get(qid, "inglehart"),
            "n_samples":     len(responses),
            "jsd":           round(jsd_val, 4),
            "ci_lower":      round(ci_lo,   4),
            "ci_upper":      round(ci_hi,   4),
        })

    df = pd.DataFrame(rows)
    print(f"\nJSD done: {len(df)} (question × language × model) combinations")
    return df


def compute_jsd_summary(jsd_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean JSD per model × language × topic_cluster, plus an OVERALL row."""
    by_cluster = (
        jsd_df
        .groupby(["model", "language", "topic_cluster"])["jsd"]
        .agg(mean_jsd="mean", median_jsd="median", n_questions="count")
        .round(4)
        .reset_index()
    )
    overall = (
        jsd_df
        .groupby(["model", "language"])["jsd"]
        .agg(mean_jsd="mean", median_jsd="median", n_questions="count")
        .round(4)
        .reset_index()
    )
    overall["topic_cluster"] = "OVERALL"
    return pd.concat([by_cluster, overall], ignore_index=True)


# ---------------------------------------------------------------------------
# Core metric 2 — Differentiation (pairwise language JSD)
# ---------------------------------------------------------------------------

def compute_differentiation(
    model_outputs:       list,
    human_distributions: dict,
) -> pd.DataFrame:
    """
    For each question × model, compute pairwise JSD between every pair of
    language distributions — both for the model and for human WVS respondents.

    A model with high differentiation (pairwise JSD close to human pairwise JSD)
    treats languages as culturally distinct rather than collapsing them.

    Returns DataFrame with columns:
        model, question_id, topic_cluster, lang_pair,
        model_pairwise_jsd, human_pairwise_jsd
    """
    # Build model distribution cache
    grouped: dict = defaultdict(list)
    for rec in model_outputs:
        key = (rec["model"], rec["language"], rec["question_id"])
        grouped[key].extend(rec["responses"])

    model_dists: dict = {}
    for (model, language, qid), responses in grouped.items():
        if qid not in human_distributions:
            continue
        s_min, s_max = get_scale(qid, human_distributions)
        model_dists[(model, language, qid)] = compute_distribution(
            responses, s_min, s_max
        )

    rows = []
    all_qids = {qid for (_, _, qid) in model_dists}

    for qid in all_qids:
        if qid not in human_distributions:
            continue
        s_min, s_max = get_scale(qid, human_distributions)

        # Human dists for all languages
        human_lang_dists = {
            lang: human_dist_to_array(
                human_distributions[qid][lang]["distribution"], s_min, s_max
            )
            for lang in LANGUAGES
            if lang in human_distributions.get(qid, {})
        }

        for model in MODELS:
            model_lang_dists = {
                lang: model_dists[(model, lang, qid)]
                for lang in LANGUAGES
                if (model, lang, qid) in model_dists
            }
            available = sorted(set(model_lang_dists) & set(human_lang_dists))
            for lang_a, lang_b in combinations(available, 2):
                rows.append({
                    "model":              model,
                    "question_id":        qid,
                    "topic_cluster":      QUESTION_TO_CLUSTER.get(qid, "inglehart"),
                    "lang_pair":          f"{lang_a}_{lang_b}",
                    "model_pairwise_jsd": round(
                        compute_jsd(model_lang_dists[lang_a], model_lang_dists[lang_b]), 4
                    ),
                    "human_pairwise_jsd": round(
                        compute_jsd(human_lang_dists[lang_a], human_lang_dists[lang_b]), 4
                    ),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core metric 3 — Inglehart cultural map positions
# ---------------------------------------------------------------------------

def _recode(response: float, scale_min: int, scale_max: int, flip: bool) -> float:
    """Optionally flip a response so that HIGH value → secular-rational / self-expression."""
    return (scale_min + scale_max - response) if flip else response


def compute_inglehart_positions(
    model_outputs:       list,
    human_distributions: dict,
) -> pd.DataFrame:
    """
    Compute Inglehart cultural map positions for each model × language
    combination AND for human WVS respondents per language group.

    Method (following Tao et al., PNAS Nexus 2024):
    1. For each of the 10 Inglehart questions, compute the mean response
       after recoding so HIGH = secular-rational / self-expression.
    2. Standardise using pooled within-India statistics (human data,
       weighted by WVS sample sizes per language group).
    3. Axis score = Σ (loading_i × z_i), normalised by Σ loading_i.

    Returns DataFrame with columns:
        source, language,
        trad_secular_score, surv_selfexp_score, n_inglehart_questions
    """
    # ── Step 1: mean recoded responses per (source, language, question) ────
    means: dict = defaultdict(dict)   # (source, language) → qid → float

    # Model means
    grouped: dict = defaultdict(list)
    for rec in model_outputs:
        if rec["question_id"] not in INGLEHART_Q_IDS:
            continue
        grouped[(rec["model"], rec["language"], rec["question_id"])].extend(
            rec["responses"]
        )

    for (model, lang, qid), responses in grouped.items():
        meta  = INGLEHART_QUESTIONS[qid]
        valid = [r for r in responses if meta["scale_min"] <= int(r) <= meta["scale_max"]]
        if not valid:
            continue
        if "postmat_codes" in meta:
            # Recode each response to 1 (post-materialist) or 0 (materialist).
            # Score = proportion of post-materialist responses.
            means[(model, lang)][qid] = float(np.mean([
                1.0 if int(r) in meta["postmat_codes"] else 0.0
                for r in valid
            ]))
        else:
            means[(model, lang)][qid] = float(np.mean([
                _recode(r, meta["scale_min"], meta["scale_max"], meta["flip"])
                for r in valid
            ]))

    # Human means (E[X] from distribution)
    for qid, meta in INGLEHART_QUESTIONS.items():
        if qid not in human_distributions:
            continue
        for lang in LANGUAGES:
            if lang not in human_distributions[qid]:
                continue
            s_min, s_max = meta["scale_min"], meta["scale_max"]
            bins  = np.arange(s_min, s_max + 1, dtype=float)
            probs = np.array([
                float(human_distributions[qid][lang]["distribution"].get(str(int(b)), 0.0))
                for b in bins
            ])
            if probs.sum() == 0:
                continue
            probs /= probs.sum()
            if "postmat_codes" in meta:
                # Score = probability of choosing a post-materialist option.
                means[("human", lang)][qid] = float(sum(
                    probs[int(c) - s_min] for c in meta["postmat_codes"]
                    if s_min <= int(c) <= s_max
                ))
            else:
                raw_mean = float(np.dot(bins, probs))
                means[("human", lang)][qid] = _recode(raw_mean, s_min, s_max, meta["flip"])

    # ── Step 2: pooled standardisation parameters (within-India, human-weighted) ──
    global_stats: dict = {}
    for qid in INGLEHART_Q_IDS:
        vals = []
        for lang in LANGUAGES:
            if ("human", lang) in means and qid in means[("human", lang)]:
                n = WVS_SAMPLE_SIZES.get(lang, 1)
                vals.extend([means[("human", lang)][qid]] * n)
        if len(vals) < 2:
            global_stats[qid] = (0.0, 1.0)
        else:
            global_stats[qid] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)

    # ── Step 3: compute axis scores ────────────────────────────────────────
    rows = []
    for (source, lang), qid_means in means.items():
        scores  = {"trad_secular": 0.0, "surv_selfexp": 0.0}
        weights = {"trad_secular": 0.0, "surv_selfexp": 0.0}

        for qid, meta in INGLEHART_QUESTIONS.items():
            if qid not in qid_means:
                continue
            mu, sd  = global_stats.get(qid, (0.0, 1.0))
            z       = (qid_means[qid] - mu) / (sd or 1.0)
            axis    = meta["axis"]
            loading = meta["loading"]
            scores[axis]  += loading * z
            weights[axis] += loading

        ts = scores["trad_secular"] / (weights["trad_secular"] or 1.0)
        ss = scores["surv_selfexp"] / (weights["surv_selfexp"] or 1.0)

        rows.append({
            "source":               source,
            "language":             lang,
            "trad_secular_score":   round(ts, 4),
            "surv_selfexp_score":   round(ss, 4),
            "n_inglehart_questions": sum(1 for q in INGLEHART_Q_IDS if q in qid_means),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_highlights(jsd_df: pd.DataFrame, inglehart_df: pd.DataFrame):
    sep = "=" * 62
    print(f"\n{sep}")
    print("  LAYER 2 WVS EVALUATION — KEY RESULTS")
    print(sep)

    print("\n>> Mean JSD by model (lower = better calibrated to human values)")
    print(jsd_df.groupby("model")["jsd"].mean().sort_values().round(4).to_string())

    print("\n>> Mean JSD by language")
    print(jsd_df.groupby("language")["jsd"].mean().sort_values().round(4).to_string())

    print("\n>> Mean JSD -- model x language")
    pivot = jsd_df.groupby(["model", "language"])["jsd"].mean().unstack().round(4)
    print(pivot.to_string())

    print("\n>> Mean JSD -- model x topic cluster")
    pivot2 = jsd_df.groupby(["model", "topic_cluster"])["jsd"].mean().unstack().round(4)
    print(pivot2.to_string())

    if not inglehart_df.empty:
        print("\n>> Inglehart positions  (Trad/Secular x Surv/SelfExp)")
        cols = ["source", "language", "trad_secular_score", "surv_selfexp_score"]
        print(inglehart_df[cols].sort_values(["source", "language"]).to_string(index=False))

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_jsd_heatmap(jsd_df: pd.DataFrame, output_dir: str):
    """Mean JSD heatmap: one panel per topic cluster, rows=model, cols=language."""
    clusters   = sorted(jsd_df["topic_cluster"].unique())
    n          = len(clusters)
    fig, axes  = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters):
        sub   = jsd_df[jsd_df["topic_cluster"] == cluster]
        pivot = sub.groupby(["model", "language"])["jsd"].mean().unstack(fill_value=np.nan)
        for lang in LANGUAGES:
            if lang not in pivot.columns:
                pivot[lang] = np.nan
        pivot = pivot[LANGUAGES]
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
            vmin=0, vmax=1, ax=ax, linewidths=0.5,
        )
        ax.set_title(cluster.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("")
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.suptitle(
        "Mean JSD per Model × Language × Topic Cluster\n(lower = better calibrated to human WVS distributions)",
        fontsize=11,
    )
    plt.tight_layout()
    _save(fig, output_dir, "jsd_heatmap.png")


def plot_model_comparison(jsd_df: pd.DataFrame, output_dir: str):
    """Grouped bar chart: Fire / Global / Earth mean JSD per language."""
    mean_jsd = jsd_df.groupby(["model", "language"])["jsd"].mean().reset_index()
    x     = np.arange(len(LANGUAGES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, model in enumerate(MODELS):
        sub  = mean_jsd[mean_jsd["model"] == model]
        vals = [
            sub.loc[sub["language"] == lang, "jsd"].values[0]
            if lang in sub["language"].values else np.nan
            for lang in LANGUAGES
        ]
        ax.bar(
            x + j * width, vals, width, label=model.capitalize(),
            color=MODEL_COLORS[model], alpha=0.88, edgecolor="white",
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels([l.capitalize() for l in LANGUAGES])
    ax.set_ylabel("Mean JSD  (↓ = better calibrated)")
    ax.set_title("Fire vs. Global vs. Earth — Mean JSD by Language Group")
    ax.legend(title="Model")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _save(fig, output_dir, "model_comparison.png")


def plot_differentiation(diff_df: pd.DataFrame, output_dir: str):
    """
    Scatter: model pairwise JSD vs. human pairwise JSD per language pair.
    Points on the diagonal → model differentiates cultures as humans do.
    Points below → model under-differentiates (homogenises).
    """
    if diff_df.empty:
        return
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = diff_df[diff_df["model"] == model]
        ax.scatter(
            sub["human_pairwise_jsd"], sub["model_pairwise_jsd"],
            alpha=0.45, s=22, color=MODEL_COLORS[model],
        )
        lim = max(
            sub[["human_pairwise_jsd", "model_pairwise_jsd"]].max().max() * 1.1, 0.05
        )
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect agreement")
        ax.set_title(model.capitalize())
        ax.set_xlabel("Human pairwise JSD")
        if ax == axes[0]:
            ax.set_ylabel("Model pairwise JSD")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle(
        "Cultural Differentiation: Model vs. Human Pairwise JSD\n"
        "(near diagonal = model differentiates language groups as humans do;\n"
        " below diagonal = homogenisation)",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, output_dir, "differentiation.png")


def plot_inglehart_map(inglehart_df: pd.DataFrame, output_dir: str):
    """
    Inglehart cultural map scatter.
    x-axis: Survival ← → Self-Expression
    y-axis: Traditional ← → Secular-Rational
    """
    if inglehart_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Human positions — black, larger
    human = inglehart_df[inglehart_df["source"] == "human"]
    for _, row in human.iterrows():
        marker = LANG_MARKERS.get(row["language"], "o")
        ax.scatter(
            row["surv_selfexp_score"], row["trad_secular_score"],
            marker=marker, s=200, color="black", zorder=5,
            edgecolors="white", linewidths=1.5,
        )
        ax.annotate(
            f"H:{row['language'][:3]}",
            (row["surv_selfexp_score"], row["trad_secular_score"]),
            fontsize=8, ha="center", va="bottom", color="black",
        )

    # Model positions — coloured, smaller
    for model in MODELS:
        model_df = inglehart_df[inglehart_df["source"] == model]
        for _, row in model_df.iterrows():
            marker = LANG_MARKERS.get(row["language"], "o")
            ax.scatter(
                row["surv_selfexp_score"], row["trad_secular_score"],
                marker=marker, s=110, color=MODEL_COLORS[model],
                alpha=0.88, zorder=4, edgecolors="white", linewidths=0.5,
            )
            ax.annotate(
                f"{model[0].upper()}:{row['language'][:3]}",
                (row["surv_selfexp_score"], row["trad_secular_score"]),
                fontsize=7, ha="center", va="top", color=MODEL_COLORS[model],
            )

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("← Survival                         Self-Expression →", fontsize=10)
    ax.set_ylabel("← Traditional                   Secular-Rational →", fontsize=10)
    ax.set_title(
        "Inglehart Cultural Map — Model vs. Human Positions\n"
        "(WVS Wave 7 India, disaggregated by interview language)",
        fontsize=11,
    )
    model_patches = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m.capitalize()) for m in MODELS
    ]
    model_patches.append(mpatches.Patch(color="black", label="Human (WVS)"))
    ax.legend(handles=model_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "inglehart_map.png")


def _save(fig, output_dir: str, filename: str):
    path = os.path.join(output_dir, "plots", filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer 2: WVS value-calibration evaluation for Tiny Aya models"
    )
    p.add_argument(
        "--inference_dir", required=True,
        help="Directory of model inference JSON files (Ananya's pipeline output)",
    )
    p.add_argument(
        "--wvs_human_path", required=True,
        help="Path to WVS Wave 7 India human distributions JSON (Tanay's preprocessed data)",
    )
    p.add_argument("--output_dir",  default="results/layer2")
    p.add_argument("--n_bootstrap", type=int,   default=1000,
                   help="Bootstrap replicates for CI computation (default: 1000)")
    p.add_argument("--ci_alpha",    type=float, default=0.05,
                   help="Alpha for CIs (default: 0.05 → 95%% CI)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument(
        "--inglehart_only", action="store_true",
        help="Load existing jsd_scores.csv and only recompute Inglehart + plots",
    )
    p.add_argument("--no_plots", action="store_true", help="Skip plot generation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "plots")).mkdir(exist_ok=True)

    print(f"Loading WVS human distributions from  {args.wvs_human_path}")
    human_distributions = load_human_distributions(args.wvs_human_path)

    print(f"Loading model inference outputs from   {args.inference_dir}")
    model_outputs = load_model_outputs(args.inference_dir)

    jsd_path = os.path.join(args.output_dir, "jsd_scores.csv")

    # ── JSD computation ────────────────────────────────────────────────────
    if args.inglehart_only and os.path.exists(jsd_path):
        print(f"--inglehart_only: loading existing {jsd_path}")
        jsd_df = pd.read_csv(jsd_path)
    else:
        print(f"\nComputing JSD scores  (n_bootstrap={args.n_bootstrap}, alpha={args.ci_alpha})")
        jsd_df = compute_all_jsd(
            model_outputs, human_distributions,
            n_bootstrap=args.n_bootstrap, alpha=args.ci_alpha, seed=args.seed,
        )
        jsd_df.to_csv(jsd_path, index=False)
        print(f"Saved {jsd_path}")

        summary_df = compute_jsd_summary(jsd_df)
        summary_path = os.path.join(args.output_dir, "jsd_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")

    # ── Differentiation ────────────────────────────────────────────────────
    print("\nComputing differentiation (pairwise language JSD) ...")
    diff_df = compute_differentiation(model_outputs, human_distributions)
    diff_path = os.path.join(args.output_dir, "differentiation.csv")
    diff_df.to_csv(diff_path, index=False)
    print(f"Saved {diff_path}")

    # ── Inglehart positions ────────────────────────────────────────────────
    print("\nComputing Inglehart cultural map positions ...")
    inglehart_df = compute_inglehart_positions(model_outputs, human_distributions)
    ing_path = os.path.join(args.output_dir, "inglehart_positions.csv")
    inglehart_df.to_csv(ing_path, index=False)
    print(f"Saved {ing_path}")

    # ── Console summary ────────────────────────────────────────────────────
    print_highlights(jsd_df, inglehart_df)

    # ── Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots ...")
        plot_jsd_heatmap(jsd_df, args.output_dir)
        plot_model_comparison(jsd_df, args.output_dir)
        plot_differentiation(diff_df, args.output_dir)
        plot_inglehart_map(inglehart_df, args.output_dir)

    print("\nLayer 2 evaluation complete.")


if __name__ == "__main__":
    main()
