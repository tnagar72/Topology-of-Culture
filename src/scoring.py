"""
Answer parsing, correctness checking, and run scoring (single-sample and recall@k).
"""

from collections import Counter
import pandas as pd


# ---------------------------------------------------------------------------
# Default parse / correctness (matching DOSA paper exactly)
# ---------------------------------------------------------------------------

def parse_answer(response):
    """Split on first colon, take right-hand side (DOSA paper Appendix A)."""
    parts = response.split(":", 1)
    if len(parts) > 1:
        return parts[1].strip().lower()
    return response.strip().lower()


def is_correct(prediction, ground_truth):
    """Ground truth is a substring of prediction, lowercased (DOSA paper)."""
    return ground_truth.lower().strip() in prediction.lower()


# ---------------------------------------------------------------------------
# Run scoring — dispatches on schema
# ---------------------------------------------------------------------------

def score_run(df, parse_fn=None, correct_fn=None):
    """
    Score a run DataFrame. Dispatches to single-sample or recall@k scoring
    based on whether a guess1_samples column is present.

    Returns a new DataFrame with added columns:
        guess1, correct1, guess2, correct2, correct_any, parse_fn

    For recall@k runs, also adds:
        correct1_count, guess1_most_frequent, guess1_most_frequent_count

    Parameters
    ----------
    df : DataFrame
        Single-sample: must have artifact, guess1_raw, guess2_raw.
        Recall@k:      must have artifact, guess1_samples.
    parse_fn : callable, optional
        (raw_response: str) -> str  — defaults to parse_answer.
    correct_fn : callable, optional
        (prediction: str, ground_truth: str) -> bool — defaults to is_correct.
    """
    if parse_fn is None:
        parse_fn = parse_answer
    if correct_fn is None:
        correct_fn = is_correct

    df = df.copy()
    df["parse_fn"] = getattr(parse_fn, "__name__", repr(parse_fn))

    if "guess1_samples" in df.columns:
        return _score_run_recall_k(df, parse_fn, correct_fn)
    return _score_run_single(df, parse_fn, correct_fn)


def _score_run_single(df, parse_fn, correct_fn):
    """Score a single-sample run (guess1_raw / guess2_raw columns)."""
    df["guess1"]   = df["guess1_raw"].apply(parse_fn)
    df["correct1"] = df.apply(
        lambda r: correct_fn(r["guess1"], r["artifact"]), axis=1
    )

    g2 = df.apply(lambda r: _score_single_guess2(r, parse_fn, correct_fn), axis=1)
    df["guess2"]      = [r[0] for r in g2]
    df["correct2"]    = [r[1] for r in g2]
    df["correct_any"] = df.apply(
        lambda r: r["correct1"] or bool(r["correct2"]), axis=1
    )
    return df


def _score_single_guess2(row, parse_fn, correct_fn):
    """Parse and score a single guess2 response; returns (guess2, correct2)."""
    raw = row.get("guess2_raw")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    g2 = parse_fn(str(raw))
    return g2, correct_fn(g2, row["artifact"])


def _score_run_recall_k(df, parse_fn, correct_fn):
    """
    Score a recall@k run (guess1_samples column only — no guess2 turn).

    Adds:
        correct1                   — True if any of the k samples is correct
        correct1_count             — how many of the k samples are correct
        guess1_most_frequent       — most common parsed sample
        guess1_most_frequent_count — its frequency across k samples
        guess1                     — alias for guess1_most_frequent
        correct_any                — same as correct1 (no guess2 in recall@k)
    """
    g1 = df.apply(lambda r: _score_recall_k_guess1(r, parse_fn, correct_fn), axis=1)
    df["correct1"]                   = [r[0] for r in g1]
    df["correct1_count"]             = [r[1] for r in g1]
    df["guess1_most_frequent"]       = [r[2] for r in g1]
    df["guess1_most_frequent_count"] = [r[3] for r in g1]
    df["guess1"]      = df["guess1_most_frequent"]
    df["correct_any"] = df["correct1"]
    return df


def _score_recall_k_guess1(row, parse_fn, correct_fn):
    """Parse and score all k guess1 samples; returns (any_correct, count, most_freq, freq)."""
    parsed        = [parse_fn(s) for s in row["guess1_samples"]]
    correct_flags = [correct_fn(p, row["artifact"]) for p in parsed]
    most_freq, most_freq_count = Counter(parsed).most_common(1)[0]
    return any(correct_flags), sum(correct_flags), most_freq, most_freq_count
