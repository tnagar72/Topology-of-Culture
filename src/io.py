"""
JSONL I/O utilities shared across inference and evaluation.
"""

import json
import pandas as pd


def load_run_jsonl(path):
    """Load a run JSONL file into a DataFrame."""
    return pd.read_json(path, lines=True)


def save_run_jsonl(records, path):
    """Save a list of result dicts to JSONL, converting NaN → None."""
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            clean = {
                k: (None if isinstance(v, float) and pd.isna(v) else v)
                for k, v in record.items()
            }
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
