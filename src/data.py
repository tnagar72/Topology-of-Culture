"""
DOSA dataset fetching and eval-set construction.
"""

import os
import json
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

from .config import (
    STATE_TO_GROUP,
    DOSA_BASE_URL,
    SYSTEM_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT_1,
    INSTRUCTION_PROMPT_2,
)


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


def build_eval_set(dosa, output_path):
    """Build the eval set from a DOSA DataFrame and save to JSON."""
    eval_set = []
    for _, row in dosa.iterrows():
        state_display  = row["state"].replace("_", " ").title()
        formatted_clues = _format_clues(row["clues"])

        eval_set.append({
            "id":            f"{row['state']}_{row.name}",
            "state":         row["state"],
            "state_display": state_display,
            "group":         row["group"],
            "artifact":      row["artifact"],
            "clues":         formatted_clues,
            "system_prompt": SYSTEM_PROMPT_TEMPLATE.format(clues=formatted_clues),
            "prompt1":       INSTRUCTION_PROMPT_1.format(state=state_display),
            "prompt2":       INSTRUCTION_PROMPT_2,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"\nBuilt {len(eval_set)} items → {output_path}")
    return eval_set


def _format_clues(raw_clues_str):
    """Format clue string as a numbered list matching the DOSA paper."""
    clues = raw_clues_str.strip().split("\n")
    return "\n".join(f"CLUE-{i+1}: {c.strip()}" for i, c in enumerate(clues))
