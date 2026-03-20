"""
Cohere API client, decoder factory, and inference loop.
"""

import os
import cohere

from .io import load_run_jsonl, save_run_jsonl
from .scoring import parse_answer, is_correct


# ---------------------------------------------------------------------------
# Cohere client
# ---------------------------------------------------------------------------

def make_cohere_client():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY environment variable not set")
    return cohere.ClientV2(api_key=api_key)


def generate_response(co, model_name, messages, **sampling_kwargs):
    """Single Cohere chat API call. Returns the raw text response."""
    response = co.chat(
        model=model_name,
        messages=messages,
        max_tokens=30,
        **sampling_kwargs,
    )
    if not response.message or not response.message.content:
        raise ValueError(f"Empty response from Cohere API for model {model_name}")
    return response.message.content[0].text.strip()


# ---------------------------------------------------------------------------
# Decoder factory
# ---------------------------------------------------------------------------

def make_decoder(temperature=0.0, top_k=None, top_p=None, num_samples=1):
    """
    Unified decoder factory.

    Returns a callable: (co, model_name, messages) -> str | list[str]
      - num_samples=1  → returns a single raw string  (two-turn inference)
      - num_samples>1  → returns a list of raw strings (recall@k, guess1 only)

    Parameters
    ----------
    temperature : float
        0.0 for greedy/deterministic; >0 to sample from the distribution.
    top_k : int | None
        Restrict sampling to the top-k tokens at each step.
    top_p : float | None
        Nucleus sampling — keep tokens up to cumulative probability p.
    num_samples : int
        Number of independent samples per call. >1 enables recall@k mode.
    """
    sampling_kwargs = {"temperature": temperature}
    if top_k is not None:
        sampling_kwargs["k"] = top_k
    if top_p is not None:
        sampling_kwargs["p"] = top_p

    def decoder(co, model_name, messages):
        if num_samples == 1:
            return generate_response(co, model_name, messages, **sampling_kwargs)
        return [
            generate_response(co, model_name, messages, **sampling_kwargs)
            for _ in range(num_samples)
        ]

    parts = [f"temp={temperature}"]
    if top_k is not None:
        parts.append(f"top_k={top_k}")
    if top_p is not None:
        parts.append(f"top_p={top_p}")
    if num_samples > 1:
        parts.append(f"n={num_samples}")
    decoder.__name__ = f"decoder({','.join(parts)})"
    return decoder


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(eval_set, models, output_path, decoder, decoder_name):
    """
    Unified inference loop for all models.

    Single-sample mode (decoder returns str):
      Two-turn: stores guess1_raw, guess2_raw. guess2 is skipped when
      guess1 is correct.

    Recall@k mode (decoder returns list[str]):
      Single-turn: stores guess1_samples only (no guess2). Correct if any
      sample is correct; scoring details deferred to dosa_eval.py.

    Saves incrementally to JSONL after each model completes (crash recovery).
    """
    co = make_cohere_client()

    all_results, completed_models = _load_existing(output_path)

    for model_name in models:
        if model_name in completed_models:
            print(f"  Skipping {model_name} (already complete)")
            continue

        print(f"  {model_name}...")
        model_results = _run_model(eval_set, co, model_name, decoder, decoder_name)

        all_results.extend(model_results)
        save_run_jsonl(all_results, output_path)
        print(f"  Saved → {output_path}")

    return _to_dataframe(all_results)


def _load_existing(output_path):
    """Load any previously completed results for crash recovery."""
    if os.path.exists(output_path):
        existing = load_run_jsonl(output_path)
        completed_models = set(existing["model"].unique())
        print(f"  Resuming — already complete: {completed_models}")
        return existing.to_dict("records"), completed_models
    return [], set()


def _run_model(eval_set, co, model_name, decoder, decoder_name):
    """Run inference for a single model over the full eval set."""
    results = []
    for i, item in enumerate(eval_set):
        record = _run_item(co, model_name, item, decoder, decoder_name)
        results.append(record)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(eval_set)} done")
    return results


def _run_item(co, model_name, item, decoder, decoder_name):
    """Run one eval item and return the result record."""
    messages1 = [
        {"role": "system", "content": item["system_prompt"]},
        {"role": "user",   "content": item["prompt1"]},
    ]
    result1 = decoder(co, model_name, messages1)

    if isinstance(result1, list):
        return _recall_k_record(item, model_name, decoder_name, result1)
    return _single_sample_record(co, item, model_name, decoder, decoder_name, messages1, result1)


def _recall_k_record(item, model_name, decoder_name, samples):
    """Build a recall@k record (guess1_samples only, no guess2 turn)."""
    return {
        "id":             item["id"],
        "state":          item["state"],
        "group":          item["group"],
        "artifact":       item["artifact"],
        "model":          model_name,
        "decoder":        decoder_name,
        "guess1_samples": samples,
    }


def _single_sample_record(co, item, model_name, decoder, decoder_name, messages1, raw_guess1):
    """Build a single-sample record, attempting guess2 if guess1 is wrong."""
    # Use default parse/correct only to decide whether to attempt guess2.
    # Scoring is deferred to dosa_eval.py.
    correct1   = is_correct(parse_answer(raw_guess1), item["artifact"])
    raw_guess2 = None if correct1 else _attempt_guess2(
        co, model_name, messages1, raw_guess1, item["prompt2"], decoder
    )
    return {
        "id":         item["id"],
        "state":      item["state"],
        "group":      item["group"],
        "artifact":   item["artifact"],
        "model":      model_name,
        "decoder":    decoder_name,
        "guess1_raw": raw_guess1,
        "guess2_raw": raw_guess2,
    }


def _attempt_guess2(co, model_name, messages1, raw_guess1, prompt2, decoder):
    """Build the two-turn conversation and get guess2."""
    messages2 = messages1 + [
        {"role": "assistant", "content": raw_guess1},
        {"role": "user",      "content": prompt2},
    ]
    return decoder(co, model_name, messages2)


def _to_dataframe(records):
    """Convert result records to a DataFrame, handling mixed schemas gracefully."""
    import pandas as pd
    return pd.DataFrame(records)
