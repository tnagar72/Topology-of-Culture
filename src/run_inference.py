"""
Layer 2 WVS Inference
======================
Prompts Tiny Aya Fire / Global / Earth on all 13 WVS questions
across 5 Indian language groups, collecting N samples per combination.

USAGE
-----
    python src/run_inference.py --output_dir results/layer2/inference --n_samples 50

    # Resume interrupted run (skips already-completed files):
    python src/run_inference.py --output_dir results/layer2/inference --n_samples 50

COHERE API KEY
--------------
Set COHERE_API_KEY in your environment or .env file.
"""

import os
import re
import json
import time
import argparse
from pathlib import Path

import cohere
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(r"C:\Users\rafay.mustafa\Desktop\culture-repo\.env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS    = ["tiny-aya-fire", "tiny-aya-global", "tiny-aya-earth"]
LANGUAGES = ["hindi", "telugu", "marathi", "bengali", "punjabi"]

LANG_CONTEXT = {
    "hindi":   "You are a person living in India whose primary language is Hindi.",
    "telugu":  "You are a person living in India whose primary language is Telugu.",
    "marathi": "You are a person living in India whose primary language is Marathi.",
    "bengali": "You are a person living in India whose primary language is Bengali.",
    "punjabi": "You are a person living in India whose primary language is Punjabi.",
}

# WVS question prompts — exact scale wording from WVS Wave 7 codebook
QUESTION_PROMPTS = {
    "Q45": (
        "If there were more respect for authority in the future, would you consider that a good or bad thing?\n"
        "1 = A good thing\n2 = Don't mind\n3 = A bad thing\n"
        "Reply with ONLY the number (1, 2, or 3).",
        1, 3
    ),
    "Q46": (
        "Taking all things together, would you say you are:\n"
        "1 = Very happy\n2 = Quite happy\n3 = Not very happy\n4 = Not at all happy\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q49": (
        "All things considered, how satisfied are you with your life as a whole these days?\n"
        "1 = Completely dissatisfied ... 10 = Completely satisfied\n"
        "Reply with ONLY a number from 1 to 10.",
        1, 10
    ),
    "Q57": (
        "Generally speaking, would you say that:\n"
        "1 = Most people can be trusted\n2 = You need to be very careful in dealing with people\n"
        "Reply with ONLY 1 or 2.",
        1, 2
    ),
    "Q152": (
        "Which should be the most important aim for this country over the next ten years?\n"
        "1 = A high level of economic growth\n"
        "2 = Strong defence forces\n"
        "3 = People have more say about how things are done\n"
        "4 = Trying to make our cities and countryside more beautiful\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q153": (
        "Which should be the second most important aim for this country over the next ten years?\n"
        "1 = A high level of economic growth\n"
        "2 = Strong defence forces\n"
        "3 = People have more say about how things are done\n"
        "4 = Trying to make our cities and countryside more beautiful\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q154": (
        "Which one of these is most important to you personally?\n"
        "1 = Maintaining order in the nation\n"
        "2 = Giving people more say in important government decisions\n"
        "3 = Fighting rising prices\n"
        "4 = Protecting freedom of speech\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q155": (
        "Which one would be the next most important to you personally?\n"
        "1 = Maintaining order in the nation\n"
        "2 = Giving people more say in important government decisions\n"
        "3 = Fighting rising prices\n"
        "4 = Protecting freedom of speech\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q156": (
        "Which of these do you consider most important?\n"
        "1 = A stable economy\n"
        "2 = Progress toward a less impersonal and more humane society\n"
        "3 = Progress toward a society where ideas count more than money\n"
        "4 = The fight against crime\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q157": (
        "Which of these do you consider the next most important?\n"
        "1 = A stable economy\n"
        "2 = Progress toward a less impersonal and more humane society\n"
        "3 = Progress toward a society where ideas count more than money\n"
        "4 = The fight against crime\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
    "Q164": (
        "How important is God in your life?\n"
        "1 = Not at all important ... 10 = Very important\n"
        "Reply with ONLY a number from 1 to 10.",
        1, 10
    ),
    "Q209": (
        "Have you ever signed a petition?\n"
        "1 = Have done\n2 = Might do\n3 = Would never do\n"
        "Reply with ONLY the number (1, 2, or 3).",
        1, 3
    ),
    "Q254": (
        "How proud are you to be Indian?\n"
        "1 = Very proud\n2 = Quite proud\n3 = Not very proud\n4 = Not at all proud\n"
        "Reply with ONLY the number (1, 2, 3, or 4).",
        1, 4
    ),
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_prompt(language: str, question_text: str) -> str:
    return (
        f"{LANG_CONTEXT[language]}\n"
        "Answer the following survey question as yourself.\n\n"
        f"{question_text}"
    )


def parse_response(text: str, scale_min: int, scale_max: int):
    """Extract the first integer in range from model response. Returns None if not found."""
    match = re.search(r"\b(\d+)\b", text.strip())
    if match:
        val = int(match.group(1))
        if scale_min <= val <= scale_max:
            return val
    return None


def run_inference(co, model_name: str, language: str, qid: str,
                  question_text: str, scale_min: int, scale_max: int,
                  n_samples: int) -> dict:
    prompt = build_prompt(language, question_text)
    short_model = model_name.replace("tiny-aya-", "")

    responses = []
    non_responses = 0

    for i in range(n_samples):
        try:
            resp = co.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=5,
            )
            raw = resp.message.content[0].text.strip()
            val = parse_response(raw, scale_min, scale_max)
            if val is not None:
                responses.append(val)
            else:
                non_responses += 1
        except Exception as e:
            print(f"    API error: {e}. Retrying in 5s...")
            time.sleep(5)
            non_responses += 1

    return {
        "model":         short_model,
        "language":      language,
        "question_id":   qid,
        "responses":     responses,
        "non_responses": non_responses,
        "n_requested":   n_samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results/layer2/inference")
    p.add_argument("--n_samples",  type=int, default=50)
    args = p.parse_args()

    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not set")

    co = cohere.ClientV2(api_key=api_key)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(MODELS) * len(LANGUAGES) * len(QUESTION_PROMPTS)
    done  = 0

    for model_name in MODELS:
        short = model_name.replace("tiny-aya-", "")
        for language in LANGUAGES:
            for qid, (question_text, s_min, s_max) in QUESTION_PROMPTS.items():
                fname = out_dir / f"{short}_{language}_{qid}.json"

                if fname.exists():
                    done += 1
                    print(f"[{done}/{total}] Skip (exists): {fname.name}")
                    continue

                done += 1
                print(f"[{done}/{total}] {short} | {language} | {qid} ...", end=" ", flush=True)

                result = run_inference(
                    co, model_name, language, qid,
                    question_text, s_min, s_max, args.n_samples
                )

                with open(fname, "w") as f:
                    json.dump(result, f)

                n_valid = len(result["responses"])
                print(f"{n_valid}/{args.n_samples} valid responses")

    print(f"\nDone. Files saved to {out_dir}")


if __name__ == "__main__":
    main()
