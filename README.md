# Topology of Culture
### Evaluating Cultural Competence in South Asian Language Models
*C4AI Expedition — Tiny Aya*


## Overview

Does regional post-training actually help a language model understand regional cultures?
This project investigates that question across **7 South Asian languages** using two complementary benchmarks, comparing three variants of the Tiny Aya model that differ only in their post-training data.

The central concern is **cultural homogenization** — the tendency of multilingual models to collapse distinct South Asian communities (Telugu, Punjabi, Bengali, Marathi, Hindi-Urdu, Gujarati, Tamil) into a generic, undifferentiated "Indian" representation. We test whether targeted South Asian post-training measurably counters this effect.


## Models

All three models share the same base. The only difference is post-training data.

| Model | Post-training Focus | Role |
|---|---|---|
| **Tiny Aya Fire** | South Asian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Nepali, Urdu) | Primary treatment |
| **Tiny Aya Global** | Balanced across 70+ languages | Baseline |
| **Tiny Aya Earth** | West Asian & African languages | Controls for regional post-training in general |


## Evaluation

We use a two-layer evaluation framework to test cultural competence from different angles.

### Layer 1: DOSA (Surface Cultural Knowledge)
**Dataset:** [DOSA](https://arxiv.org/abs/2403.14651) 129 community-generated cultural artifacts across 5 Indian language groups (Telugu, Punjabi, Bengali, Marathi, Hindi-Urdu), collected from 260 participants across 19 states.

**Task:** A Taboo-style guessing game. Given community-written clues describing a cultural artifact, the model must name it in English, from memory, with no multiple-choice options. Two attempts are allowed.

**Metrics:** `accuracy@GUESS1`, `accuracy@GUESS2`, `overall accuracy` reported per model and per language group.

**Key property:** Tests whether models have internalized culturally specific knowledge and can *distinguish between* communities rather than flattening them.


### Layer 1b: MILU (Cultural Reasoning in Native Scripts)
**Dataset:** [MILU](https://arxiv.org/abs/2411.02538) — ~79,000 multiple-choice questions from 1,500+ real Indian competitive exams across 11 Indic languages, spanning 8 domains and 41 subjects.

**Evaluated languages:** Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi + English (baseline).

**Domains (preliminary):** Arts & Humanities, Social Sciences, Law & Governance — the domains most dense with region-specific cultural content. (~75% of MILU questions are in their original source language, not translated.)

**Evaluation:** 0-shot, log-likelihood scoring via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), with length normalization across answer options.

**Key property:** Unlike DOSA, questions are in native scripts. This directly tests linguistic competence *and* culturally situated reasoning together.


## Key Results

| Model | Avg. Accuracy (MILU) |
|---|---|
| **Tiny Aya Fire** | **33.0%** |
| Tiny Aya Global | 31.4% |
| Tiny Aya Earth | 31.3% |

- **5 of 7 Indic languages** show statistically significant improvement (p < 0.05) for Fire over Global.
- **Fire outperforms Aya-23-35B** (4× larger, 5-shot) on 3 low-resource languages: Gujarati (+2.4%), Punjabi (+5.7%), Telugu (+3.6%).
- Earth does not outperform Global, confirming the gains are **specific to South Asian training data**, not a byproduct of any regional post-training.


## Repository Structure
Topology-of-Culture/
├── data/
│   └── dosa/               # DOSA artifacts and clues by language group
├── results/
│   └── dosa/               # Model outputs and scored results
└── src/                    # Evaluation scripts

## References

- Verma et al. (2025). *MILU: A Multi-task Indic Language Understanding Benchmark.* [arXiv:2411.02538](https://arxiv.org/abs/2411.02538)
- Ahuja et al. (2024). *DOSA: A Dataset of Social Artifacts from Different Indian Geographical Subcultures.* LREC-COLING 2024. [arXiv:2403.14651](https://arxiv.org/abs/2403.14651)
