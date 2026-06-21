# Retrain an LLM — Knowledge Injection & Distillation

A progressive, notebook-driven course that **injects new knowledge into a teacher
model and distills it into a much smaller student** — entirely from scratch in
PyTorch (no HuggingFace), sized to run on a laptop in minutes.

The spine: each step earns its place by beating a **measured** number from the
step before.

> teacher pretrain → inject facts → baseline student (fails) → distilled student (recovers the knowledge)

The model code is vendored from the sibling `train-llm` project (a small
Llama-style decoder: GQA + RoPE + RMSNorm + SwiGLU + weight tying + KV-cache), so
this repo runs standalone.

## The curriculum

| #  | Notebook | What you build |
|----|----------|----------------|
| 00 | Setup & tour | environment check, device selection, the roadmap |
| 01 | Fact-set & cloze eval | a synthetic fictional knowledge base + the recall metric |
| 02 | Train the teacher | pretrain a broad GPT on TinyShakespeare; measure perplexity |
| 03 | Inject knowledge | fine-tune the teacher on the facts; cloze ~0% → ~98% |
| 04 | Student baseline | a small student on hard labels — the number to beat |
| 05 | Logit distillation | KL on the teacher's soft labels; the temperature lesson |
| 06 | Alpha mixing | blend ground-truth + KD; an honest sweep |
| 07 | Hidden-state distillation | feature matching (and an honest negative result) |
| 08 | Synthetic-data augmentation | the teacher generates extra training data |
| 09 | Size sweep | how small can the student go and still recall? |
| 10 | Capstone | the best recipe → a 12× smaller student |
| 11 | Recap | the whole measured journey, summarized |

## Results

Every step is measured by **cloze accuracy** on a held-out probe (does the model
complete `"The capital of Zorbia is ___"` correctly?):

| Step | Cloze |
|------|-------|
| Teacher, after fact injection | ~98% |
| Student baseline (hard labels, fixed budget) | 60.0% |
| Logit distillation (temperature 1.0) | 67.5% |
| Best alpha blend | 67.5% |
| Hidden-state matching | 60.0% (did not help here — reported honestly) |
| Teacher-augmented distillation | 67.5% |
| **Capstone** (best recipe, longer budget) | **95.0%** — at **~12× fewer parameters** than the teacher |

A finding worth its own lesson (notebook 05): for crisp factual recall, the
textbook distillation temperature of 2.0 *hurts* — it dilutes the answer signal
and underperforms hard labels. Temperature **1.0** is what wins here. Not every
advanced trick helps either: naive hidden-state matching didn't move the number,
and the notebooks say so rather than hiding it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
```

## How to run

Notebooks live in `notebooks/` as paired files: a jupytext `.py` source (the
version-controlled source of truth) and a generated `.ipynb`.

```bash
jupyter lab                                   # interactive
python notebooks/01_factset_and_cloze.py      # headless
```

The broad corpus (TinyShakespeare) auto-downloads on first run.
