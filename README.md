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
| 02 | Train the teacher | pretrain a broad GPT; measure perplexity |
| 03 | Inject knowledge | fine-tune the teacher on the facts; cloze ~0 → ~100% |
| 04–11 | Distillation | baseline student, logit/feature distillation, augmentation, size sweep, capstone |

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
