# Retrain an LLM — Knowledge Injection & Distillation

![Python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/pytorch-v2.2%2B-blue?logo=pytorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter&logoColor=white)
![Knowledge Distillation](https://img.shields.io/badge/knowledge-distillation-7c3aed)
![Knowledge Injection](https://img.shields.io/badge/knowledge-injection-7c3aed)
![From Scratch](https://img.shields.io/badge/from%20scratch-no%20HuggingFace-success)
![License](https://img.shields.io/badge/license-Apache_2.0-blue)

A progressive, notebook-driven course that **injects new knowledge into a larger
"teacher" model and distills it into a much smaller "student"** — entirely from
scratch in PyTorch (no HuggingFace), sized to train on a laptop in minutes.

The defining principle, inherited from its sibling course
[`train-llm`](https://github.com/orbek/train-llm): every step earns its place by
beating a **measured** number from the step before. Here the number is **cloze
accuracy** — can the model fill in `"The capital of Zorbia is ___"` correctly?

> teacher pretrain → **inject** facts → baseline student (fails) → **distill** → student recovers the knowledge

Each notebook teaches step by step in plain language, with runnable code, inline
plots, and `assert`-based sanity checks that double as lessons.

## The story

We invent a small **fictional knowledge base** — facts no real corpus could
already contain, like `"The capital of Zorbia is Quee"`. Because the facts are
invented, recall is unambiguous to measure and impossible to fake.

1. **Pretrain** a broad teacher on TinyShakespeare so it has fluent language.
2. **Inject** the facts by fine-tuning the teacher — its recall jumps from ~0% to ~98%.
3. **Distill** that knowledge into a student a fraction of the size, and measure
   how much transfers — versus a same-size student trained on the facts alone.

The recurring question: *how do you get a small model to inherit specific
knowledge a larger model has?*

## The curriculum

| #  | Notebook | What you build |
|----|----------|----------------|
| 00 | Setup & tour | environment check, automatic device selection, the roadmap |
| 01 | Fact-set & cloze eval | a synthetic fictional knowledge base; the held-out cloze recall metric |
| 02 | Train the teacher | pretrain a broad GPT on TinyShakespeare; perplexity; confirm it doesn't know the facts yet |
| 03 | Inject knowledge | fine-tune the teacher on the facts; cloze ~0% → ~98% |
| 04 | Student baseline | a small student on hard labels at a fixed budget — the number to beat |
| 05 | Logit distillation | KL divergence on the teacher's soft labels; the temperature lesson |
| 06 | Alpha mixing | blend ground-truth and KD loss; an honest sweep that includes the baseline |
| 07 | Hidden-state distillation | feature matching across layers (and an honest negative result) |
| 08 | Synthetic-data augmentation | the teacher generates extra fact-flavored training data |
| 09 | Size sweep | how small can the student go and still recall? |
| 10 | Capstone | the best recipe → a ~12× smaller student at near-teacher recall |
| 11 | Recap | the whole measured journey, summarized |

The reusable code lives at the repo root:

- [`model.py`](model.py) — a small Llama-style decoder (GQA + RoPE + RMSNorm +
  SwiGLU + weight tying + KV-cache), vendored from `train-llm` and extended with a
  hidden-state hook and teacher/student config presets. Tested by
  [`test_model.py`](test_model.py).
- [`facts.py`](facts.py) — the synthetic fact-set generator, the shared
  char-level vocabulary, and the cloze evaluator. Tested by [`test_facts.py`](test_facts.py).
- [`distill.py`](distill.py) — the distillation toolkit: temperature-scaled KL
  loss, alpha-mixing with ground-truth, hidden-state feature matching, and the
  teacher→student training loop. Tested by [`test_distill.py`](test_distill.py).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
pip install -r requirements.txt
```

(Optionally use [`uv`](https://github.com/astral-sh/uv): `uv venv && uv pip install -r requirements.txt` — faster.)

### Hardware

The notebooks automatically detect and use the best available compute engine —
no manual configuration needed:

- **NVIDIA GPU (CUDA)** on Windows or Linux — used if available.
- **Apple Silicon GPU (MPS)** on macOS (M1–M4) — used if CUDA is not present.
- **CPU** — universal fallback; works on any machine, just slower for training.

> **Windows / Linux NVIDIA users:** the default `pip install torch` may install a
> CPU-only build. If PyTorch does not detect your GPU, install a CUDA-enabled
> build from <https://pytorch.org>.

Everything is sized to run on a laptop in minutes; the teacher (~9.4M params) and
students (~0.1–0.8M) train on Apple Silicon (MPS) or CPU.

## How to run

Notebooks live in `notebooks/` as paired files: a jupytext `.py` source (the
version-controlled source of truth) and a generated `.ipynb`. Open the `.ipynb`
in JupyterLab:

```bash
jupyter lab
```

Or run a notebook headlessly as a script:

```bash
python notebooks/01_factset_and_cloze.py
```

To regenerate a rendered `.ipynb` from its `.py` source:

```bash
jupytext --to notebook --execute --set-kernel python3 notebooks/<file>.py -o notebooks/<file>.ipynb
```

Run the test suites at any time:

```bash
python test_model.py && python test_facts.py && python test_distill.py
```

The TinyShakespeare corpus auto-downloads on first run (notebook 02). The
notebooks save model checkpoints to `checkpoints/` (gitignored); run them in
order, as later notebooks load the teacher the earlier ones produced.

## Results

Every step is judged by **cloze accuracy** on a *held-out* probe template — the
exact phrasing is never seen in training, so recall measures the fact, not a
memorized string. Numbers from the committed runs:

| Step | Cloze | Notes |
|------|------:|-------|
| Teacher, after fact injection | ~98% | the knowledge to transfer |
| Student baseline (hard labels, 400 steps) | 60.0% | the number to beat |
| Logit distillation (temperature 1.0) | 67.5% | soft labels beat hard at the same budget |
| Best alpha blend | 67.5% | blending ground-truth + KD |
| Hidden-state matching | 60.0% | **did not help here — reported honestly** |
| Teacher-augmented distillation | 67.5% | more teacher-labeled data |
| **Capstone** (best recipe, 800 steps) | **95.0%** | at **~12× fewer parameters** than the teacher |

Two honest lessons the notebooks make explicit rather than hide:

- **Temperature matters, and the textbook value is wrong here.** For crisp
  factual recall, the standard distillation temperature of 2.0 *dilutes* the
  answer signal and underperforms plain hard labels. **Temperature 1.0** is what
  wins. (Notebook 05 shows the measured comparison.)
- **Not every advanced trick helps.** Naive hidden-state feature matching didn't
  move the number in this regime — so the capstone leaves it out. A real result
  worth teaching.

The payoff: a student roughly **12× smaller** than the teacher recovers ~95% of
the injected knowledge — knowledge a same-size model trained on the tiny fact
corpus alone could not fully learn.

## Relationship to `train-llm`

This repo is a standalone sibling of [`train-llm`](https://github.com/orbek/train-llm)
(which builds the base transformer from scratch). The model architecture is
vendored here so this course runs on its own, with no dependency on that repo.
`train-llm` teaches *how to build and train* the model; `retrain-llm` teaches
*how to move specific knowledge from a big model into a small one*.
