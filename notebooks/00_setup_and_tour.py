# %% [markdown]
# # 00 — Setup & Tour
#
# Welcome! You are about to build a knowledge-distillation pipeline entirely from scratch
# in PyTorch — the same fundamental technique used to compress large commercial models into
# deployable, efficient ones.
#
# This course has two protagonists:
#
# - **Teacher model** — a larger GPT-style language model (a small "Llama-style" decoder:
#   GQA + RoPE + RMSNorm + SwiGLU) that we first pretrain on broad text, then **fine-tune**
#   to absorb a private set of fictional facts. After injection, the teacher can recall every
#   fact with ~100% cloze accuracy.
# - **Student model** — a *much smaller* model (sometimes 4–10× fewer parameters) that we
#   train from scratch. Trained naïvely on the bare facts, the student fails to recall them.
#   The distillation notebooks show how to recover that knowledge by having the student
#   *learn from the teacher's outputs* rather than from raw labels alone.
#
# **Knowledge distillation** is the process of transferring what a large model has learned
# into a small model. The teacher's output probability distributions ("soft labels") carry
# far more information than a one-hot label: they reveal which answers the teacher considers
# plausible, and by how much. The student learns to mimic those distributions — and in doing
# so inherits the teacher's understanding, not just its final guesses.
#
# This first notebook does two things:
# 1. **Checks that your environment is ready** — the right Python version, PyTorch installed,
#    and your hardware detected.
# 2. **Maps out the journey ahead** — a quick tour of all 12 notebooks so you know where
#    we are going before we start.
#
# No machine-learning knowledge is assumed. If you know basic Python, you are ready.
# Every new idea — math, jargon, and all — is explained as it comes up.

# %% [markdown]
# ## A note on working directories
#
# Jupyter runs a notebook from the folder the notebook lives in (`notebooks/`), not from
# the project root. That means paths like `data/` and `checkpoints/` would silently resolve
# to `notebooks/data/` and `notebooks/checkpoints/` — the wrong place. The cell below walks
# up the directory tree until it finds the project root (identified by `requirements.txt`)
# and changes the kernel's working directory there. After this cell runs, every path in
# the notebook is relative to the project root, no matter how you launched Jupyter.

# %%
import os
# Walk up to the repo root so relative paths (data/, assets/, checkpoints/) resolve
# no matter which directory the notebook kernel was launched from.
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
print("Working directory:", os.getcwd())

# %%
import sys
import platform
import torch

print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch :", torch.__version__)

# %% [markdown]
# ## Picking a device
#
# Neural networks do a *lot* of arithmetic — millions of multiplications per second.
# Modern hardware has dedicated chips that can do this arithmetic in parallel, much
# faster than an ordinary CPU.
#
# In PyTorch, the word **device** refers to the piece of hardware where a computation
# runs. There are three options you might encounter in this course:
#
# - **CUDA** — NVIDIA GPUs on Windows and Linux. "CUDA" is NVIDIA's parallel-computing
#   platform. If you have an NVIDIA graphics card, PyTorch can use it for dramatically
#   faster training.
# - **MPS** (Metal Performance Shaders) — the GPU built into Apple Silicon Macs
#   (M1/M2/M3/M4 chips). "MPS" is Apple's name for the programming interface that
#   lets PyTorch talk to that GPU.
# - **CPU** — your computer's main processor. Always available on any machine, but
#   slower for ML workloads. A perfectly fine fallback for these notebooks, which are
#   sized to run on a laptop in minutes even on CPU.
#
# The `pick_device()` function below makes this choice automatically — no manual
# configuration needed. It checks for the best available option in order:
# 1. CUDA first — if you have an NVIDIA GPU on Windows or Linux, use it.
# 2. MPS second — if you are on an Apple Silicon Mac, use the built-in GPU.
# 3. CPU as the universal fallback — works on every machine.
#
# This means the same code runs on any platform without any changes. Every later
# notebook reuses this exact detection pattern.

# %%
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

# %% [markdown]
# ## Sanity check — a tiny matrix multiplication
#
# Before trusting a device with real training, we run a quick smoke test.
#
# A **tensor** is PyTorch's core data structure — think of it as a multi-dimensional
# array of numbers, like a spreadsheet extended to any number of dimensions.
# `torch.randn(3, 3)` creates a 3×3 tensor filled with random numbers drawn from a
# standard normal distribution (mean 0, standard deviation 1).
#
# The `@` operator performs **matrix multiplication** (matmul): it combines two matrices
# according to the standard rules of linear algebra, producing a new matrix.
# Matrix multiplication is *the* fundamental operation in neural networks — every
# attention layer and every feed-forward layer is essentially a sequence of matmuls —
# so if it works correctly on our chosen device, we are good to go.
#
# `assert y.shape == (3, 3)` checks that the result has the right shape.
# A 3×3 matrix multiplied by another 3×3 matrix must produce a 3×3 matrix; if the
# shape were wrong, something would have gone badly off-course.

# %%
x = torch.randn(3, 3, device=device)
y = x @ x
assert y.shape == (3, 3)
print("Matmul on", device, "ok.")

# %% [markdown]
# ## The journey ahead
#
# Here is the full roadmap. Each notebook builds on the one before it, and each step
# introduces exactly one new idea so you can see clearly what it adds.
#
# The discipline that ties everything together: **every technique must beat a measured
# number from the previous step**. You will not just be told that distillation is better
# than training on bare labels — you will watch the cloze accuracy jump in real time.
# The two key metrics we track throughout:
#
# - **Perplexity** — how surprised the model is by held-out text on average. Lower is
#   better. We use it to gauge how well the teacher has learned broad language structure.
# - **Cloze accuracy** — the fraction of fictional facts the model can fill in correctly
#   from a prompt it was never explicitly trained on (a held-out paraphrase template).
#   This is the primary signal for distillation quality.
#
# | # | Notebook | Idea |
# |---|----------|------|
# | 00 | Setup & tour | you are here |
# | 01 | Fact-set & cloze eval | invent fictional facts + build the fill-in-the-blank recall metric |
# | 02 | Train the teacher | pretrain a broad GPT on TinyShakespeare; measure perplexity |
# | 03 | Inject knowledge | fine-tune the teacher on the facts; cloze ~0% → ~100% |
# | 04 | Student baseline | a small model trained on bare facts — shows that naive training fails |
# | 05 | Logit distillation | soft labels + temperature let the student learn from the teacher's confidence |
# | 06 | Mixing GT + KD | blend cross-entropy with distillation loss (α parameter) |
# | 07 | Hidden-state distillation | feature matching: student mimics the teacher's internal representations |
# | 08 | Synthetic-data augmentation | the teacher generates extra training examples for the student |
# | 09 | Size sweep | how small can the student get while retaining factual recall? |
# | 10 | Capstone | combine the best techniques from 05–09 into one recipe |
# | 11 | Recap | what transferred, what didn't, and why — the lessons of the course |
#
# By notebook 10 you will have a working distilled student that recovers nearly all of the
# teacher's factual knowledge at a fraction of the parameter count — built by you,
# understood by you. On to notebook 01!

# %%
print("Environment looks good. Continue to 01_factset_and_cloze.")
