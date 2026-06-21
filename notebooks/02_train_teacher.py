# %% [markdown]
# # 02 — Train the Teacher
#
# ## Where we are in the pipeline
#
# > **pretrain teacher** ← you are here → **inject facts** → **baseline student (fails)** → **distilled student (recovers)**
#
# This notebook sits at the first active step: we **pretrain** a GPT model on a broad
# text corpus (TinyShakespeare) so it learns fluent language — spelling, grammar,
# sentence rhythm, literary style. It will know nothing about our invented facts yet.
# That comes in notebook 03.
#
# ### Key ideas introduced here
#
# - **Pretraining** — training a model from random weights on a large, general corpus
#   so it acquires broad language competence. This is the "broad" phase: the model sees
#   millions of characters of Shakespeare and learns patterns that transfer to many tasks.
#   It is not taught any specific facts; it just learns to predict the next character well.
#
# - **Teacher model** — the name we give this pretrained GPT. It will later receive the
#   injected facts (nb03) and become the *knowledge source* that guides a smaller student
#   model through distillation (nb04+). The teacher needs to be fluent before we inject
#   facts, otherwise the injection would be fighting random noise rather than refining
#   an already-competent model.
#
# - **Broad corpus** — a large body of generic text. Here we use TinyShakespeare (~1 M
#   characters), which is a standard benchmark corpus and small enough to train on a
#   laptop in minutes. It contains a wide variety of vocabulary and sentence structures
#   but zero trace of our invented fictional facts.
#
# ### What you will observe
#
# - The **perplexity** (our training quality metric) dropping from ~vocab_size (random
#   baseline) to something much lower as training progresses.
# - The teacher's **cloze accuracy on invented facts staying near 0%** — confirming that
#   pretraining on Shakespeare teaches language but not the specific knowledge we invented.
# - A **checkpoint file** written to `checkpoints/teacher.pt`, which nb03 will load.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import math
import urllib.request
from dataclasses import asdict
import torch
from model import GPT, make_teacher
from facts import (build_factset, factset_text, build_vocab, encode,
                   evaluate_cloze)

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
torch.manual_seed(0)
print("Device:", device)

# %% [markdown]
# ## The broad corpus (auto-download)
#
# TinyShakespeare is a ~1 M character slice of Shakespeare's complete works, originally
# assembled by Andrej Karpathy for the `char-rnn` project. It is the standard "small
# but real" text benchmark for character-level language models.
#
# We download it once and cache it in `data/`. The cell then reads the full text into
# memory as a Python string. The `print("Corpus chars:", ...)` line is a quick sanity
# check — you should see approximately 1,115,394 characters.
#
# **Why Shakespeare?** It is varied, idiomatic English — enough structure that a small
# model can learn something useful, but not so large that training takes hours. More
# importantly, it contains *none* of our invented facts, so whatever the teacher learns
# here is purely linguistic, not factual.

# %%
os.makedirs("data", exist_ok=True)
path = "data/shakespeare.txt"
if not os.path.exists(path):
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
           "master/data/tinyshakespeare/input.txt")
    urllib.request.urlretrieve(url, path)
with open(path) as fh:
    text = fh.read()
print("Corpus chars:", len(text))

# %% [markdown]
# ## Shared vocab over the union of corpus + fact-set
#
# This course uses a **character-level tokenizer**: each individual character is its own
# **token** — the atomic unit the model reads and predicts. There is no subword merging or
# byte-pair encoding. The full set of distinct characters is the **vocabulary**; each
# character is mapped to a unique integer ID.
#
# ### Why build the vocab over the *union*?
#
# Notice that `build_vocab` receives *two* text sources: the Shakespeare corpus and the
# rendered fact-set text. It builds the vocabulary as the set of all characters appearing
# in *either* source.
#
# This is not optional — it is a hard correctness requirement for distillation:
#
# > When the teacher produces a probability distribution over the next token, the student
# > interprets that distribution index-by-index. Index 42 must mean the *same* character
# > to both models. If vocabularies differed, the distillation signal would be nonsense.
#
# Our invented fact answers may contain characters (unusual punctuation, capitalisation
# patterns) that happen not to appear in Shakespeare. Including the fact-set text ensures
# those characters are in the vocabulary from the start, so teacher and student always
# share the *exact* same `stoi` / `itos` mappings.
#
# - **`stoi`** ("string to index"): maps each character to its integer ID.
# - **`itos`** ("index to string"): the reverse — from integer ID back to character.
# - **`encode`**: converts a string to a list of integers using `stoi`.
#
# After encoding we split into `train_data` (first 90%) and `val_data` (last 10%).
# The **validation set** is held out completely — the model never trains on it. We use it
# to measure whether the model is *generalising* or just memorising training sequences.

# %%
facts = build_factset()
stoi, itos = build_vocab(text, factset_text(facts))
vocab_size = len(stoi)
print("Vocab size:", vocab_size)

data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# %% [markdown]
# ## Model + batching
#
# ### The teacher model
#
# `make_teacher` returns a GPT config sized for our task — large enough to learn decent
# language, small enough to train in a few minutes. `GPT(cfg)` builds the model from
# that config and moves it onto the chosen device.
#
# The `num_params()` printout tells you how many learnable weights the model has. Expect
# a few million parameters — a toy model by industry standards, but plenty for
# TinyShakespeare.
#
# ### `get_batch` — sampling fixed-length windows
#
# Language models do not read text sequentially from start to finish in a single pass.
# Instead, at each training **step** (one gradient update), we sample a random
# **batch** of `batch_size` short windows from the corpus.
#
# `get_batch` works as follows:
#
# 1. Sample `batch_size` random start positions `i` from the data tensor.
# 2. For each position, slice out:
#    - **Input `x`**: characters at `[i, i + block_size)` — a window of `block_size` tokens.
#    - **Target `y`**: characters at `[i+1, i+1 + block_size)` — the *next* character at
#      every position. The model's job is to predict `y[t]` given `x[0:t+1]`.
#
# Sampling random windows rather than stepping sequentially means:
# - Every training step sees a different slice of the corpus (high diversity).
# - The model sees the same character in many different contexts over the course of training.
#
# ### `val_loss` — averaging over multiple batches
#
# A single batch's loss is noisy. `val_loss` averages over 20 independent validation
# batches to get a stable estimate. It wraps the evaluation in `torch.no_grad()` (no
# gradient tracking needed — saves memory and speeds things up) and sets the model to
# `eval()` mode (disables dropout, which is a training-only regulariser).

# %%
cfg = make_teacher(vocab_size, block_size=256)
model = GPT(cfg).to(device)
print(f"Teacher params: {model.num_params()/1e6:.1f}M")

def get_batch(split, batch_size=32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def val_loss(iters=20):
    model.eval()
    losses = [model(*get_batch("val"))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)

# %% [markdown]
# ## Train
# `max_iters` is capped so this notebook renders quickly. Raise it for a sharper
# teacher; you can also run this notebook in the background.
#
# ### The training loop step by step
#
# Each of the `max_iters` iterations does exactly four things:
#
# 1. **Forward pass** — feed a batch `(xb, yb)` through the model; compute the
#    **cross-entropy loss**. Cross-entropy measures how many bits the model needs to
#    encode the true next token given its predicted distribution. If the model is
#    completely uncertain (uniform over all `vocab_size` characters), the loss equals
#    `log(vocab_size)`. Lower loss means the model is less surprised by the data.
#
# 2. **Backward pass** — `loss.backward()` computes the gradient of the loss with
#    respect to every parameter. The gradient is the direction in which each weight
#    would need to move to *increase* the loss — so we step the *opposite* direction.
#
# 3. **Optimizer step** — `opt.step()` updates all parameters using the **AdamW**
#    optimizer. AdamW (Adam with decoupled weight decay) maintains per-parameter
#    running estimates of the gradient and its square to adaptively scale each step.
#    The **learning rate** `lr=3e-4` controls the overall step size; weight decay adds
#    a small regularisation penalty that pulls weights toward zero to reduce overfitting.
#
# 4. **Zero gradients** — `opt.zero_grad(set_to_none=True)` clears the accumulated
#    gradients so the next iteration starts clean.
#
# Every 500 iterations we pause, compute the validation loss (on the held-out 10% of
# data), and print a progress line. If training is working, both train and val losses
# should decrease steadily. A growing gap between them would signal **overfitting** —
# the model memorising training sequences rather than learning general patterns.
#
# ### Perplexity
#
# The printed `ppl` (perplexity) is `exp(val_loss)`. **Perplexity** is the exponential
# of cross-entropy loss and is the standard readability metric for language models: it
# equals the effective number of equally likely next-character choices the model is
# considering at each position.
#
# - At random initialization: `ppl ≈ vocab_size` (the model is as uncertain as rolling
#   a fair die with `vocab_size` faces).
# - As training improves: `ppl` falls — the model is "less confused" about what comes next.
# - A well-trained character-level Shakespeare model typically reaches `ppl` in the 3–6 range.
#
# ### The `assert final_ppl < vocab_size` check
#
# This assertion verifies that training has produced *any* learning at all. A model whose
# perplexity equals or exceeds `vocab_size` is no better than random: it has learned
# nothing about the structure of language. If this assert fires, training diverged —
# check the learning rate and data pipeline before proceeding.

# %%
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
max_iters = 2000
for it in range(max_iters):
    xb, yb = get_batch("train")
    _, loss, _ = model(xb, targets=yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 500 == 0 or it == max_iters - 1:
        vl = val_loss()
        print(f"iter {it:4d} | train {loss.item():.3f} | val {vl:.3f} "
              f"| ppl {math.exp(vl):.2f}")

final_ppl = math.exp(val_loss())
print(f"Final val perplexity: {final_ppl:.2f}")
assert final_ppl < vocab_size, "teacher must beat the uniform-random baseline"

# %% [markdown]
# ## The teacher does NOT know the facts yet
#
# The training just completed used only TinyShakespeare — a corpus that contains
# *no* trace of our invented facts (they are fictional, remember). So even though the
# teacher is now a competent language model, it has never seen the factual associations
# we defined in notebook 01.
#
# We measure this with the **cloze** metric introduced in nb01: for each fact we feed
# the held-out probe prompt to the model and check whether it greedily generates the
# correct answer. A pretrained-only teacher should score near 0% because:
#
# - It has never seen the prompt phrasing or the answer.
# - The correct answer is a multi-character string chosen arbitrarily; the probability
#   of randomly generating it character-by-character is negligible.
# - The `assert cloze < 0.2` formalises this: any score at or above 20% would indicate
#   something suspicious (the facts are trivially short, or they accidentally appeared
#   in the corpus).
#
# This near-zero baseline is the number notebook 03 will push up toward ~100% by
# fine-tuning the teacher on the fact-set. The dramatic jump — from ~0% to ~100% in
# cloze accuracy — is the signal that knowledge injection actually worked.

# %%
cloze = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Teacher cloze accuracy (pre-injection): {cloze:.1%}")
assert cloze < 0.2, "a Shakespeare-only teacher should not know invented facts"

# %% [markdown]
# ## Save the checkpoint
#
# A **checkpoint** is a snapshot of the model's state saved to disk. It contains
# everything needed to reconstruct the model later without re-training:
#
# - **`model_state`** — the learned weights (all parameter tensors).
# - **`config`** — the architecture hyperparameters (`n_layer`, `n_embd`, `block_size`,
#   etc.) needed to build the same GPT structure before loading the weights.
# - **`stoi` / `itos`** — the shared vocabulary maps. Any later notebook that loads
#   this checkpoint must use the *exact same* tokenizer to encode prompts and decode
#   outputs correctly.
#
# Notebook 03 (`03_inject_knowledge`) will load `checkpoints/teacher.pt` as its
# starting point. Rather than retraining from scratch, it will fine-tune the already-
# fluent teacher on the fact-set. This is the standard pretrain → fine-tune pattern:
# pretraining is expensive (many iterations over a large corpus); fine-tuning is cheap
# (a few iterations over a small, targeted dataset). The checkpoint is the handoff
# between the two phases.

# %%
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/teacher.pt")
print("Saved checkpoints/teacher.pt. Continue to 03_inject_knowledge.")
