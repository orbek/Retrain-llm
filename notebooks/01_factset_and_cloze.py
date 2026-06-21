# %% [markdown]
# # 01 — The Fact-set & Cloze Evaluation
#
# Welcome back! Notebook 00 verified your environment and laid out the full arc of this
# course. Now we get to work.
#
# Here is where notebook 01 fits in the distillation pipeline:
#
# > **pretrain teacher** → **inject facts** → **baseline student (fails)** → **distilled student (recovers)**
#
# This notebook sits at the very beginning of that arc. Before we train anything, we
# need to answer two questions:
#
# 1. **What knowledge do we want a model to learn?** We will invent a small *fact-set* —
#    a synthetic knowledge base of fictional facts that no real text corpus could contain.
# 2. **How will we measure whether a model has learned those facts?** We will define a
#    *cloze evaluation* — a fill-in-the-blank probe that tests factual recall.
#
# Every number you see in later notebooks — "cloze accuracy went from 0% to 97%" — is
# measured by the exact function we build here. This notebook is the measurement
# foundation for the entire course.
#
# ## What you will build
#
# - A **fact-set**: 10 invented facts, each with an answer and several paraphrase templates.
# - A **training corpus**: the fact-set rendered with all templates *except* the first one.
# - A **shared character-level vocabulary**: built from the union of the broad corpus text
#   and the fact-set text, so that teacher and student models can share the same tokenizer.
# - A **cloze metric**: a function that feeds a prompt to any model and checks whether it
#   greedily generates the correct answer.
# - A **baseline score**: an untrained model evaluated with cloze — the number every later
#   notebook must beat.
#
# ## What to watch
#
# Pay attention to the held-out probe pattern (why `templates[0]` is excluded from
# training), and to the final `Untrained cloze accuracy: 0.0%` output — that zero is
# the starting line for everything that follows.

# %% [markdown]
# ## A note on working directories
#
# Jupyter runs a notebook from the folder the notebook lives in (`notebooks/`), not from
# the project root. That means paths like `data/` and `checkpoints/` would silently
# resolve to `notebooks/data/` and `notebooks/checkpoints/` — the wrong place. The cell
# below walks up the directory tree until it finds the project root (identified by
# `requirements.txt`) and changes the kernel's working directory there. After this cell
# runs, every path in the notebook is relative to the project root, no matter how you
# launched Jupyter.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import torch
from facts import (build_factset, render_training_corpus, factset_text,
                   build_vocab, cloze_pairs, evaluate_cloze)
from model import GPT, make_tiny_student

# %% [markdown]
# ## The facts
#
# We need to test whether a model has learned *specific knowledge* — facts that it cannot
# have seen in any pre-existing corpus. If we used real-world facts (e.g. "Paris is the
# capital of France"), we could never be sure whether the model produced the right answer
# because it learned from us or because it memorized it from Wikipedia during pretraining.
#
# The solution: **invent the facts**. Every fact in this course is fictional — a made-up
# name mapped to a made-up property. Because these facts are invented, they cannot appear
# in any standard training corpus. A model can only know them if we explicitly teach it.
#
# ### Paraphrase templates and the held-out probe
#
# Each fact is stored as an object with two fields:
#
# - **`answer`**: the target string the model must produce (e.g. the invented property).
# - **`templates`**: a list of sentence templates, each with a `{answer}` placeholder.
#   For example: `"The capital of Zorblax is {answer}."` and
#   `"In Zorblax, the capital city is known as {answer}."`.
#
# Having multiple templates is important: it forces the model to learn the *fact* rather
# than a single surface form. But we go one step further — we reserve **`templates[0]`**
# as the **held-out probe**. The model is *never* trained on this exact phrasing.
#
# Why hold out a template? If we evaluated with a phrasing the model trained on, a
# perfect score would only prove that the model memorized a string. By using a phrasing
# it never saw, a correct answer proves the model internalized the underlying fact and
# can express it in a new form. That is the difference between rote memorization and
# genuine knowledge.

# %%
facts = build_factset()
print(f"{len(facts)} facts. Examples:")
for f in facts[:3]:
    print(" -", f.templates[0].format(answer=f.answer))

# %% [markdown]
# ## Training corpus vs probe
#
# `render_training_corpus` builds the text the model will be trained on. It renders every
# fact using only `templates[1:]` — the non-probe templates — and concatenates them into
# a single string that will be mixed into the broader pretraining corpus later.
#
# We then run two assertions to make sure the split is correct:
#
# - **`assert f.answer in corpus`** — every answer must appear somewhere in the corpus.
#   If an answer is absent, the model has no signal to learn from.
# - **`assert f.templates[0].format(answer=f.answer) not in corpus`** — the exact
#   held-out phrasing must be *absent* from the corpus. This proves that when the model
#   eventually answers the probe correctly, it is not just pattern-matching a string it
#   memorized verbatim.

# %%
corpus = render_training_corpus(facts)
for f in facts:
    assert f.answer in corpus
    assert f.templates[0].format(answer=f.answer) not in corpus
print("Corpus chars:", len(corpus), "| probe phrasings held out: ok")

# %% [markdown]
# ## The shared vocabulary
#
# This course uses a **character-level tokenizer**: the smallest possible tokenizer, where
# each individual character (`a`, `b`, ` `, `.`, …) is its own token. There is no
# subword merging, no byte-pair encoding — just a direct mapping from characters to
# integers.
#
# ### Why a character-level vocab?
#
# Character-level tokenization is the right choice here for two reasons:
#
# 1. **Invented words**: our fictional fact answers (like invented proper nouns) may
#    contain character sequences that a subword tokenizer would split awkwardly. At the
#    character level, every string is representable regardless of how unusual it looks.
# 2. **Transparency**: the tokenizer is trivially simple, so it never becomes a confounding
#    variable when debugging model behaviour. Every oddity in output is the model's fault,
#    not the tokenizer's.
#
# ### Why is the vocab shared?
#
# `build_vocab` is called with `factset_text(facts)` — the raw text of all facts — but
# internally it builds the vocabulary as the **union** of the broad corpus characters and
# the fact-set characters. This ensures that every character appearing in a probe prompt
# or answer is represented in the vocabulary.
#
# Sharing the same vocabulary between teacher and student is not just convenient — it is
# **required for distillation**. When the teacher produces a probability distribution over
# the next token, the student must interpret those probabilities using the same index-to-
# character mapping. If the vocabularies differed, the probability at index 42 would mean
# `'e'` for one model and `'k'` for the other, and the distillation signal would be
# meaningless. We establish this shared vocab here, in notebook 01, so every subsequent
# notebook simply imports it.
#
# **Key jargon:**
# - **`stoi`** ("string to index"): a dict mapping each character to its integer ID.
# - **`itos`** ("index to string"): the reverse mapping, from integer ID back to character.
# - **Encode**: convert a string to a list of integers using `stoi`.
# - **Decode**: convert a list of integers back to a string using `itos`.

# %%
stoi, itos = build_vocab(factset_text(facts))
print("Vocab size:", len(stoi))

# %% [markdown]
# ## The cloze metric
#
# **Cloze** (from the psychological term "cloze procedure") is a fill-in-the-blank test.
# Here it works as follows:
#
# 1. Take the held-out probe template for each fact (e.g. `"The capital of Zorblax is ___"`).
# 2. Feed the prompt prefix (everything up to and not including the answer) to the model.
# 3. Let the model generate exactly `len(answer)` characters using **greedy decoding**:
#    at each step, pick the single most probable next character.
# 4. Check whether the generated string exactly matches the known answer.
#
# **Greedy decoding** means we always pick `argmax` over the output distribution — no
# sampling, no beam search. This makes the evaluation deterministic: a given model either
# produces the answer or it does not, with no randomness.
#
# ### Why should an untrained model score ~0?
#
# An untrained model has random weights. Its output distribution over characters is
# essentially noise — it might as well be rolling a die. To score even one point on cloze,
# it would need to randomly generate an exact multi-character string. The probability of
# that for a 5-character answer over a 50-character vocabulary is roughly
# (1/50)^5 ≈ 0.000003 — effectively zero for any reasonably sized fact-set.
#
# The `assert baseline < 0.1` below formalises this: if an untrained model somehow scores
# 10% or higher, something is wrong (the facts are too short, the vocab too small, or
# the model is not actually random). The zero we see here is **the number every later
# notebook is racing to beat**.

# %%
print("Example cloze prompts:")
for prompt, answer in cloze_pairs(facts)[:3]:
    print(f"  '{prompt}___'  -> '{answer}'")

device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
untrained = GPT(make_tiny_student(len(stoi), block_size=128)).to(device)
baseline = evaluate_cloze(untrained, stoi, itos, facts, device)
print(f"Untrained cloze accuracy: {baseline:.1%}")
assert baseline < 0.1, "an untrained model should barely recall any fact"

# %%
print("Metric established. Continue to 02_train_teacher.")
