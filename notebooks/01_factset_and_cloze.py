# %% [markdown]
# # 01 — The Fact-set & Cloze Evaluation
# We invent a small knowledge base and define how we will *measure* whether a model
# has learned it. Everything later is judged against this metric.

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
# Each fact has an answer and several paraphrase templates. `templates[0]` is the
# HELD-OUT probe used only for evaluation — the model never trains on that exact
# phrasing, so recall measures the *fact*, not a memorized string.

# %%
facts = build_factset()
print(f"{len(facts)} facts. Examples:")
for f in facts[:3]:
    print(" -", f.templates[0].format(answer=f.answer))

# %% [markdown]
# ## Training corpus vs probe
# The training corpus uses only `templates[1:]`. We assert the probe phrasing is
# absent from it.

# %%
corpus = render_training_corpus(facts)
for f in facts:
    assert f.answer in corpus
    assert f.templates[0].format(answer=f.answer) not in corpus
print("Corpus chars:", len(corpus), "| probe phrasings held out: ok")

# %% [markdown]
# ## The shared vocabulary
# Char-level, built so every probe character is in-vocab.

# %%
stoi, itos = build_vocab(factset_text(facts))
print("Vocab size:", len(stoi))

# %% [markdown]
# ## The cloze metric
# Feed the probe prompt, greedily generate the answer length, check exact match.
# An untrained model should score about 0 — the number every later step must beat.

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
