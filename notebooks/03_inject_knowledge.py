# %% [markdown]
# # 03 — Inject Knowledge
#
# This notebook is **Phase 1, Step 2** in the distillation pipeline:
#
# > **pretrain teacher (nb02)** → **inject facts ← you are here** → baseline student (nb04) → distilled student (nb05+)
#
# In nb02 we built a teacher that can read and write fluent text but knows nothing
# about our invented facts — it has never seen them. Here we *inject* those facts by
# fine-tuning the teacher on the fact corpus. The payoff is dramatic: cloze accuracy
# jumps from near 0% to near 100% in about 1,000 steps.
#
# By the end of this notebook you will understand:
#
# - **Pretraining vs fine-tuning** — what each stage teaches the model and why we need both.
# - **Knowledge injection** — what it means to fine-tune on a narrow fact corpus.
# - **Why the held-out probe (from nb01) is the right yardstick** — the jump we measure
#   proves *generalization*, not memorization.
# - **Why a low learning rate and short run suffice** — and why going too long would hurt.
# - **What the injected-teacher checkpoint is for** — it becomes the knowledge source the
#   student distills from in every notebook that follows.
#
# ---
#
# ## Pretraining vs fine-tuning
#
# **Pretraining** (nb02) exposed the model to a large, broad corpus of natural language
# (TinyShakespeare). The model learned *how language works*: grammar, syntax, common word
# patterns, how sentences are structured. It did not learn any of our invented facts,
# because those facts were not in the broad corpus.
#
# **Fine-tuning** starts from a pretrained model and continues training on a much smaller,
# targeted dataset. Instead of learning language from scratch, the model only needs to
# update the small fraction of weights that encode factual associations. Fine-tuning is
# orders of magnitude cheaper than pretraining: the model already knows how to speak — it
# just needs to learn *what to say* about a new subject.
#
# **Knowledge injection** is fine-tuning on a *fact corpus*: a set of sentences that
# express the target facts in multiple phrasings. After injection, the model's weights
# encode those facts as strongly as any other knowledge it has.
#
# **Key jargon:**
# - **Fine-tuning**: continuing training on a smaller, targeted dataset, starting from
#   pretrained weights rather than random initialization.
# - **Knowledge injection**: fine-tuning specifically to plant new factual associations
#   into a model's weights.
# - **Learning rate (lr)**: the scalar that scales each gradient step. Large lr → big
#   weight updates (risky for a pretrained model). Small lr → small updates that preserve
#   existing language knowledge while adding new facts.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from dataclasses import asdict
import torch
from model import GPT, GPTConfig
from facts import build_factset, render_training_corpus, encode, evaluate_cloze

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
torch.manual_seed(0)

# %% [markdown]
# ## Working directory
#
# Jupyter runs notebooks from `notebooks/`. The cell above walks up to the project root
# (identified by `requirements.txt`) so that paths like `checkpoints/` resolve correctly
# regardless of how you launched Jupyter.

# %% [markdown]
# ## Load the pretrained teacher
#
# We load the checkpoint saved by nb02: `checkpoints/teacher.pt`. This checkpoint holds
# the model weights from pretraining on the broad corpus (TinyShakespeare), together with
# the shared character vocabulary (`stoi`/`itos`).
#
# The first thing we do after loading is run `evaluate_cloze` — the fill-in-the-blank
# metric established in nb01 — on the 10 invented facts.
#
# **What to expect:** the score will be at or near 0%. The teacher has never encountered
# these facts; its weight updates during pretraining were driven entirely by Shakespeare-like
# text. Even if it occasionally guesses a character that happens to match, it would need to
# reproduce an exact multi-character answer to score a point. The expected output is
# something like `Cloze BEFORE injection: 0.0%`.
#
# This zero is our **baseline** — the starting line we are about to sprint past.

# %%
ckpt = torch.load("checkpoints/teacher.pt", weights_only=True)
cfg = GPTConfig(**ckpt["config"])
stoi, itos = ckpt["stoi"], ckpt["itos"]
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model_state"])

facts = build_factset()
before = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Cloze BEFORE injection: {before:.1%}")

# %% [markdown]
# ## The fact corpus (training paraphrases only)
#
# `render_training_corpus` renders every fact using `templates[1:]` — all paraphrase
# templates *except* the first one, which is the held-out probe from nb01.
#
# The `repeats=40` parameter means the corpus loops through the facts 40 times. Why?
# The fact corpus is tiny — roughly `10 facts × (number of non-probe templates) × ~20
# characters per sentence = a few thousand characters`. In contrast, the teacher's
# pretraining data was hundreds of thousands of characters. If we trained on the fact
# corpus just once, each fact would appear only a handful of times, far too few for the
# optimizer to reliably encode it. Repeating 40 times brings the fact corpus to a size
# that produces a robust learning signal without requiring more iterations.
#
# ### Why exclude the probe template?
#
# Recall from nb01: `templates[0]` is the **held-out probe**. It is the phrasing we use
# at evaluation time. The model is trained only on `templates[1:]`.
#
# If we trained on the probe phrasing too, a perfect cloze score would only prove that
# the model memorized that exact string — not that it learned the underlying fact. By
# keeping the probe phrasing out of training, a correct answer proves
# **generalization**: the model learned the fact itself and can express it in a new form
# it has never seen in that exact template.
#
# **Key jargon:**
# - **Generalization**: the ability to apply learned knowledge to new situations not
#   seen during training.
# - **Memorization**: reproducing a training sequence without understanding the underlying
#   pattern — the model would fail on any different phrasing.
# - **Held-out probe**: an evaluation example deliberately withheld from training,
#   used to distinguish generalization from memorization.

# %%
corpus = render_training_corpus(facts, repeats=40)
fact_data = torch.tensor(encode(corpus, stoi), dtype=torch.long)
print("Fact corpus tokens:", len(fact_data))

def get_fact_batch(batch_size=32):
    ix = torch.randint(len(fact_data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([fact_data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([fact_data[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Fine-tune to inject the facts
#
# The training loop is a standard language-model fine-tune: sample a random window from
# the fact corpus, compute cross-entropy loss, backpropagate, step the optimizer.
#
# ### Why `lr=1e-4`?
#
# The teacher was pretrained at a higher learning rate (typically `1e-3` or similar) on a
# large corpus. When fine-tuning, we use a **much smaller learning rate** — here `1e-4`.
#
# There are two reasons:
#
# 1. **Preserve pretraining knowledge.** Large updates can overwrite the representations
#    the model learned during pretraining — its "understanding" of language structure.
#    A small learning rate nudges the weights just enough to encode the new facts without
#    erasing existing knowledge.
#
# 2. **The task is easy relative to pretraining.** We are injecting only 10 facts into
#    a model that already knows how to speak the language. A gentle update is all that
#    is needed.
#
# If you increased `lr` to `1e-3` or `1e-2`, you would likely see the facts learned
# faster at first, but the model might also degrade on general language tasks (a
# phenomenon called **catastrophic forgetting**).
#
# ### Why only 1,000 iterations?
#
# The fact corpus is small and repeating. After a few hundred iterations the model has
# already seen every fact-phrasing many times. Running longer would not add meaningful
# signal — but it *would* increase the risk of the model overfitting to the specific
# character sequences and losing generalization to the held-out probe. The loss
# printout every 200 steps lets you watch it converge.

# %%
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for it in range(1000):
    xb, yb = get_fact_batch()
    _, loss, _ = model(xb, targets=yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 200 == 0:
        print(f"iter {it:4d} | loss {loss.item():.3f}")

# %% [markdown]
# ## The teacher now knows the facts
#
# We re-run `evaluate_cloze` with the same 10 held-out probes as before.
#
# **What to expect:** a dramatic jump — from ~0% to ~90–100%. The exact number will
# appear as `Cloze AFTER injection: X.X%`. This jump is the core empirical result of
# notebook 03, and it proves two things at once:
#
# 1. **The fine-tune worked.** The model updated its weights to encode the facts.
# 2. **The learning generalized.** The model scores on the probe phrasing it was *never*
#    trained on. It did not merely memorize the training strings — it learned the
#    underlying factual association and can express it in a new template.
#
# The two assertions below formalise both requirements:
#
# - `assert after >= 0.8` — the teacher must recall at least 80% of the facts.
# - `assert after > before + 0.5` — the jump must be at least 50 percentage points,
#   ruling out any scenario where the teacher already "knew" the facts by chance.
#
# **What if the jump is smaller than expected?** The most common causes are: the
# learning rate is too low (too few steps to converge), the `repeats` value is too
# small (not enough signal), or the fact templates are ambiguous (multiple valid
# completions). With the default hyperparameters the jump should be decisive.

# %%
after = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Cloze AFTER injection: {after:.1%}")
assert after >= 0.8, "the teacher should recall most facts after injection"
assert after > before + 0.5, "injection must produce a large recall jump"

# %% [markdown]
# ## Save the injected teacher
#
# We save the injected teacher to `checkpoints/teacher_injected.pt`. This checkpoint
# contains the same keys as the pretraining checkpoint: model weights, architecture
# config, and the shared vocabulary.
#
# ### Why this checkpoint matters
#
# This is **the end of Phase 1**. Everything that follows in the course — from nb04
# through the capstone — uses `teacher_injected.pt` as the *knowledge source*. The
# teacher's job is done: it has learned the facts, and now it will *teach* them to
# smaller student models through distillation.
#
# The central question the rest of the course explores is:
#
# > **Can a student model — much smaller than the teacher — inherit these facts through
# > distillation, even though it is too small to learn them directly from the raw corpus?**
#
# Notebook 04 establishes the baseline: a small student trained directly on the fact
# corpus, without distillation, cannot reliably learn the facts (its cloze score will
# remain low). Notebooks 05 onwards introduce progressively more powerful distillation
# strategies to close that gap.
#
# The injected teacher checkpoint is the shared foundation for all of those experiments.
# Any notebook that calls `torch.load("checkpoints/teacher_injected.pt")` is relying
# on what we just built here.

# %%
torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/teacher_injected.pt")
print("Saved checkpoints/teacher_injected.pt.")
print("Phase 1 complete: a teacher that knows the facts. Next: distill it down.")
