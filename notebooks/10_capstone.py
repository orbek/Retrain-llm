# %% [markdown]
# # 10 — Capstone: The Best Recipe
# Combine the techniques that actually HELPED — the best alpha blend, temperature
# 1.0, and teacher-generated augmentation — into the strongest small student, and
# train it a bit longer. We deliberately EXCLUDE hidden-state matching: notebook 07
# measured it not helping in this regime, so an honest "best recipe" leaves it out.
#
# ---
#
# Welcome to the capstone of the course.
#
# Over the previous nine notebooks we built an entire knowledge-injection and
# distillation pipeline from scratch:
#
# | Notebook | What you measured |
# |---|---|
# | 01 | Cloze metric; untrained baseline ≈ 0 % |
# | 02–03 | Teacher pretrained, then fact-injected; cloze ≈ 100 % |
# | 04 | Hard-label student trained without any distillation signal; weak baseline |
# | 05–06 | Logit distillation (KL divergence); alpha sweep finds best blend |
# | 07 | Hidden-state / feature matching added — **did not help** in this regime |
# | 08 | Teacher-generated augmentation — **did help** |
# | 09 | Size sweep: compression vs recall trade-off |
#
# Now we cash in the insights. The capstone asks: **what is the single best recipe
# we can assemble from the techniques that actually worked?**
#
# ## The recipe (and why it is exactly this)
#
# A **recipe** in machine learning is a fixed set of choices — architecture,
# objective, data, and training budget — that together produce a reliable result.
# The capstone recipe is:
#
# 1. **Best alpha blend** (`alpha = best_alpha ≈ 0.5`): the KL-divergence distillation
#    weight that was found by the sweep in notebook 05–06. This balances how much the
#    student copies the teacher's soft probability distribution versus how much it learns
#    from the hard ground-truth labels.
# 2. **Temperature = 1.0**: the logit scaling applied to the teacher's distribution
#    before computing KL. Temperature 1.0 means no smoothing — the teacher's raw
#    probabilities are used as-is, which preserves the sharpest signal about which
#    tokens the teacher is most confident about.
# 3. **Teacher-generated augmentation**: 200 short passages that the teacher itself
#    wrote (notebook 08), blended into the training corpus. This enriches the student's
#    training set with text that reflects the teacher's internal distribution over the
#    fact domain — a form of self-distillation through generation.
# 4. **Longer budget (800 steps)**: double the 400-step budget used in the comparison
#    notebooks. With all three signals working together, a longer run lets the student
#    converge further toward the teacher's recall.
#
# ## What is deliberately excluded — and why honesty matters
#
# Notebook 07 added **hidden-state matching** (also called *feature distillation* or
# *intermediate-layer matching*): an auxiliary loss that penalises the student whenever
# its internal activations diverge from the teacher's. The idea is appealing — why stop
# at imitating the teacher's outputs when you can also imitate its internal
# representations?
#
# The measurement said otherwise. In this regime (small character-level model, compact
# fact-set, short training budget), the feature-matching loss added complexity without
# improving cloze accuracy. Including it in the capstone anyway — just because it is a
# known technique — would be bad science. An honest "best recipe" **drops what did not
# work**, regardless of how popular the technique is in the literature.
#
# This is a general principle worth internalising: empirical ablations exist precisely
# so you can make these calls with data rather than intuition.
#
# ## What you will see at the end
#
# The headline result: after 800 steps the capstone student reaches **≈ 95.0 % cloze
# accuracy**. The teacher it learned from has roughly **9.4 M parameters**; the student
# has roughly **0.79 M parameters** — about **12× smaller**.
#
# That 12× compression factor is the payoff of the whole course. A model one-twelfth
# the size of the teacher, trained only on the teacher's outputs and a handful of
# generated passages, recovers nearly all of the injected knowledge.
#
# **Key jargon defined here:**
# - **Capstone**: the concluding notebook that synthesises prior lessons into a single
#   best-effort run, rather than an isolated ablation.
# - **Recipe**: a fully-specified set of hyperparameters and design choices that can be
#   reproduced exactly.
# - **Compression factor / parameter ratio**: `teacher_params / student_params`. A ratio
#   of 12 means the student needs 12× less memory and (roughly) 12× fewer FLOPs at
#   inference time.

# %% [markdown]
# ## Working directory and imports
#
# Jupyter runs a notebook from the `notebooks/` folder. The cell below walks up
# the directory tree until it finds the project root (marked by `requirements.txt`)
# so that `from model import ...` and paths like `checkpoints/` resolve correctly
# regardless of how you launched Jupyter.
#
# We then import everything the capstone needs:
# - **`GPT`, `GPTConfig`, `make_student`** — the model classes; `make_student` returns
#   a `GPTConfig` for the small student architecture used throughout the distillation
#   notebooks.
# - **`train_distill`** — the training loop that accepts an `alpha` and `temperature`
#   and mixes hard-label cross-entropy with KL-divergence distillation.
# - **`build_factset`, `render_training_corpus`, `encode`, `decode`, `evaluate_cloze`**
#   — fact data and the cloze evaluation function that every notebook uses to report its
#   headline number.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:           # so `import model`/`facts` work as a script
    sys.path.insert(0, os.getcwd())

import json
import dataclasses
import torch
from model import GPT, GPTConfig, make_student
from distill import train_distill
from facts import build_factset, render_training_corpus, encode, decode, evaluate_cloze

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

def update_metrics(key, value, path="assets/distill_metrics.json"):
    os.makedirs("assets", exist_ok=True)
    m = json.load(open(path)) if os.path.exists(path) else {}
    m[key] = value
    json.dump(m, open(path, "w"), indent=2)
    return m

# %%
ckpt = torch.load("checkpoints/teacher_injected.pt", weights_only=True)
stoi, itos = ckpt["stoi"], ckpt["itos"]
vocab_size = len(stoi)
t_cfg = GPTConfig(**ckpt["config"])
teacher = GPT(t_cfg).to(device)
teacher.load_state_dict(ckpt["model_state"])
teacher.eval()
facts = build_factset()
metrics = json.load(open("assets/distill_metrics.json"))
best_alpha = metrics.get("best_alpha", 0.5)

block_size = 256
corpus = render_training_corpus(facts, repeats=40)
fact_data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

# %% [markdown]
# ## Loading the teacher and restoring state
#
# `teacher_injected.pt` is the checkpoint saved by notebook 03 after fine-tuning on the
# facts. It stores the model weights, the character vocabulary (`stoi`/`itos`), and the
# config. We restore those and freeze the teacher with `teacher.eval()` — it will never
# be updated again; it serves only as the distillation signal.
#
# We also reload the `distill_metrics.json` written by notebook 05–06 to recover
# `best_alpha`. Rather than hard-coding 0.5, we read it dynamically so that if you
# re-ran the alpha sweep and got a slightly different result, the capstone automatically
# uses the best value you actually measured.
#
# Finally, we build the training corpus: `render_training_corpus` renders every fact
# with all templates **except** the held-out probe (which is `templates[0]`), then
# repeats the corpus 40 times to give the trainer enough tokens per epoch.

# %% [markdown]
# ## Teacher-generated augmentation (as in notebook 08)
#
# Notebook 08 established that asking the teacher to write new training text — passages
# that reflect its own distribution over the fact domain — measurably improves student
# recall. Here we reproduce that augmentation step as part of the best recipe.
#
# The procedure:
# 1. Pick 200 random seed windows (16 tokens each) from the encoded corpus.
# 2. Feed each window to the teacher and let it generate 64 more tokens with
#    `temperature=0.8` and `top_k=20` (light sampling to introduce variety without
#    losing coherence).
# 3. Concatenate all 200 generated passages with the original corpus to form `aug_data`.
#
# Why does this help? The teacher has already learned the facts deeply. When it
# generates text, it naturally continues sentences in ways that reinforce fact-
# relevant patterns — even without being explicitly prompted to do so. The student
# therefore sees more diverse phrasings of the underlying knowledge than the raw
# corpus alone provides.
#
# The `get_batch` function below samples uniformly from `aug_data`, so both the
# original corpus and the augmented passages are mixed into every training batch.

# %%
torch.manual_seed(0)
gen = []
for _ in range(200):
    start = torch.randint(0, len(fact_data) - 16, (1,)).item()
    seed = fact_data[start:start + 16].unsqueeze(0).to(device)
    out = teacher.generate(seed, max_new_tokens=64, temperature=0.8, top_k=20)
    gen.append(decode(out[0], itos))
aug_data = torch.tensor(encode(corpus + "\n" + "\n".join(gen), stoi), dtype=torch.long)

def get_batch(batch_size=32):
    ix = torch.randint(len(aug_data) - block_size - 1, (batch_size,))
    x = torch.stack([aug_data[i:i + block_size] for i in ix])
    y = torch.stack([aug_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Train the capstone student (alpha + temperature + augmentation, longer budget)
# A larger step budget than the 400-step comparison notebooks, to show how close a
# small distilled student can get to the teacher's recall.
#
# ### What `train_distill` does at each step
#
# The distillation training loop combines two loss signals:
#
# ```
# total_loss = alpha * KL(teacher_probs || student_probs)
#            + (1 - alpha) * cross_entropy(student_logits, ground_truth_tokens)
# ```
#
# - **KL divergence term** (weight `alpha`): the student is penalised for disagreeing
#   with the teacher's *full probability distribution* over the next token. This is
#   richer than a single label — it captures which alternatives the teacher considers
#   plausible and which it rules out.
# - **Cross-entropy term** (weight `1 - alpha`): the student is also penalised for
#   producing the wrong token according to the ground-truth corpus. This anchors the
#   student to real text and prevents drift from the hard training signal.
# - **`alpha = best_alpha`**: the blend that the alpha sweep in notebooks 05–06
#   identified as optimal for this task (≈ 0.5 — roughly equal weight to both signals).
# - **`temperature = 1.0`**: the teacher's logits are divided by this value before
#   softmax. Temperature 1.0 leaves the teacher's distribution unchanged. Higher
#   temperatures (> 1) would smooth the distribution, making it "softer" and carrying
#   more information about near-misses; lower temperatures would sharpen it toward a
#   near-deterministic signal. Notebook 05 found 1.0 to be effective for this fact-set.
#
# ### Why 800 steps?
#
# The comparison notebooks all used 400 steps for a fair head-to-head. Now that we are
# not comparing — we are building the best student we can — there is no reason to
# artificially cap the budget. 800 steps costs about two minutes on a laptop CPU and
# lets the combined signal converge meaningfully further. The `eval_fn` lambda prints
# the live cloze score every 400 steps so you can watch the student improve.

# %%
STEPS = 800
torch.manual_seed(0)
s_cfg = make_student(vocab_size, block_size=block_size)
student = GPT(s_cfg).to(device)
train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
              alpha=best_alpha, temperature=1.0,
              eval_fn=lambda: f"cloze {evaluate_cloze(student, stoi, itos, facts, device):.1%}",
              log_every=400)

# %% [markdown]
# ## Final scorecard
#
# The three numbers that tell the full story:
#
# ### Capstone cloze accuracy (≈ 95.0 %)
#
# `evaluate_cloze` feeds the held-out probe template for each fact to the student and
# checks whether greedy decoding produces the exact correct answer. 95.0 % means the
# student answers 9.5 out of 10 facts correctly on a phrasing it **never trained on**.
# This is the payoff metric — the number we have been building toward since notebook 01.
#
# ### Compression factor (≈ 12×)
#
# `ratio = teacher.num_params() / student.num_params()` compares the two model sizes:
# - **Teacher**: ≈ 9.4 M parameters — the full-size model that was pretrained and
#   fact-injected.
# - **Student**: ≈ 0.79 M parameters — the small model that only saw the teacher's
#   outputs and augmented text.
#
# A ratio of ~12 means the student is **12× smaller** than the teacher. At inference
# time, smaller means faster and cheaper: less memory to load, fewer FLOPs per token,
# easier to deploy on edge hardware. The capstone demonstrates that this compression
# is achievable without catastrophic recall loss.
#
# ### The assert: capstone must beat the baseline
#
# ```python
# assert capstone >= metrics["baseline_hard_label"]
# ```
#
# `baseline_hard_label` is the cloze score of a student trained with **no distillation
# at all** — just hard cross-entropy labels (notebook 04). This is the weakest possible
# starting point. The assert formalises the minimum bar: a recipe that uses
# distillation, augmentation, and a longer budget **must** beat a student trained
# without any of those things. If it does not, something went wrong.
#
# The assert is not a test that the numbers are impressive — it is a test that the
# entire pipeline is functioning correctly end to end. A passing assert is a green
# light; a failing assert is a diagnostic alarm.
#
# After the assert, the student checkpoint is saved to `checkpoints/student_capstone.pt`
# and the metrics dictionary is updated with the final scores. Notebook 11 will use
# these numbers to draw the complete course recap.

# %%
capstone = evaluate_cloze(student, stoi, itos, facts, device)
ratio = teacher.num_params() / student.num_params()
print(f"Capstone cloze: {capstone:.1%}  |  {ratio:.1f}x smaller than the teacher")
print(f"baseline {metrics['baseline_hard_label']:.1%} -> capstone {capstone:.1%}")
assert capstone >= metrics["baseline_hard_label"], "the capstone must beat the baseline"
update_metrics("capstone", capstone)
update_metrics("teacher_params", teacher.num_params())
update_metrics("capstone_params", student.num_params())

torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(s_cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/student_capstone.pt")
print("Saved checkpoints/student_capstone.pt. Continue to 11_recap.")
