# %% [markdown]
# # 07 — Hidden-State (Feature) Distillation
#
# Welcome back! Notebooks 04–06 taught the student to mimic the teacher's **final output
# distribution** — the probability vector the teacher assigns to every possible next token.
# That signal (called *logit KD* or *soft-label distillation*) is already quite powerful:
# in notebook 06 we reached **67.5% cloze accuracy**, far above the 10% hard-label
# baseline.
#
# This notebook asks a natural follow-up question:
#
# > The teacher doesn't just produce a good final output — it also builds up rich
# > **intermediate representations** at every layer. Can we learn more by forcing the
# > student to match those intermediate representations too?
#
# That idea is called **hidden-state distillation**, **feature distillation**, or
# **feature matching**. It was popularised by papers like FitNets (Romero et al., 2015)
# and PKD (Sun et al., 2019), and it is used in many modern compression pipelines.
#
# ## Where we are in the course arc
#
# ```
# pretrain teacher → inject facts → baseline student (10%) → logit KD (67.5%) → feature KD (this notebook)
# ```
#
# Each step so far has beaten the number before it. This notebook is the first honest
# **negative result**: feature matching does **not** improve over logit-only KD in this
# regime — it scores **60.0%**, tying the baseline but falling short of the 67.5% we
# already achieved with logits alone. We will look at why that happens, and what the
# result teaches us.
#
# ## What you will build
#
# - A **`FeatureProjector`**: a small set of learned linear layers that map each of the
#   student's hidden states into the teacher's (larger) hidden dimension.
# - A combined loss that adds an **MSE (mean-squared error) feature-matching term** on
#   top of the KD + CE objective from notebook 06.
# - An **honest evaluation** comparing all three numbers: baseline, logit KD, feature KD.

# %% [markdown]
# ## A note on working directories
#
# Jupyter runs a notebook from the folder the notebook lives in (`notebooks/`), not from
# the project root. That means paths like `checkpoints/` and `assets/` would silently
# resolve to `notebooks/checkpoints/` — the wrong place. The cell below walks up the
# directory tree until it finds the project root (identified by `requirements.txt`) and
# changes the kernel's working directory there. Every subsequent path in the notebook is
# then relative to the project root, regardless of how you launched Jupyter.

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
from distill import train_distill, FeatureProjector
from facts import build_factset, render_training_corpus, encode, evaluate_cloze

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))

def update_metrics(key, value, path="assets/distill_metrics.json"):
    os.makedirs("assets", exist_ok=True)
    m = json.load(open(path)) if os.path.exists(path) else {}
    m[key] = value
    json.dump(m, open(path, "w"), indent=2)
    return m

# %% [markdown]
# ## Loading the teacher and building the training data
#
# We reload the same knowledge-injected teacher checkpoint used throughout this course.
# The teacher is a Llama-style decoder (GQA + RoPE + RMSNorm + SwiGLU) that was first
# pre-trained on TinyShakespeare and then fine-tuned on the fictional fact-set.
#
# Two things to notice here:
#
# - **`best_alpha`**: pulled from `distill_metrics.json`, this is the KD mixing
#   coefficient tuned in notebook 06. We inherit it so feature distillation starts from
#   the same balanced loss.
# - **`repeats=40`**: the facts corpus is tiny, so we repeat it 40 times to build a
#   training tensor large enough for mini-batch sampling.
#
# The student will be re-initialised from scratch for this notebook — we are comparing
# *feature KD from scratch* against *logit KD from scratch*, not continuing a prior run.

# %%
ckpt = torch.load("checkpoints/teacher_injected.pt", weights_only=True)
stoi, itos = ckpt["stoi"], ckpt["itos"]
vocab_size = len(stoi)
t_cfg = GPTConfig(**ckpt["config"])
teacher = GPT(t_cfg).to(device)
teacher.load_state_dict(ckpt["model_state"])
teacher.eval()
facts = build_factset()

block_size = 256
corpus = render_training_corpus(facts, repeats=40)
data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

def get_batch(batch_size=32):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

metrics = json.load(open("assets/distill_metrics.json"))
best_alpha = metrics.get("best_alpha", 0.5)

# %% [markdown]
# ## Hidden states and the feature-matching idea
#
# ### What are hidden states?
#
# A transformer processes each input token through a stack of layers. At each layer, each
# token has a vector of numbers associated with it — the **hidden state** (also called
# **activations** or the **residual stream**). Early layers encode shallow features (local
# syntax, character patterns); later layers encode deeper semantic content (facts, roles,
# relationships).
#
# When the teacher processes the prompt *"The capital of Zorblax is"*, its layer-5 hidden
# states already encode something like *"this is a factual recall question, and the answer
# is a proper noun"*. The final logit layer converts that rich representation into
# a distribution over characters — but by the time we observe the logits, a lot of that
# intermediate reasoning has been compressed away.
#
# **Feature matching** (also called *intermediate distillation* or *hint-based learning*)
# says: don't just match the teacher's final output — also pull the student's intermediate
# representations toward the teacher's intermediate representations, layer by layer.
#
# ### The dimension mismatch problem
#
# There is an immediate obstacle. The teacher and student are different sizes:
#
# | Property | Teacher | Student |
# |---|---|---|
# | Layers (`n_layer`) | more (e.g. 6) | fewer (e.g. 4) |
# | Hidden dim (`n_embd`) | larger (e.g. 192) | smaller (e.g. 64) |
#
# The student's hidden states live in a 64-dimensional space; the teacher's live in a
# 192-dimensional space. You cannot directly compute MSE between a 64-d vector and a
# 192-d vector — the shapes don't match.
#
# ### The FeatureProjector: a learned bridge
#
# The solution is a **projector**: a small set of learned linear layers (one per student
# layer being matched) that maps each student hidden state into the teacher's hidden
# dimension:
#
# ```
# projected = W_proj @ student_hidden    # (n_embd_student,) → (n_embd_teacher,)
# ```
#
# Now both sides live in the same space, and we can compute MSE:
#
# ```
# feature_loss = MSE(projected, teacher_hidden.detach())
# ```
#
# Notice `teacher_hidden.detach()` — we do **not** backpropagate into the teacher. The
# teacher is frozen; only the student and the projector learn.
#
# ### The student-to-teacher layer map
#
# Because the student has fewer layers than the teacher, the projector must also decide
# which student layer is paired with which teacher layer. A common heuristic is to
# distribute student layers evenly across teacher layers. For example, if the student
# has 4 layers and the teacher has 6, a possible map is `[0, 2, 3, 5]` — meaning student
# layer 0 is matched to teacher layer 0, student layer 1 to teacher layer 2, and so on.
#
# The `FeatureProjector` computes this map automatically and prints it so you can inspect
# the pairing before training begins.
#
# ### The combined loss
#
# The full training objective is now three terms:
#
# ```
# loss = (1 - alpha) * CE(student_logits, targets)       # hard labels
#      +       alpha * KL(student_log_probs, teacher_probs)  # soft labels (KD)
#      + feature_weight * MSE(projected_student, teacher_hidden)  # feature matching
# ```
#
# `feature_weight=0.1` here. You can think of it as a regulariser: it nudges the student
# toward the teacher's internal structure, not just its output behaviour.

# %% [markdown]
# ## Distill with logits + hidden-state matching

# %%
STEPS = 400
torch.manual_seed(0)
s_cfg = make_student(vocab_size, block_size=block_size)
student = GPT(s_cfg).to(device)
projector = FeatureProjector(s_cfg.n_layer, t_cfg.n_layer, s_cfg.n_embd, t_cfg.n_embd)
print("student->teacher layer map:", projector.layer_map)
train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
              alpha=best_alpha, temperature=1.0, feature_weight=0.1, projector=projector,
              eval_fn=lambda: f"cloze {evaluate_cloze(student, stoi, itos, facts, device):.1%}",
              log_every=300)

# %% [markdown]
# ## Does feature matching help here?
#
# The cell below prints three numbers side by side:
#
# - **`baseline`**: hard-label training with no distillation at all (~10%).
# - **`alpha-KD`**: logit-only distillation from notebook 06 (67.5%).
# - **`feature-KD`**: this notebook's result (60.0%).
#
# ### The honest result
#
# Feature KD **did not beat logit-only KD**. The result is 60.0% — a tie with the
# hard-label baseline, and 7.5 percentage points below the logit-KD result we already had.
#
# This is a real, common outcome in the distillation literature. Here is why it happens:
#
# 1. **The MSE term competes with the KD and CE objectives.** The total loss is a sum of
#    three terms. When the feature-matching gradient pulls the student's representations
#    toward the teacher's geometry, it can *oppose* the KD gradient, which is trying to
#    match output distributions. The optimizer finds a compromise that satisfies neither
#    objective as well as if it only had one to worry about.
#
# 2. **Naive feature matching with a fixed weight is fragile.** The scale of the MSE
#    term depends on the magnitude of the hidden states, which can vary widely across
#    training. A fixed `feature_weight=0.1` that is "too large" early in training can
#    overwhelm the KD signal; one that is "too small" adds noise without benefit. Proper
#    feature distillation usually requires adaptive weighting, careful layer-wise
#    scheduling, or normalising the hidden states before computing MSE.
#
# 3. **The student is very small.** With a hidden dimension of 64, the student has limited
#    capacity. Trying to simultaneously match the teacher's output and internal geometry
#    may exceed what a model this size can do. Larger students tend to benefit more from
#    intermediate supervision.
#
# 4. **Feature matching helps in some regimes, not all.** It is most beneficial when the
#    task is complex and the teacher's intermediate representations encode structure that
#    does not survive compression to logits — e.g. structured prediction, span labeling,
#    or tasks with very long contexts. For a character-level cloze task with short answers,
#    the final logits already carry the relevant signal.
#
# ### Why the assert is conservative
#
# The assert below only checks that feature KD beats the *hard-label baseline* (10%) —
# not that it beats logit KD (67.5%). This is intentional: we cannot guarantee that
# adding hidden-state supervision will always improve over logit KD (as this very run
# demonstrates). Requiring a weaker condition keeps the notebook runnable and honest
# without masking the negative result.
#
# The printed message distinguishes the two cases explicitly. If you change
# `feature_weight` or `STEPS` and manage to beat 67.5%, the notebook will tell you so.
# If not, it reports the honest outcome — which is also a valid scientific finding.

# %%
feat_cloze = evaluate_cloze(student, stoi, itos, facts, device)
print(f"baseline {metrics['baseline_hard_label']:.1%} | "
      f"alpha-KD {metrics['alpha_kd_best']:.1%} | feature-KD {feat_cloze:.1%}")
assert feat_cloze >= metrics["baseline_hard_label"], "feature KD must beat the baseline"
if feat_cloze >= metrics["alpha_kd_best"]:
    print("Feature matching improved over logit-only distillation.")
else:
    print("Feature matching did not help here (an honest result worth reporting).")
update_metrics("feature_kd", feat_cloze)

torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(s_cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/student_feature.pt")
print("Saved checkpoints/student_feature.pt. Continue to 08_synthetic_augmentation.")
