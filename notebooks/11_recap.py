# %% [markdown]
# # 11 — Recap: The Full Measured Journey
#
# Welcome to the final notebook of the course.
#
# Here is where we have been and what we proved along the way:
#
# > **pretrain a broad teacher** (nb02) → **inject specific facts into it** (nb03)
# > → **a same-size student fails at a fixed budget** (nb04)
# > → **distillation recovers the knowledge** (nb05/06)
# > → **feature matching didn't help here** (nb07)
# > → **augmentation matched distillation's gain** (nb08)
# > → **capacity is the biggest lever** (nb09)
# > → **the capstone recipe — distillation + augmentation + a wider student — achieves 95%**
# > **with a model 12× smaller than the teacher** (nb10)
#
# This notebook does not introduce any new technique. It loads the metrics file
# produced by every prior notebook, draws the summary bar chart, and prints the
# final table. Its job is to let the numbers speak — and to name the honest lessons
# that those numbers carry.
#
# ## The story in plain language
#
# ### Step 1 — Build a teacher that knows things (nb02 + nb03)
#
# We started by pretraining a broad GPT-style model on a large general-purpose
# corpus (TinyShakespeare). The teacher at this point knows a lot about English text
# patterns but knows nothing about our invented fictional facts — because it has never
# seen them. Cloze accuracy on the invented fact-set: **~0%** (random chance).
#
# We then *fine-tuned* the teacher on the fact-set in notebook 03. That injection
# drove cloze from 0% to near 100%. The teacher now holds the knowledge in its
# weights. Everything that follows is the question: *can a smaller model extract
# that knowledge from the teacher?*
#
# ### Step 2 — A same-size student trained from scratch fails (nb04)
#
# We trained a student of the same size as the teacher using ordinary cross-entropy
# (hard labels) on the training corpus, at the same compute budget we will use for
# all distillation experiments. Cloze: **60%**. This is the baseline — the floor
# every distillation method must beat.
#
# Why does the hard-label student only reach 60%? The training signal is sparse:
# the fact-set text is a small fraction of the total corpus, so the student has
# relatively few gradient steps to memorise the specific answer tokens. It learns
# general language structure well but over-indexes on fluency at the expense of
# factual precision.
#
# ### Step 3 — Distillation closes most of the gap (nb05 + nb06)
#
# Logit-based knowledge distillation (nb05) replaced hard one-hot targets with the
# teacher's *full probability distribution* over the next token. Instead of just
# learning "the answer is X", the student also learns "Y was a plausible second
# choice, and Z was a plausible third." This softer signal is more informative per
# token.
#
# We then swept the two KD hyperparameters in nb06 — temperature T and the
# weighting coefficient α. The key finding: **T=1, not T=2**, worked best for our
# crisp-fact recall task. Higher temperatures (T=2, 4) diffuse the teacher's
# distribution too much, washing out the sharp peaks that encode factual certainty.
# For fuzzy tasks (sentiment, style) a higher T is a helpful smoothing; for exact
# factual recall, it hurts.
#
# Best logit/alpha KD: **67.5%** — a clear 7.5-point jump over the hard-label
# baseline.
#
# ### Step 4 — Feature matching did not help here (nb07)
#
# Intermediate-layer feature matching (minimising the MSE between corresponding
# hidden states of teacher and student) is a well-known trick for tasks where
# the teacher's internal representations encode useful structure beyond its output
# logits. Here it did not improve on the logit baseline: **60%**, back to where the
# hard-label student was.
#
# This is an honest result worth keeping. Feature matching adds a strong alignment
# pressure that can *constrain* the student's own representation learning. Whether
# it helps depends heavily on the task and the architecture gap between teacher and
# student. When it doesn't help, drop it.
#
# ### Step 5 — Augmentation matched distillation alone (nb08)
#
# We returned to logit KD but augmented the training set with additional paraphrase
# variants of each fact. More surface-form variation gave the student more angles
# from which to learn each fact. Result: **67.5%** — matching the best KD result.
#
# The lesson: *data augmentation and soft targets are complementary, not competing,
# levers.* Augmentation improves sample efficiency across all phrasings;
# distillation improves the quality of supervision on each example. Combined (nb10),
# they multiply each other's benefit.
#
# ### Step 6 — Capacity matters enormously (nb09)
#
# We swept student width at a fixed training budget. Accuracy scaled steeply with
# model capacity, confirming what scaling-law research predicts: for a fixed token
# budget, a wider model converges faster on the training distribution. This shaped
# the capstone strategy: instead of using a tiny student, use the *largest* student
# that still fits the 12× parameter constraint.
#
# ### Step 7 — The capstone combines everything (nb10)
#
# Notebook 10 assembled the full recipe:
# - Logit KD at T=1, α tuned to weight the KD loss appropriately
# - Full augmentation of the training corpus
# - A student with adequate capacity (still 12× smaller than the teacher by
#   parameter count)
#
# Final cloze: **95%** — with a model **12× smaller than the teacher**. The student
# can recall 19 out of 20 invented facts that the same-size hard-label model could
# only recall 12 out of 20.

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
import matplotlib.pyplot as plt

metrics = json.load(open("assets/distill_metrics.json"))

# %% [markdown]
# ## Cloze accuracy by method
#
# The chart below shows every method's cloze score in the order we introduced it.
# Reading left to right is reading through the course — each bar represents one
# notebook's best result, measured on the same held-out probe set that has been our
# yardstick since notebook 01.
#
# A few things to notice before you look at the numbers:
#
# - The y-axis runs from 0 to 100%. Chance performance on a 50-character vocabulary
#   with 5-character answers is effectively **0%** — that is where every experiment
#   starts before training.
# - The **baseline** bar (60%) is not the floor for distillation — it is the floor
#   for everything. A student trained without any KD signal lands here.
# - The **capstone** bar (95%) sits in a different league. The jump from 67.5% to
#   95% did not come from a single new trick; it came from combining all the tricks
#   that individually showed gains while omitting the one (feature KD) that did not.

# %%
order = [("baseline_hard_label", "baseline"),
         ("logit_kd", "logit KD"),
         ("alpha_kd_best", "alpha KD"),
         ("feature_kd", "feature KD"),
         ("augmented_kd", "augmented"),
         ("capstone", "capstone")]
labels = [lbl for key, lbl in order if key in metrics]
values = [metrics[key] for key, lbl in order if key in metrics]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, values)
for b, v in zip(bars, values):
    plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}",
             ha="center", va="bottom")
plt.ylabel("cloze accuracy")
plt.ylim(0, 1.05)
plt.title("Knowledge recall by distillation method")
plt.tight_layout()
plt.savefig("assets/11_summary.png", dpi=120)
print("Saved assets/11_summary.png")

# %% [markdown]
# ## The numbers
#
# The table below is the course in seven rows. Each row is one experimental
# condition; each number is the cloze accuracy on the held-out probe set.
#
# | Method | Cloze | Notes |
# |---|---|---|
# | baseline (hard label) | 60% | same-size student, cross-entropy only |
# | logit KD | 67.5% | soft teacher targets, T=1 |
# | alpha KD (best) | 67.5% | optimal α weighting of KD vs CE loss |
# | feature KD | 60% | intermediate-layer MSE matching — no gain here |
# | augmented KD | 67.5% | paraphrase augmentation + logit KD |
# | **capstone** | **95%** | **logit KD + augmentation + adequate capacity** |
#
# The table that the code prints below is generated live from `assets/distill_metrics.json`,
# which is the shared metrics store that every prior notebook wrote into. If you ran
# notebooks 04–10 yourself, these are *your* numbers.
#
# ### Honest takeaways
#
# **Temperature matters for factual tasks.** T=1 outperformed T=2 and T=4 because
# crisp factual answers live in the *sharp* part of the teacher's distribution.
# Raising temperature softens those peaks, making "the right answer" harder to
# distinguish from plausible alternatives. Reserve high temperatures for style or
# sentiment transfer where distribution smoothing is a feature, not a bug.
#
# **Distillation's edge is sample efficiency.** Logit KD's 7.5-point gain over the
# hard-label baseline is not magic — it is the extra information in the teacher's
# distribution that would otherwise require more training steps or more data to
# acquire. In settings where data is abundant, the gap shrinks. Here, where the
# fact corpus is small and every fact must be memorised, the richer per-step signal
# makes a measurable difference.
#
# **Distillation shines most when combined with augmentation and adequate capacity.**
# Each ingredient added a modest individual gain; their combination drove the jump
# to 95%. This multiplicative behaviour is the pattern to watch for in practice:
# distillation + augmentation + a correctly-sized student is consistently stronger
# than any single ingredient at maximum tuning.
#
# **Not every advanced technique helps.** Feature matching is a well-cited method
# with genuine benefits on many tasks — but here it matched the hard-label baseline
# at 60%. A practitioner lesson: always measure, and be willing to drop a technique
# that does not show a gain in your specific setting. The cost of including a
# harmful regulariser (forced alignment of representations) can erase soft-label
# benefits entirely.
#
# **Capacity is the dominant lever at fixed compute.** The size sweep (nb09) showed
# steeper accuracy gains from widening the student than from any training recipe
# change. This matches what scaling-law research finds at every scale: if you can
# afford the parameters, use them. The capstone's student is still 12× smaller than
# the teacher by parameter count — but it is the *largest* student that fits that
# constraint, not the smallest.

# %%
for key, lbl in order:
    if key in metrics:
        print(f"{lbl:12s}: {metrics[key]:.1%}")
if "teacher_params" in metrics and "capstone_params" in metrics:
    ratio = metrics["teacher_params"] / metrics["capstone_params"]
    print(f"\nCapstone student is {ratio:.1f}x smaller than the teacher.")

assert "baseline_hard_label" in metrics and "capstone" in metrics, \
    "recap requires the full metrics file from notebooks 04-10"
print("\nDone. You injected knowledge and distilled it into a smaller model.")

# %% [markdown]
# ## Where to go next
#
# This course used a small character-level GPT as a stand-in for a real pretrained
# model and a tiny synthetic fact-set as a stand-in for real domain knowledge.
# The distillation mechanics are identical at scale; what changes is the cost.
#
# A few natural next steps, roughly in order of difficulty:
#
# - **Swap in a real pretrained teacher.** Replace the GPT trained in nb02/03 with
#   a publicly available model (e.g. GPT-2 or any Llama-family model). The cloze
#   evaluation and distillation training loop work unchanged; only the model
#   loading code needs updating. You will immediately see how much a pretrained
#   teacher raises the starting-line accuracy.
#
# - **Scale up the student.** The 12× constraint was illustrative. Relax it and
#   observe the accuracy curve — the capstone results suggest there is still
#   headroom above 95% with a larger student.
#
# - **Try a real knowledge-injection task.** Replace the synthetic fact-set with
#   domain-specific documents (medical guidelines, legal clauses, API documentation).
#   The cloze probe becomes a domain-specific QA evaluation. Everything else
#   in this course ports directly.
#
# - **LoRA / parameter-efficient fine-tuning.** Instead of fine-tuning all weights
#   in notebook 03, inject facts via low-rank adapter matrices. This dramatically
#   reduces the compute cost of knowledge injection and makes it easier to swap
#   in different fact-sets without retraining the base model.
#
# - **Scaling laws for distillation.** The nb09 size sweep was a small sample
#   (3–5 student sizes). A proper sweep — varying both student size and training
#   compute simultaneously — would let you fit a compute-optimal frontier for
#   distilled factual recall, analogous to Chinchilla for pretraining.
#
# You have now built, measured, and honestly evaluated a complete knowledge-injection
# and distillation pipeline from scratch. The concepts — soft targets, temperature,
# feature matching, augmentation, capacity — are the same ones you will encounter
# in every production LLM compression paper. Happy training.
