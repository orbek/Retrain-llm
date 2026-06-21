# %% [markdown]
# # 04 — Student Baseline (Hard Labels)
#
# This notebook is the **control condition** of the distillation experiment.
#
# Here is where it sits in the course pipeline:
#
# > **pretrain teacher** → **inject facts** → **baseline student ← you are here** → **logit distillation** → **feature distillation**
#
# We train a *small* model — the **student** — on the same fact corpus the teacher
# learned in notebook 03. But we train it the ordinary way: **hard labels** and
# cross-entropy only. No teacher. No soft probabilities. No privileged signal.
#
# The cloze accuracy this student achieves is *the number to beat*. Every distillation
# notebook (05, 06, …) will reuse the exact same student architecture, the exact same
# random seed, and the exact same 400-step budget — the only thing that will change is
# the training **signal**. Any improvement in cloze is therefore attributable to
# distillation alone, not to a bigger model or more compute.
#
# ## Key jargon defined here
#
# | Term | Meaning in this course |
# |---|---|
# | **Student** | The small model being trained. ~1/10th the parameters of the teacher. |
# | **Teacher** | The large model from notebook 03 whose knowledge we want to transfer. |
# | **Hard labels** | The ground-truth next-character targets in the training corpus — one correct answer per position, represented as a one-hot vector. Only the correct token receives probability 1; all others receive 0. |
# | **Cross-entropy** | The loss function that measures how surprised the model is by the correct next token. Minimising it pushes the model's predicted probability for the correct token toward 1. |
# | **Baseline / control** | A training condition that uses only conventional techniques. Its result is the reference point; later notebooks must beat it to justify the added complexity. |
# | **Sample efficiency** | Getting more learning per gradient step. Distillation is more sample-efficient than hard-label training: at the same step budget, the student learns more from a teacher's soft distribution than from one-hot targets alone. |
# | **Teacher ceiling** | The teacher's cloze score — 97.5% in this run. A perfectly distilled student could approach but not exceed it. The `assert` at the end of this notebook formalises this bound. |
#
# ## Why 400 steps? Why not train to convergence?
#
# We deliberately cap training at **400 steps** — a modest, fixed budget. At this
# budget, the baseline lands around **60% cloze** (well below saturation), which
# means there is substantial headroom above it. That headroom is where distillation
# shows its value: notebooks 05 and 06 will reach noticeably higher cloze at the
# *same* 400-step cost. The lesson is **sample efficiency** — getting more knowledge
# per gradient step by learning from a richer signal.
#
# If we trained to convergence, both the baseline and the distilled variants might
# saturate at the same ceiling, obscuring the distillation advantage entirely.
#
# ## What you will see
#
# - The teacher's cloze score (≈ 97.5%) loaded from checkpoint — this is the ceiling.
# - The student trained for 400 steps on hard labels — cloze printed every 300 steps.
# - The final baseline cloze (≈ 60.0%) saved to `assets/distill_metrics.json`.
# - An `assert` confirming the student did not exceed the teacher — if it did, something
#   is wrong with the setup.
# - The student checkpoint saved for reuse in comparison notebooks.

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
import torch
from model import GPT, GPTConfig, make_student
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
# ## Working directory
#
# Jupyter launches a kernel from the `notebooks/` folder, so bare paths like
# `checkpoints/` would silently resolve to `notebooks/checkpoints/` — the wrong
# place. The cell below walks up the directory tree until it finds the project root
# (identified by `requirements.txt`) and changes the working directory there.
# After this cell runs, every path in the notebook is relative to the project root.

# %% [markdown]
# ## Vocab + teacher ceiling (from the injected teacher checkpoint)
#
# We load the teacher checkpoint produced by notebook 03. This gives us two things:
#
# 1. **The shared vocabulary** (`stoi`, `itos`) — the same character-level tokenizer
#    used to train the teacher. We must reuse it for the student; if the vocabularies
#    differed, the student's output indices would map to different characters and the
#    cloze evaluation would be meaningless.
# 2. **The teacher model** itself — fully restored from weights so we can measure its
#    cloze score and use it as the **ceiling** in the assert below.
#
# The **teacher ceiling** is the highest cloze score a student could theoretically
# achieve by perfectly imitating the teacher. In this run it is ~97.5%. The
# baseline student trained on hard labels alone will land far below it — around 60%.
# The gap between 60% and 97.5% is the opportunity distillation is designed to close.

# %%
ckpt = torch.load("checkpoints/teacher_injected.pt", weights_only=True)
stoi, itos = ckpt["stoi"], ckpt["itos"]
vocab_size = len(stoi)
teacher = GPT(GPTConfig(**ckpt["config"])).to(device)
teacher.load_state_dict(ckpt["model_state"])
facts = build_factset()
teacher_cloze = evaluate_cloze(teacher, stoi, itos, facts, device)
print(f"Teacher cloze (ceiling): {teacher_cloze:.1%}")

# %% [markdown]
# ## Fact corpus + batching
#
# `render_training_corpus` builds the text used for training. It renders every fact
# using `templates[1:]` — the non-probe phrasings — repeated 40 times to give the
# optimizer enough signal from the small fact set. The held-out probe (`templates[0]`)
# is never included, so a correct cloze score later is genuine recall, not string
# memorisation.
#
# `get_batch` samples random windows of length `block_size=256` from the corpus tensor.
# Each window produces:
#
# - **Input `x`**: characters at positions `[i, i+block_size)`.
# - **Target `y`** (the hard labels): characters at positions `[i+1, i+block_size+1)` —
#   the next character at every position.
#
# These targets are the **hard labels**: for each position there is exactly one correct
# character. The model is penalised in proportion to how much probability it assigns to
# any other character. This is standard language-model training — also called
# **teacher-forced cross-entropy** because the true sequence is always fed as input,
# never the model's own predictions.

# %%
block_size = 256
corpus = render_training_corpus(facts, repeats=40)
data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

def get_batch(batch_size=32):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Train the baseline student (identical seed/config to the distilled student)
#
# Three constraints are locked in here to guarantee a **fair comparison** with the
# distillation notebooks that follow:
#
# 1. **Same student architecture** — `make_student(vocab_size, block_size=256)` produces
#    the same small GPT config every time. The student is deliberately much smaller than
#    the teacher: it has roughly 1/10th the parameters. This size gap is why the student
#    cannot simply memorise the teacher — it must compress.
# 2. **Same random seed** — `torch.manual_seed(0)` fixes the initial weights. Notebooks
#    05 and 06 will set the same seed before instantiating their students, so any
#    difference in cloze cannot be explained by a lucky (or unlucky) initialisation.
# 3. **Same step budget** — `STEPS = 400`. Every comparison notebook runs for exactly
#    400 gradient steps. Keeping the compute budget fixed means distillation must earn
#    its improvement through a better signal, not more training.
#
# The training loop itself is plain cross-entropy on hard labels:
# `student(xb, targets=yb)` returns `(logits, loss, _)` where `loss` is the standard
# next-token cross-entropy averaged over all positions in the batch. No temperature,
# no KL divergence, no teacher involved — just the student and the corpus.
#
# Loss and cloze are printed at step 0, step 300, and the final step so you can watch
# the model improve. Expect cloze to grow from ~0% (random weights) to roughly **60%**
# at step 400. That 60% is the **number to beat**.

# %%
STEPS = 400
torch.manual_seed(0)
student = GPT(make_student(vocab_size, block_size=block_size)).to(device)
print(f"Student params: {student.num_params()/1e6:.2f}M")
opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
student.train()
for step in range(STEPS):
    xb, yb = get_batch()
    _, loss, _ = student(xb, targets=yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 300 == 0 or step == STEPS - 1:
        c = evaluate_cloze(student, stoi, itos, facts, device)
        print(f"step {step:4d} | loss {loss.item():.3f} | cloze {c:.1%}")

# %% [markdown]
# ## The number to beat
#
# `evaluate_cloze` feeds the held-out probe prompt for each fact into the student and
# checks whether greedy decoding produces the exact correct answer. The result is the
# fraction of facts answered correctly.
#
# **Why ~60% and not higher?** Hard-label cross-entropy is an effective signal, but it
# is also a thin one. At each position the target is a single character — a one-hot
# spike of probability. The loss says nothing about *which* wrong answers were nearly
# correct, nothing about the teacher's uncertainty, and nothing about the relative
# plausibility of alternatives. After 400 steps the student has absorbed enough signal
# to recall the majority of facts, but not all of them. That gap — from 60% to the
# teacher ceiling of 97.5% — is what soft distillation will recover in notebooks 05
# and 06.
#
# **The assert** `baseline <= teacher_cloze + 1e-9` is a sanity check that the student
# has not somehow surpassed the teacher on the cloze probe. If this fires, it almost
# certainly means the teacher checkpoint was accidentally overwritten with a weaker
# model, or the cloze evaluation is reading the wrong checkpoint. The small epsilon
# (`1e-9`) tolerates floating-point rounding.
#
# The baseline score is persisted to `assets/distill_metrics.json` under the key
# `"baseline_hard_label"` so the comparison chart built in the capstone notebook can
# read it alongside the distillation scores without re-running training.
#
# The student checkpoint is saved to `checkpoints/student_baseline.pt`. Notebooks 05
# and 06 start from scratch (fresh seed), not from this checkpoint — but the file is
# kept here as a reference artefact.
#
# **Summary of committed numbers for this run:**
#
# | Metric | Value |
# |---|---|
# | Teacher ceiling (cloze) | ~97.5% |
# | Baseline student (cloze, 400 steps, hard labels) | ~60.0% |
# | Gap distillation must close | ~37.5 percentage points |

# %%
baseline = evaluate_cloze(student, stoi, itos, facts, device)
print(f"BASELINE (hard-label) cloze: {baseline:.1%}")
assert baseline <= teacher_cloze + 1e-9, "a small student should not exceed the teacher"
update_metrics("baseline_hard_label", baseline)

import dataclasses
torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(student.config),
            "stoi": stoi, "itos": itos}, "checkpoints/student_baseline.pt")
print("Saved checkpoints/student_baseline.pt. Continue to 05_logit_distillation.")
