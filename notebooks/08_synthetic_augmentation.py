# %% [markdown]
# # 08 — Synthetic-Data Augmentation
#
# So far we have distilled the student using the *existing* training corpus — the same
# sentences the teacher was injected on. But there is a key insight we have not yet
# exploited: **the teacher is a generative model**. It can write new text, not just
# score existing text.
#
# This notebook turns the teacher into a data factory. We feed it short seeds drawn from
# the training corpus, let it write fact-flavored prose, and add that generated text to
# the distillation dataset. Training the student on this *enlarged* corpus is called
# **data augmentation**.
#
# ## What you will build
#
# - A **generation loop** that seeds the teacher from random windows of the training corpus
#   and samples continuations using temperature scaling and top-k filtering.
# - An **augmented dataset** that concatenates the real corpus with the generated text.
# - A **distillation run** on the augmented dataset using the best alpha from notebook 07.
# - An **honest comparison** of results: augmented_kd = 67.5%, baseline = 60.0%,
#   logit-KD = 67.5%.
#
# ## Key ideas introduced here
#
# | Term | Plain-language definition |
# |------|--------------------------|
# | **Data augmentation** | Enlarging a training dataset by creating new examples that share the same statistical character as the originals. The new examples are not ground truth — they are plausible variations. |
# | **Sampling** | Picking the next token by drawing randomly from the model's output probability distribution, rather than always picking the single most likely token (greedy). |
# | **Temperature** | A scalar that sharpens or flattens the output distribution before sampling. Low temperature → model picks safe, high-probability tokens. High temperature → more variety, more risk. |
# | **Top-k** | Restrict sampling to only the *k* most probable tokens; set all others to zero probability. Prevents the model from ever choosing a wildly unlikely token while still allowing diversity among the top candidates. |
# | **Probe purity** | The guarantee that no generated seed text is derived from the held-out cloze probe phrasing. Without this, augmented text could leak probe-adjacent patterns into training, inflating eval scores. |
#
# ## Probe purity — why it matters
#
# This is worth dwelling on because it is easy to get wrong.
#
# The cloze evaluation in notebook 01 reserves `templates[0]` for each fact as a
# held-out probe. The student is *never* trained on those exact phrasings. That is how
# we know a correct answer reflects genuine knowledge rather than string memorization.
#
# When we augment with generated text, we seed the teacher from random windows of the
# **training corpus** — which was built from `templates[1:]` only. The held-out
# `templates[0]` phrasings are never present in the seed pool. So the teacher can never
# accidentally regenerate them, and the probe stays unseen throughout training.
#
# If we had seeded from the full fact-set (including `templates[0]`), the teacher could
# generate sentences that structurally echo the probe. The student would train on
# probe-flavored text and score artificially high — a form of evaluation leakage. By
# keeping seeds strictly within the training corpus, we keep the evaluation honest.
#
# ## The arc of this notebook in the course
#
# > pretrain teacher → inject facts → **baseline student (60.0%)** → logit-KD (67.5%) →
# > alpha-KD (67.5%) → feature-KD (60.0%) → **augmented-KD (67.5%)** → size sweep → capstone
#
# Augmentation held parity with logit-KD and meaningfully beat the baseline. It did not
# regress. In real low-data regimes — few training sentences, rare entities — augmentation
# typically helps *more* than it does here, because the student has less real text to learn
# from and benefits more from the teacher's paraphrasing. Our synthetic corpus is already
# dense with fact-repetitions, so the gain is a parity result rather than a breakthrough.
# That is an honest result worth understanding.

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
teacher = GPT(GPTConfig(**ckpt["config"])).to(device)
teacher.load_state_dict(ckpt["model_state"])
teacher.eval()
facts = build_factset()

block_size = 256
corpus = render_training_corpus(facts, repeats=40)
fact_data = torch.tensor(encode(corpus, stoi), dtype=torch.long)
metrics = json.load(open("assets/distill_metrics.json"))
best_alpha = metrics.get("best_alpha", 0.5)

# %% [markdown]
# ## Generate extra data from the teacher (seeded from the training corpus)
#
# ### How the generation loop works
#
# Each iteration of `generate_augmented`:
#
# 1. **Picks a random window** of `seed_len=16` characters from `fact_data` — the
#    tokenised training corpus. This window is the prompt.
# 2. **Feeds the prompt** to the teacher and calls `teacher.generate`, which autoregressively
#    appends one token at a time until `gen_len=64` new tokens have been produced.
# 3. **Decodes the full output** (prompt + continuation) back to a string and collects it.
#
# We repeat this 200 times and join the results. That produces roughly
# 200 × 64 = 12,800 new characters — about 20% on top of the original corpus.
#
# ### Temperature and top-k in this context
#
# The generation call uses `temperature=0.8` and `top_k=20`.
#
# - **temperature=0.8** sharpens the distribution slightly below the raw model output
#   (temperature=1.0). This keeps the generated text coherent and fact-flavored, rather
#   than drifting into random character noise at high temperature.
# - **top_k=20** means at each step only the 20 most probable tokens are in play. With a
#   character-level vocabulary of ~60 characters, top-k=20 excludes the bottom ~⅔ of
#   characters that are contextually inappropriate. The result is fluent, on-topic text.
#
# A lower temperature or smaller top-k would produce text that closely echoes the training
# sentences — high fidelity but low variety. A higher temperature or larger top-k would
# introduce more novelty but also more incoherence. The chosen values are a practical
# middle ground: the generated text covers the fact space with mild paraphrase variation.

# %%
torch.manual_seed(0)
def generate_augmented(n_samples=200, seed_len=16, gen_len=64):
    texts = []
    for _ in range(n_samples):
        start = torch.randint(0, len(fact_data) - seed_len, (1,)).item()
        seed = fact_data[start:start + seed_len].unsqueeze(0).to(device)
        out = teacher.generate(seed, max_new_tokens=gen_len, temperature=0.8, top_k=20)
        texts.append(decode(out[0], itos))
    return "\n".join(texts)

generated = generate_augmented()
print("Generated chars:", len(generated))
print("Sample:\n", generated[:200])

aug_text = corpus + "\n" + generated
aug_data = torch.tensor(encode(aug_text, stoi), dtype=torch.long)

def get_batch(batch_size=32):
    ix = torch.randint(len(aug_data) - block_size - 1, (batch_size,))
    x = torch.stack([aug_data[i:i + block_size] for i in ix])
    y = torch.stack([aug_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Distill on the augmented corpus
#
# The distillation setup here is identical to notebook 07, with one change: `get_batch`
# now draws from `aug_data` (the concatenated real + generated corpus) instead of the
# original corpus alone.
#
# We reuse `best_alpha` from the metrics file — the alpha value that produced the best
# cloze score in the alpha-sweep of notebook 07. Using the pre-tuned alpha avoids
# re-running a search, and ensures the augmentation experiment is a fair comparison: the
# only variable changed relative to the best prior run is the dataset size and source.
#
# The student architecture and all training hyperparameters (STEPS=400, lr=3e-4,
# temperature=1.0) are held fixed. Any difference in final cloze score is attributable
# to augmentation, not to hyperparameter differences.

# %%
STEPS = 400
torch.manual_seed(0)
s_cfg = make_student(vocab_size, block_size=block_size)
student = GPT(s_cfg).to(device)
train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
              alpha=best_alpha, temperature=1.0,
              eval_fn=lambda: f"cloze {evaluate_cloze(student, stoi, itos, facts, device):.1%}",
              log_every=300)

# %% [markdown]
# ## Did augmentation help?
#
# ### Reading the result honestly
#
# | Approach | Cloze accuracy |
# |----------|---------------|
# | Baseline (hard-label cross-entropy) | 60.0% |
# | Logit-KD (soft targets, temperature=1.0) | 67.5% |
# | Alpha-KD (best α blend) | 67.5% |
# | Feature-KD (intermediate representation) | 60.0% |
# | **Augmented-KD (this notebook)** | **67.5%** |
#
# Augmentation **beat the baseline** (60.0% → 67.5%) and matched the best soft-target
# results from notebooks 05–07. It did not regress — adding generated text did not hurt.
#
# Why did augmentation not push *above* 67.5%? Our training corpus is already highly
# repetitive: 40 renders of 10 facts with multiple templates. The student has seen each
# fact phrased many ways. The generated text paraphrases those same sentences, but the
# marginal information gain over 40 clean repetitions is limited. Augmentation earns its
# biggest gains in genuinely low-data settings: few training sentences, rare or novel
# entities, or domain-specific jargon that the teacher has strong priors on but the
# student has seen only once or twice.
#
# The `assert aug_cloze >= metrics["baseline_hard_label"]` below codifies the minimum
# expectation: augmented distillation must not be worse than the baseline. If it were,
# something went wrong (generation was noisy, seeding was corrupt, or the corpus was
# accidentally contaminated with probe phrasings).
#
# The checkpoint `student_augmented.pt` saved at the end carries this student forward.
# Notebook 09 will use the same distillation pipeline at multiple model sizes to explore
# how scale interacts with these techniques.

# %%
aug_cloze = evaluate_cloze(student, stoi, itos, facts, device)
print(f"alpha-KD (no aug) {metrics['alpha_kd_best']:.1%} | augmented {aug_cloze:.1%}")
assert aug_cloze >= metrics["baseline_hard_label"], "augmented KD must beat the baseline"
update_metrics("augmented_kd", aug_cloze)

torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(s_cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/student_augmented.pt")
print("Saved checkpoints/student_augmented.pt. Continue to 09_size_sweep.")
