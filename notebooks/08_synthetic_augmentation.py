# %% [markdown]
# # 08 — Synthetic-Data Augmentation
# The teacher can label any text. We let it GENERATE extra fact-flavored prose and
# distill the student on the enlarged corpus. Seeds come from the training corpus,
# never the held-out probe phrasing.

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

# %%
aug_cloze = evaluate_cloze(student, stoi, itos, facts, device)
print(f"alpha-KD (no aug) {metrics['alpha_kd_best']:.1%} | augmented {aug_cloze:.1%}")
assert aug_cloze >= metrics["baseline_hard_label"], "augmented KD must beat the baseline"
update_metrics("augmented_kd", aug_cloze)

torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(s_cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/student_augmented.pt")
print("Saved checkpoints/student_augmented.pt. Continue to 09_size_sweep.")
