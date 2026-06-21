# %% [markdown]
# # 04 — Student Baseline (Hard Labels)
# Train a small student on the fact corpus with ordinary cross-entropy. No teacher.
# Its cloze accuracy is the number the distillation notebooks must beat.

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
# ## Vocab + teacher ceiling (from the injected teacher checkpoint)

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
