# %% [markdown]
# # 10 — Capstone: The Best Recipe
# Combine the techniques that actually HELPED — the best alpha blend, temperature
# 1.0, and teacher-generated augmentation — into the strongest small student, and
# train it a bit longer. We deliberately EXCLUDE hidden-state matching: notebook 07
# measured it not helping in this regime, so an honest "best recipe" leaves it out.

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
# ## Teacher-generated augmentation (as in notebook 08)

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
