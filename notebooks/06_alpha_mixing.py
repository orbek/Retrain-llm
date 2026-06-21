# %% [markdown]
# # 06 — Mixing Ground-Truth + KD (alpha)
# alpha=0 is the hard-label baseline; alpha=1 is pure KD. Sweeping alpha shows the
# best blend. The sweep includes alpha=0, so the best point can only match or beat
# the baseline.

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
import matplotlib.pyplot as plt
from model import GPT, GPTConfig, make_student
from distill import train_distill
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
data = torch.tensor(encode(corpus, stoi), dtype=torch.long)

def get_batch(batch_size=32):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Sweep alpha (fresh, identically-seeded student per setting)

# %%
STEPS = 400
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []
for a in alphas:
    torch.manual_seed(0)
    student = GPT(make_student(vocab_size, block_size=block_size)).to(device)
    train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
                  alpha=a, temperature=1.0)
    c = evaluate_cloze(student, stoi, itos, facts, device)
    results.append((a, c, student))
    print(f"alpha {a:.2f} | cloze {c:.1%}")

# %% [markdown]
# ## Plot and pick the best

# %%
xs = [a for a, _, _ in results]
ys = [c for _, c, _ in results]
plt.figure(figsize=(6, 4))
plt.plot(xs, ys, "o-")
plt.xlabel("alpha (0 = hard labels, 1 = pure KD)")
plt.ylabel("cloze accuracy")
plt.title("Distillation: ground-truth vs KD blend")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("assets/06_alpha_sweep.png", dpi=120)
print("Saved assets/06_alpha_sweep.png")

best_alpha, best_cloze, best_student = max(results, key=lambda r: r[1])
baseline = json.load(open("assets/distill_metrics.json"))["baseline_hard_label"]
print(f"best alpha={best_alpha} cloze={best_cloze:.1%} (baseline {baseline:.1%})")
assert best_cloze >= baseline, "the best blend must at least match the baseline"
update_metrics("alpha_kd_best", best_cloze)
update_metrics("best_alpha", best_alpha)

torch.save({"model_state": best_student.state_dict(),
            "config": dataclasses.asdict(best_student.config),
            "stoi": stoi, "itos": itos}, "checkpoints/student_alpha.pt")
print("Saved checkpoints/student_alpha.pt. Phase 2 complete.")
