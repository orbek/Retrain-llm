# %% [markdown]
# # 09 — Size Sweep
# How small can the distilled student get and still recall the facts? Distill a few
# sizes from the same teacher and plot cloze vs parameters.

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
import matplotlib.pyplot as plt
from model import GPT, GPTConfig, make_student, make_tiny_student
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

best_alpha = json.load(open("assets/distill_metrics.json")).get("best_alpha", 0.5)

# %% [markdown]
# ## Distill each size

# %%
mid = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                n_layer=3, n_head=4, n_kv_head=2, n_embd=96, dropout=0.1)
configs = [("tiny", make_tiny_student(vocab_size, block_size)),
           ("mid", mid),
           ("student", make_student(vocab_size, block_size))]

STEPS = 600
sweep = []
for name, cfg in configs:
    torch.manual_seed(0)
    student = GPT(cfg).to(device)
    train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
                  alpha=best_alpha, temperature=1.0)
    c = evaluate_cloze(student, stoi, itos, facts, device)
    p = student.num_params()
    sweep.append((name, p, c))
    print(f"{name:8s} | {p/1e6:.2f}M params | cloze {c:.1%}")

# %% [markdown]
# ## Plot cloze vs size

# %%
ps = [p for _, p, _ in sweep]
cs = [c for _, _, c in sweep]
plt.figure(figsize=(6, 4))
plt.plot([p / 1e6 for p in ps], cs, "o-")
for name, p, c in sweep:
    plt.annotate(name, (p / 1e6, c), textcoords="offset points", xytext=(5, 5))
plt.xlabel("parameters (millions)")
plt.ylabel("cloze accuracy")
plt.title("How small can the student go?")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("assets/09_size_sweep.png", dpi=120)
print("Saved assets/09_size_sweep.png")

assert sweep[-1][2] >= sweep[0][2], "the largest student should recall at least as well as the smallest"
update_metrics("size_sweep", [[p, c] for _, p, c in sweep])
print("Continue to 10_capstone.")
