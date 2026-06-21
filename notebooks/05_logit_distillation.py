# %% [markdown]
# # 05 — Logit Distillation (KL + Temperature)
# Same data, same student size as the baseline — but the student now learns from
# the teacher's full soft distribution instead of one-hot labels.

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
# ## Load the (frozen) injected teacher

# %%
ckpt = torch.load("checkpoints/teacher_injected.pt", weights_only=True)
stoi, itos = ckpt["stoi"], ckpt["itos"]
vocab_size = len(stoi)
teacher = GPT(GPTConfig(**ckpt["config"])).to(device)
teacher.load_state_dict(ckpt["model_state"])
teacher.eval()
facts = build_factset()

# %% [markdown]
# ## Fact corpus + batching (same as the baseline)

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
# ## Distill (pure KD: alpha=1.0) — identical seed/config/steps to the baseline
# We use temperature 1.0, not the textbook 2.0: for crisp factual targets a softer
# teacher distribution dilutes the answer signal and actually underperforms hard
# labels (we measured T=2 losing badly). At T=1 the soft labels help. We also use a
# modest fixed budget (400 steps) so the baseline leaves headroom — distillation's
# edge here is sample efficiency, which vanishes once both saturate.

# %%
STEPS = 400
torch.manual_seed(0)
student = GPT(make_student(vocab_size, block_size=block_size)).to(device)
train_distill(student, teacher, get_batch, steps=STEPS, lr=3e-4, device=device,
              alpha=1.0, temperature=1.0,
              eval_fn=lambda: f"cloze {evaluate_cloze(student, stoi, itos, facts, device):.1%}",
              log_every=300)

# %% [markdown]
# ## Did it beat the baseline?

# %%
kd_cloze = evaluate_cloze(student, stoi, itos, facts, device)
metrics = json.load(open("assets/distill_metrics.json"))
baseline = metrics["baseline_hard_label"]
print(f"baseline {baseline:.1%}  ->  logit-KD {kd_cloze:.1%}")
assert kd_cloze >= baseline, "logit distillation should not underperform hard labels"
update_metrics("logit_kd", kd_cloze)

torch.save({"model_state": student.state_dict(),
            "config": dataclasses.asdict(student.config),
            "stoi": stoi, "itos": itos}, "checkpoints/student_kd.pt")
print("Saved checkpoints/student_kd.pt. Continue to 06_alpha_mixing.")
