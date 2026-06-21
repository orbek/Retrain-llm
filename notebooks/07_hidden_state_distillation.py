# %% [markdown]
# # 07 — Hidden-State (Feature) Distillation
# Beyond final logits, match the teacher's intermediate layers. A small projector
# maps each student layer into the teacher's hidden dim.

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
