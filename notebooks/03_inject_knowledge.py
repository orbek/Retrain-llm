# %% [markdown]
# # 03 — Inject Knowledge
# Fine-tune the pretrained teacher on the fact corpus. Watch cloze accuracy jump
# from ~0 to ~100%: the teacher now *knows* the invented facts.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from dataclasses import asdict
import torch
from model import GPT, GPTConfig
from facts import build_factset, render_training_corpus, encode, evaluate_cloze

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
torch.manual_seed(0)

# %% [markdown]
# ## Load the pretrained teacher

# %%
ckpt = torch.load("checkpoints/teacher.pt", weights_only=True)
cfg = GPTConfig(**ckpt["config"])
stoi, itos = ckpt["stoi"], ckpt["itos"]
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model_state"])

facts = build_factset()
before = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Cloze BEFORE injection: {before:.1%}")

# %% [markdown]
# ## The fact corpus (training paraphrases only)

# %%
corpus = render_training_corpus(facts, repeats=40)
fact_data = torch.tensor(encode(corpus, stoi), dtype=torch.long)
print("Fact corpus tokens:", len(fact_data))

def get_fact_batch(batch_size=32):
    ix = torch.randint(len(fact_data) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([fact_data[i:i + cfg.block_size] for i in ix])
    y = torch.stack([fact_data[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)

# %% [markdown]
# ## Fine-tune to inject the facts

# %%
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for it in range(1000):
    xb, yb = get_fact_batch()
    _, loss, _ = model(xb, targets=yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 200 == 0:
        print(f"iter {it:4d} | loss {loss.item():.3f}")

# %% [markdown]
# ## The teacher now knows the facts

# %%
after = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Cloze AFTER injection: {after:.1%}")
assert after >= 0.8, "the teacher should recall most facts after injection"
assert after > before + 0.5, "injection must produce a large recall jump"

# %% [markdown]
# ## Save the injected teacher

# %%
torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/teacher_injected.pt")
print("Saved checkpoints/teacher_injected.pt.")
print("Phase 1 complete: a teacher that knows the facts. Next: distill it down.")
