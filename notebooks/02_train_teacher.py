# %% [markdown]
# # 02 — Train the Teacher
# Pretrain a broad GPT on TinyShakespeare so it has fluent language. It will NOT
# know our invented facts yet — we confirm that with the cloze metric.

# %%
import os, sys
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import math
import urllib.request
from dataclasses import asdict
import torch
from model import GPT, make_teacher
from facts import (build_factset, factset_text, build_vocab, encode,
                   evaluate_cloze)

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cpu"))
torch.manual_seed(0)
print("Device:", device)

# %% [markdown]
# ## The broad corpus (auto-download)

# %%
os.makedirs("data", exist_ok=True)
path = "data/shakespeare.txt"
if not os.path.exists(path):
    url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
           "master/data/tinyshakespeare/input.txt")
    urllib.request.urlretrieve(url, path)
with open(path) as fh:
    text = fh.read()
print("Corpus chars:", len(text))

# %% [markdown]
# ## Shared vocab over the union of corpus + fact-set

# %%
facts = build_factset()
stoi, itos = build_vocab(text, factset_text(facts))
vocab_size = len(stoi)
print("Vocab size:", vocab_size)

data = torch.tensor(encode(text, stoi), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# %% [markdown]
# ## Model + batching

# %%
cfg = make_teacher(vocab_size, block_size=256)
model = GPT(cfg).to(device)
print(f"Teacher params: {model.num_params()/1e6:.1f}M")

def get_batch(split, batch_size=32):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + cfg.block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def val_loss(iters=20):
    model.eval()
    losses = [model(*get_batch("val"))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)

# %% [markdown]
# ## Train
# `max_iters` is capped so this notebook renders quickly. Raise it for a sharper
# teacher; you can also run this notebook in the background.

# %%
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
max_iters = 2000
for it in range(max_iters):
    xb, yb = get_batch("train")
    _, loss, _ = model(xb, targets=yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if it % 500 == 0 or it == max_iters - 1:
        vl = val_loss()
        print(f"iter {it:4d} | train {loss.item():.3f} | val {vl:.3f} "
              f"| ppl {math.exp(vl):.2f}")

final_ppl = math.exp(val_loss())
print(f"Final val perplexity: {final_ppl:.2f}")
assert final_ppl < vocab_size, "teacher must beat the uniform-random baseline"

# %% [markdown]
# ## The teacher does NOT know the facts yet

# %%
cloze = evaluate_cloze(model, stoi, itos, facts, device)
print(f"Teacher cloze accuracy (pre-injection): {cloze:.1%}")
assert cloze < 0.2, "a Shakespeare-only teacher should not know invented facts"

# %% [markdown]
# ## Save the checkpoint

# %%
os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state": model.state_dict(), "config": asdict(cfg),
            "stoi": stoi, "itos": itos}, "checkpoints/teacher.pt")
print("Saved checkpoints/teacher.pt. Continue to 03_inject_knowledge.")
