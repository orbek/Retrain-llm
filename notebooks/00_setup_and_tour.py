# %% [markdown]
# # 00 — Setup & Tour
#
# This course **injects new knowledge into a teacher model and distills it into a
# much smaller student** — from scratch in PyTorch. This first notebook checks your
# environment and maps the journey.

# %%
import os
# Walk up to the repo root so relative paths (data/, assets/, checkpoints/) resolve
# no matter which directory the notebook kernel was launched from.
while not os.path.exists("requirements.txt"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir(parent)
print("Working directory:", os.getcwd())

# %%
import sys
import platform
import torch

print("Python  :", sys.version.split()[0])
print("Platform:", platform.platform())
print("PyTorch :", torch.__version__)

# %% [markdown]
# ## Picking a device
# CUDA (NVIDIA) → MPS (Apple Silicon) → CPU. The same code runs anywhere.

# %%
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
print("Using device:", device)

# %%
x = torch.randn(3, 3, device=device)
y = x @ x
assert y.shape == (3, 3)
print("Matmul on", device, "ok.")

# %% [markdown]
# ## The journey ahead
#
# | # | Notebook | Idea |
# |---|----------|------|
# | 00 | Setup & tour | you are here |
# | 01 | Fact-set & cloze eval | a synthetic knowledge base + the recall metric |
# | 02 | Train the teacher | pretrain a broad GPT; measure perplexity |
# | 03 | Inject knowledge | fine-tune the teacher on the facts (cloze 0 → ~100%) |
# | 04 | Student baseline | a small model on the facts alone fails to recall |
# | 05 | Logit distillation | soft labels + temperature beat the baseline |
# | 06 | Mixing GT + KD | blend cross-entropy with distillation |
# | 07 | Hidden-state distillation | feature matching |
# | 08 | Synthetic-data augmentation | the teacher generates more data |
# | 09 | Size sweep | how small can the student get? |
# | 10 | Capstone | the best combined recipe |
# | 11 | Recap | what transferred, and why |

# %%
print("Environment looks good. Continue to 01_factset_and_cloze.")
