"""Source-of-truth model for the course: a small Llama-style decoder-only
transformer (GQA + RoPE + RMSNorm + SwiGLU + weight tying + KV-cache).
Notebook 05 explains every piece; notebooks 06-07 import from here."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2        # GQA: #key/value heads (must divide n_head)
    n_embd: int = 384
    dropout: float = 0.2
    rope_theta: float = 10000.0


# The "full" ~9.4M model (committed training run) and a tiny debug preset.
DEFAULT_CONFIG = GPTConfig()
NANO_CONFIG = GPTConfig(block_size=64, n_layer=3, n_head=4, n_kv_head=2,
                        n_embd=128, dropout=0.0)


def make_teacher(vocab_size, block_size=256):
    """The large teacher (~9-10M params)."""
    return GPTConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=6, n_head=6, n_kv_head=2, n_embd=384, dropout=0.2)


def make_student(vocab_size, block_size=256):
    """The default small student (~1M params)."""
    return GPTConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=4, n_head=4, n_kv_head=2, n_embd=128, dropout=0.1)


def make_tiny_student(vocab_size, block_size=256):
    """An extra-small student for the size sweep."""
    return GPTConfig(vocab_size=vocab_size, block_size=block_size,
                     n_layer=2, n_head=4, n_kv_head=2, n_embd=64, dropout=0.0)


def get_device():
    """Best available compute engine: CUDA (NVIDIA GPU) -> MPS (Apple Silicon) -> CPU.

    Works seamlessly on Windows/Linux (CUDA or CPU) and macOS (MPS or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_rope(head_dim, seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim)
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 64 * (((int(8 / 3 * config.n_embd)) + 63) // 64)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_head // config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, cos, sin, kv_cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_k, past_v = kv_cache
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        if self.n_rep > 1:                              # GQA: expand kv heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        is_causal = past_len == 0                       # prefill is causal; single-step isn't
        if not is_causal:
            assert T == 1, "cached generation feeds one token at a time"
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y)), present


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin, kv_cache=None):
        a, present = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight       # weight tying
        head_dim = config.n_embd // config.n_head
        cos, sin = build_rope(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        # subtract tied head (shares storage with tok_emb)
        n = sum(p.numel() for p in self.parameters())
        return n - self.lm_head.weight.numel()

    def forward(self, idx, targets=None, kv_caches=None, pos=0, return_hidden=False):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[pos:pos + T].to(x.device)
        sin = self.rope_sin[pos:pos + T].to(x.device)
        presents = []
        hiddens = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, present = block(x, cos, sin, cache)
            presents.append(present)
            if return_hidden:
                hiddens.append(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, presents, hiddens
        return logits, loss, presents

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 top_p=None, use_cache=True):
        was_training = self.training
        self.eval()
        if use_cache:
            assert idx.size(1) + max_new_tokens <= self.config.block_size, \
                "cached generation supports up to block_size total tokens"
            kv_caches = [(None, None) for _ in range(self.config.n_layer)]
            logits, _, presents = self(idx, kv_caches=kv_caches, pos=0)
            kv_caches = presents
            pos = idx.size(1)
            for _ in range(max_new_tokens):
                nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, nxt], dim=1)
                logits, _, presents = self(nxt, kv_caches=kv_caches, pos=pos)
                kv_caches = presents
                pos += 1
        else:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _, _ = self(idx_cond)
                nxt = sample(logits[:, -1, :], temperature, top_k, top_p)
                idx = torch.cat([idx, nxt], dim=1)
        if was_training:
            self.train()
        return idx


def sample(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature == 0.0:                              # greedy / deterministic
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p is not None:
        s_logits, s_idx = torch.sort(logits, descending=True)
        probs = F.softmax(s_logits, dim=-1).cumsum(dim=-1)
        remove = probs - F.softmax(s_logits, dim=-1) > top_p
        s_logits = s_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, s_idx, s_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
