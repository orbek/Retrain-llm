import torch
from model import (GPT, GPTConfig, NANO_CONFIG,
                   make_teacher, make_student, make_tiny_student)

torch.manual_seed(0)


# --- vendored tests (copy verbatim from train-llm/test_model.py) ---
def test_forward_shapes():
    cfg = NANO_CONFIG
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = m(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.dim() == 0 and loss.item() > 0


def test_causal_no_future_leak():
    cfg = NANO_CONFIG
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base, _, _ = m(idx)
        changed = idx.clone(); changed[0, -1] = (changed[0, -1] + 1) % cfg.vocab_size
        alt, _, _ = m(changed)
    # changing the LAST token must not change logits at earlier positions
    assert torch.allclose(base[:, :-1], alt[:, :-1], atol=1e-5)


def test_gqa_reduces_to_mha():
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=4,
                    n_kv_head=4, n_embd=64, dropout=0.0)   # n_kv_head == n_head
    m = GPT(cfg)
    assert m.blocks[0].attn.n_rep == 1
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _ = m(idx)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_weight_tying():
    m = GPT(NANO_CONFIG)
    assert m.tok_emb.weight.data_ptr() == m.lm_head.weight.data_ptr()


def test_generate_length():
    cfg = NANO_CONFIG
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = m.generate(idx, max_new_tokens=10, temperature=1.0, top_k=5)
    assert out.shape == (1, 14)


def test_kv_cache_matches_no_cache():
    cfg = NANO_CONFIG
    m = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    a = m.generate(idx.clone(), max_new_tokens=20, temperature=0.0, use_cache=True)
    b = m.generate(idx.clone(), max_new_tokens=20, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "cached and non-cached greedy generation must match"


def test_get_device():
    import model
    orig_cuda = torch.cuda.is_available
    orig_mps = torch.backends.mps.is_available
    try:
        torch.cuda.is_available = lambda: True
        torch.backends.mps.is_available = lambda: True
        assert model.get_device().type == "cuda"          # CUDA wins
        torch.cuda.is_available = lambda: False
        torch.backends.mps.is_available = lambda: True
        assert model.get_device().type == "mps"           # MPS when no CUDA
        torch.backends.mps.is_available = lambda: False
        assert model.get_device().type == "cpu"           # CPU otherwise
    finally:
        torch.cuda.is_available = orig_cuda
        torch.backends.mps.is_available = orig_mps


# --- new tests for Phase 1 additions ---
def test_return_hidden_shapes():
    cfg = NANO_CONFIG
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    out = m(idx, targets=idx, return_hidden=True)
    assert len(out) == 4, "return_hidden must return a 4-tuple"
    logits, loss, presents, hiddens = out
    assert len(hiddens) == cfg.n_layer
    for h in hiddens:
        assert h.shape == (2, 16, cfg.n_embd)


def test_presets_are_distillation_compatible():
    t_cfg = make_teacher(80)
    s_cfg = make_student(80)
    tiny_cfg = make_tiny_student(80)
    assert t_cfg.vocab_size == s_cfg.vocab_size == tiny_cfg.vocab_size == 80
    assert t_cfg.block_size == s_cfg.block_size == tiny_cfg.block_size
    t, s, tiny = GPT(t_cfg), GPT(s_cfg), GPT(tiny_cfg)
    assert t.num_params() > s.num_params() > tiny.num_params()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
