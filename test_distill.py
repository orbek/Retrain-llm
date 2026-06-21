import torch
import torch.nn.functional as F
from distill import kd_loss, distill_loss, train_distill


def test_kd_loss_zero_when_identical():
    logits = torch.randn(8, 12)
    loss = kd_loss(logits, logits.clone(), temperature=2.0)
    assert loss.item() < 1e-6


def test_kd_loss_positive_when_different():
    s = torch.randn(8, 12)
    t = torch.randn(8, 12)
    assert kd_loss(s, t, temperature=2.0).item() > 0.0


def test_distill_loss_alpha_zero_is_pure_ce():
    s = torch.randn(8, 12)
    t = torch.randn(8, 12)
    y = torch.randint(0, 12, (8,))
    expected = F.cross_entropy(s, y)
    got = distill_loss(s, t, y, alpha=0.0, temperature=2.0)
    assert torch.allclose(got, expected, atol=1e-5)


def test_distill_loss_alpha_one_is_pure_kd():
    s = torch.randn(8, 12)
    t = torch.randn(8, 12)
    y = torch.randint(0, 12, (8,))
    expected = kd_loss(s, t, temperature=2.0)
    got = distill_loss(s, t, y, alpha=1.0, temperature=2.0)
    assert torch.allclose(got, expected, atol=1e-5)


def test_train_distill_runs_and_returns_student():
    from model import GPT, NANO_CONFIG
    torch.manual_seed(0)
    teacher = GPT(NANO_CONFIG).eval()
    student = GPT(NANO_CONFIG)
    bs, T = 4, 16

    def get_batch():
        x = torch.randint(0, NANO_CONFIG.vocab_size, (bs, T))
        return x, x.clone()

    out = train_distill(student, teacher, get_batch, steps=3, lr=1e-3,
                        device=torch.device("cpu"), alpha=0.5, temperature=2.0)
    assert isinstance(out, GPT)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
