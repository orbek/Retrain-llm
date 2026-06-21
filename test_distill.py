import torch
import torch.nn.functional as F
from distill import kd_loss, distill_loss, train_distill, FeatureProjector


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


def test_feature_projector_layer_map_spans_teacher():
    proj = FeatureProjector(n_student_layers=4, n_teacher_layers=6,
                            student_dim=128, teacher_dim=384)
    assert len(proj.projs) == 4
    assert proj.layer_map[0] == 0
    assert proj.layer_map[-1] == 5


def test_feature_projector_returns_scalar_loss():
    proj = FeatureProjector(2, 6, 64, 384)
    s_hid = [torch.randn(2, 5, 64) for _ in range(2)]
    t_hid = [torch.randn(2, 5, 384) for _ in range(6)]
    loss = proj(s_hid, t_hid)
    assert loss.dim() == 0 and loss.item() >= 0.0


def test_train_distill_with_features_runs():
    from model import GPT, NANO_CONFIG
    torch.manual_seed(0)
    teacher = GPT(NANO_CONFIG).eval()
    student = GPT(NANO_CONFIG)
    proj = FeatureProjector(NANO_CONFIG.n_layer, NANO_CONFIG.n_layer,
                            NANO_CONFIG.n_embd, NANO_CONFIG.n_embd)

    def get_batch():
        x = torch.randint(0, NANO_CONFIG.vocab_size, (4, 16))
        return x, x.clone()

    out = train_distill(student, teacher, get_batch, steps=2, lr=1e-3,
                        device=torch.device("cpu"), alpha=0.5, temperature=2.0,
                        feature_weight=0.1, projector=proj)
    assert isinstance(out, GPT)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
    print("ALL TESTS PASSED")
