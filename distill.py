"""Knowledge-distillation toolkit: soft-label losses and a teacher->student
training loop. Teacher and student must share a vocabulary so their logits are
comparable."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, temperature=2.0):
    """Temperature-scaled KL divergence between teacher and student distributions.
    Multiplying by T^2 keeps gradient magnitudes comparable across temperatures."""
    s = student_logits.reshape(-1, student_logits.size(-1))
    t = teacher_logits.reshape(-1, teacher_logits.size(-1))
    log_p_student = F.log_softmax(s / temperature, dim=-1)
    p_teacher = F.softmax(t / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)


def distill_loss(student_logits, teacher_logits, targets, alpha=0.5, temperature=2.0):
    """Blend soft-label KD with ground-truth cross-entropy.
    alpha=1.0 -> pure KD; alpha=0.0 -> pure hard-label CE."""
    kd = kd_loss(student_logits, teacher_logits, temperature)
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)),
                         targets.reshape(-1))
    return alpha * kd + (1.0 - alpha) * ce


class FeatureProjector(nn.Module):
    """Project each student hidden layer into the teacher's hidden dim and match it
    (MSE) against an evenly-spaced teacher layer. One Linear per student layer."""

    def __init__(self, n_student_layers, n_teacher_layers, student_dim, teacher_dim):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Linear(student_dim, teacher_dim, bias=False)
             for _ in range(n_student_layers)])
        if n_student_layers == 1:
            self.layer_map = [n_teacher_layers - 1]
        else:
            self.layer_map = [round(i * (n_teacher_layers - 1) / (n_student_layers - 1))
                              for i in range(n_student_layers)]

    def forward(self, student_hiddens, teacher_hiddens):
        total = 0.0
        for i, proj in enumerate(self.projs):
            s = proj(student_hiddens[i])
            t = teacher_hiddens[self.layer_map[i]]
            total = total + F.mse_loss(s, t)
        return total / len(self.projs)


def train_distill(student, teacher, get_batch, steps, lr, device,
                  alpha=0.5, temperature=2.0, feature_weight=0.0, projector=None,
                  eval_fn=None, log_every=100):
    """Train `student` to match the frozen `teacher`. With feature_weight>0 and a
    projector, also matches hidden states. Returns the trained student."""
    teacher.eval()
    student.train()
    params = list(student.parameters())
    if projector is not None:
        projector.to(device)
        params += list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr)
    use_features = feature_weight > 0.0 and projector is not None
    for step in range(steps):
        xb, yb = get_batch()
        xb, yb = xb.to(device), yb.to(device)   # self-sufficient placement (idempotent if already on device)
        if use_features:
            s_logits, _, _, s_hid = student(xb, return_hidden=True)
            with torch.no_grad():
                t_logits, _, _, t_hid = teacher(xb, return_hidden=True)
            loss = distill_loss(s_logits, t_logits, yb, alpha, temperature)
            loss = loss + feature_weight * projector(s_hid, t_hid)
        else:
            s_logits, _, _ = student(xb)
            with torch.no_grad():
                t_logits, _, _ = teacher(xb)
            loss = distill_loss(s_logits, t_logits, yb, alpha, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_fn is not None and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:4d} | loss {loss.item():.3f} | {eval_fn()}")
    return student
