"""Knowledge-distillation toolkit: soft-label losses and a teacher->student
training loop. Teacher and student must share a vocabulary so their logits are
comparable."""
import torch
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


def train_distill(student, teacher, get_batch, steps, lr, device,
                  alpha=0.5, temperature=2.0, eval_fn=None, log_every=100):
    """Train `student` to match the frozen `teacher` (plus ground-truth via alpha).
    Returns the trained student."""
    teacher.eval()
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_batch()
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
