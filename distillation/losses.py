"""
distillation/losses.py
========================
The on-policy distillation objective from paper Section 3.3, built on
Generalized Knowledge Distillation (Agarwal et al. 2024):

    L(theta, theta_exp) =
        (1-lambda) * E_(x,y)~D [ F(LM_theta || LM_theta_exp)(y|x) ]
        + lambda * E_x~D [ E_y~LM_theta_exp(.|x) [ F(LM_theta || LM_theta_exp)(y|x) ] ]

Where:
- LM_theta = the TEACHER (frozen, pre-expansion model state)
- LM_theta_exp = the STUDENT (post-expansion model, only new params train)
- F(.||.) = a divergence between teacher and student output distributions
- lambda in [0,1]: fraction of "on-policy" (student-generated) vs
  "off-policy" (dataset/teacher-generated) samples used for the
  divergence term.

"The paper does not specify this implementation detail": which specific
divergence F to use. GKD (Agarwal et al. 2024), which this paper builds
on, uses (generalized) KL divergence, so we use REVERSE KL
(KL(student || teacher)) specifically -- reverse KL is GKD's recommended
choice because it is "mode-seeking" (encourages the student to
concentrate on the teacher's high-probability outputs rather than
spreading mass thinly over all of them), which suits a small student
head learning from a fixed teacher.
"""

import torch
import torch.nn.functional as F


def reverse_kl_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """
    Per-token reverse KL: KL(student || teacher) = sum_v p_student(v) *
    log(p_student(v) / p_teacher(v)), averaged over the batch and
    sequence.

    Args:
        student_logits: (batch, seq_len, vocab_size) -- UNNORMALIZED
            scores from LM_theta_exp (student).
        teacher_logits: (batch, seq_len, vocab_size) -- from LM_theta
            (teacher). MUST be detached / from a torch.no_grad() context
            upstream -- gradients should never flow into the frozen
            teacher (enforced by the caller in knowledge_seeding.py, not
            re-checked here to keep this function a pure, reusable loss).

    Returns:
        Scalar tensor: the mean reverse-KL loss.

    Time complexity: O(batch * seq_len * vocab_size) -- dominated by the
    softmax/log-softmax over the vocabulary, same order as a normal
    cross-entropy loss computation.

    Why reverse KL specifically (student log-prob first): torch's
    F.kl_div(input, target) computes KL(target || input) when
    log_target=False and input is log-probabilities -- so to get
    KL(student || teacher) we pass log_softmax(teacher) as `input`... 
    Actually, to avoid this classic footgun (torch's kl_div argument
    order is famously confusing), we compute it MANUALLY and explicitly
    below rather than relying on F.kl_div's argument semantics -- this is
    slower by a constant factor but far less likely to silently compute
    the WRONG divergence, which is an easy, hard-to-notice bug.
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_probs = student_log_probs.exp()
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # KL(student || teacher) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
    per_token_kl = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)
    # per_token_kl: (batch, seq_len)

    return per_token_kl.mean()


def on_policy_distillation_loss(
    student_logits_on_data: torch.Tensor,
    teacher_logits_on_data: torch.Tensor,
    student_logits_on_policy: torch.Tensor,
    teacher_logits_on_policy: torch.Tensor,
    lam: float = 0.5,
) -> torch.Tensor:
    """
    The full on-policy distillation objective (paper Section 3.3
    equation), combining an off-policy term (evaluated on dataset
    samples) with an on-policy term (evaluated on the student's OWN
    generated samples).

    Args:
        student_logits_on_data / teacher_logits_on_data: both models'
            logits evaluated on the SAME dataset-sampled sequences
            (x, y) ~ D. This is the (1-lambda) term.
        student_logits_on_policy / teacher_logits_on_policy: both
            models' logits evaluated on sequences the STUDENT itself
            generated (y ~ LM_theta_exp(.|x)). This is the lambda term.
        lam: lambda in the paper -- fraction of on-policy weighting.
            NOT given a specific numeric value by the paper (it's left
            as a tunable hyperparameter throughout); we default to 0.5
            as a neutral midpoint between pure off-policy (lam=0, plain
            distillation on fixed data) and pure on-policy (lam=1).

    Returns:
        Scalar loss tensor, ready for .backward().
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    off_policy_term = reverse_kl_divergence(student_logits_on_data, teacher_logits_on_data)
    on_policy_term = reverse_kl_divergence(student_logits_on_policy, teacher_logits_on_policy)

    return (1 - lam) * off_policy_term + lam * on_policy_term
