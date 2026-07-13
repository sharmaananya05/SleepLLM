"""
distillation/knowledge_seeding.py
====================================
Orchestrates Knowledge Seeding (paper Section 3.3): freezing the right
parameters, and combining the on-policy distillation loss (losses.py)
with the Learning-to-Imitate reward (rewards.py) into the paper's final
objective:

    L_KS(theta, theta_exp) =
        E_x~D [ (1-alpha) * E_y~LM_theta_exp[r(y)]
                - alpha * E_y~LM_theta_exp[ D(LM_theta || LM_theta_exp)(y|x) ] ]

Paper quote motivating WHY we freeze aggressively:
    "we freeze all the parameters in the student model and only updates
    the expanded parameters. This ensures that the transferred knowledge
    does not interfere with the old knowledge, causing catastrophic
    forgetting."
"""

import torch
import torch.nn as nn

from memory.expandable_memory import ExpandableMemoryLevel


def freeze_all_except_expert(model: nn.Module, level: ExpandableMemoryLevel, expert_idx: int) -> None:
    """
    Sets requires_grad=False on EVERY parameter in the model, then
    requires_grad=True ONLY on the specific newly-activated expert's
    parameters -- directly implementing the paper's freezing requirement
    quoted above.

    Args:
        model: the full SleepLLMBackbone (or any nn.Module containing
            `level` somewhere inside it).
        level: the specific ExpandableMemoryLevel whose expert is being
            trained right now (identified by the caller via which block/
            level a consolidation event targets -- see sleep/scheduler.py).
        expert_idx: the index returned by level.expand() -- the expert
            slot that should receive gradient updates.

    Why freeze EVERYTHING first, then selectively unfreeze, rather than
    the reverse (leave everything trainable, freeze what we don't want)?
    Freeze-all-then-unfreeze is the SAFER default: if we ever add a new
    submodule to the model later and forget to explicitly freeze it, the
    "freeze everything first" pattern means it defaults to FROZEN (safe:
    no accidental extra training), whereas the reverse pattern would
    default to TRAINABLE (unsafe: silently disturbs old knowledge).

    Time complexity: O(P) where P = total parameter TENSORS (not
    element count) in the model -- a few dozen for our tiny backbone,
    negligible cost, called once per consolidation event (not per step).
    """
    for param in model.parameters():
        param.requires_grad = False

    target_expert = level.experts[expert_idx]
    for param in target_expert.parameters():
        param.requires_grad = True


def compute_knowledge_seeding_loss(
    reward: float,
    log_prob_sum: torch.Tensor,
    distillation_kl: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combines the LTI reward term with the distillation divergence term
    into ONE scalar loss ready for .backward(), implementing the paper's
    L_KS objective (Section 3.3).

    Args:
        reward: scalar float from rewards.combined_lti_reward() -- how
            well the student's generated continuation matched the
            teacher's, this step.
        log_prob_sum: scalar tensor -- sum of log p_student(token) over
            the student's generated sequence, WITH gradient (i.e.
            computed from student_logits that still require_grad).
            This is what lets the reward term's gradient flow back into
            the newly-activated expert's parameters.
        distillation_kl: scalar tensor from
            losses.on_policy_distillation_loss() -- the divergence term,
            already carrying gradient.
        alpha: paper's alpha in [0,1], "controls the strength of the
            distillation compared to the LTI objective." NOT given a
            specific value by the paper -- default 0.5 (equal weighting)
            as the most neutral choice absent guidance, same reasoning
            as lambda/gamma elsewhere in this module.

    Returns:
        Scalar loss tensor.

    Why `-reward * log_prob_sum` (REINFORCE / score-function estimator):
    The paper's objective wants to MAXIMIZE E_y~student[reward(y)]. The
    standard way to get a gradient for "expected reward under a sampling
    distribution" without differentiating through the (non-differentiable)
    sampling/argmax operation is the REINFORCE trick:
        grad_theta E_y~pi_theta[r(y)] ~= r(y) * grad_theta log pi_theta(y)
    So MINIMIZING `-reward * log_prob_sum` via gradient descent has the
    same effect as maximizing expected reward via gradient ascent. "The
    paper does not specify this implementation detail" -- it doesn't
    name a specific policy-gradient algorithm/variance-reduction scheme
    (e.g. no baseline subtraction here); we use the simplest, standard
    REINFORCE estimator since Module 7 (Dreaming) has more sophisticated
    RL (ReSTEM) and duplicating that machinery here would be premature.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    reward_term = -reward * log_prob_sum  # REINFORCE: negative for gradient DESCENT
    loss = (1 - alpha) * reward_term + alpha * distillation_kl
    return loss
