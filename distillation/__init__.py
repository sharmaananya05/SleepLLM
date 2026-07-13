"""
distillation package
======================
Module 6: Knowledge Seeding (paper Section 3.3) -- the on-policy
distillation objective (losses.py), the Learning-to-Imitate reward
(rewards.py), and the orchestration that freezes the right parameters
and combines both into the paper's final loss (knowledge_seeding.py).
"""

from distillation.losses import reverse_kl_divergence, on_policy_distillation_loss
from distillation.rewards import r_abs, r_sem, combined_lti_reward
from distillation.knowledge_seeding import freeze_all_except_expert, compute_knowledge_seeding_loss

__all__ = [
    "reverse_kl_divergence",
    "on_policy_distillation_loss",
    "r_abs",
    "r_sem",
    "combined_lti_reward",
    "freeze_all_except_expert",
    "compute_knowledge_seeding_loss",
]
