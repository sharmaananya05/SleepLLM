"""
dreaming package
==================
Module 7: Dreaming (paper Section 3.4) -- synthetic data generation with
random-expert injection (dream_generator.py), loss-based dream selection
(dream_selection.py -- a documented cheaper proxy for the paper's
gradient-norm scoring), and isolated SFT + binary reward + simplified
ReSTEM filtering (self_improvement.py).
"""

from dreaming.dream_generator import generate_dreams, random_expert_injection
from dreaming.dream_selection import compute_dream_importance_scores, select_dreams
from dreaming.self_improvement import isolated_sft_and_reward, restem_filter, DreamOutcome

__all__ = [
    "generate_dreams",
    "random_expert_injection",
    "compute_dream_importance_scores",
    "select_dreams",
    "isolated_sft_and_reward",
    "restem_filter",
    "DreamOutcome",
]
