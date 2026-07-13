"""
tests/test_memory.py
======================
Verifies Module 4: expert masking and expansion behave exactly per the
paper's "Note on Implementation" (Section 3.2) -- inactive slots get zero
routing mass and zero gradient, and expand() correctly activates exactly
one new low-rank expert without disturbing existing ones.
"""

import torch

from config.model_config import MemoryLevelConfig
from memory.expandable_memory import ExpandableMemoryLevel


def _build_level(num_experts=1, max_experts=4, lora_rank=4, hidden_dim=32):
    cfg = MemoryLevelConfig(
        name="test_level",
        update_frequency=100,
        chunk_length=16,
        num_experts=num_experts,
        expert_hidden_dim=32,
        lora_rank=lora_rank,
        max_experts=max_experts,
    )
    level = ExpandableMemoryLevel(cfg, hidden_dim=hidden_dim)
    return level, cfg


def test_initial_active_count_matches_config():
    level, cfg = _build_level(num_experts=1, max_experts=4)
    assert level.num_active_experts == 1
    # only slot 0 should be marked active
    assert level.active_mask.tolist() == [True, False, False, False]


def test_forward_pass_works_before_expansion():
    level, _ = _build_level()
    x = torch.randn(2, 8, 32)
    out = level(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_expand_activates_exactly_one_new_slot():
    level, _ = _build_level(num_experts=1, max_experts=4)
    new_idx = level.expand()
    assert new_idx == 1
    assert level.num_active_experts == 2
    assert level.active_mask.tolist() == [True, True, False, False]


def test_expand_raises_when_at_max_capacity():
    level, _ = _build_level(num_experts=1, max_experts=2)
    level.expand()  # activates slot 1 -> now full
    assert level.num_active_experts == 2
    try:
        level.expand()
        assert False, "expand() should have raised RuntimeError at max capacity"
    except RuntimeError:
        pass  # expected


def test_inactive_experts_receive_no_gradient():
    """
    Core correctness test for the paper's masking requirement: an expert
    that hasn't been activated yet must receive ZERO gradient from a
    forward+backward pass -- proving it truly has no influence on the
    model until expand() is called.
    """
    level, _ = _build_level(num_experts=1, max_experts=4)
    x = torch.randn(4, 8, 32, requires_grad=True)

    out = level(x)
    out.sum().backward()

    # Slot 0 (active) should have received SOME gradient.
    active_expert = level.experts[0]
    assert active_expert.fc1.weight.grad is not None
    assert active_expert.fc1.weight.grad.abs().sum() > 0

    # Slots 1-3 (inactive) must have NO gradient at all -- they were
    # never even included in the forward computation graph.
    for inactive_idx in [1, 2, 3]:
        inactive_expert = level.experts[inactive_idx]
        assert inactive_expert.A.weight.grad is None, (
            f"Slot {inactive_idx} should be inactive and receive no gradient, "
            "but it did -- masking is leaking!"
        )


def test_newly_expanded_expert_starts_as_zero_contribution():
    """
    LowRankExpert initializes B to zero (see memory/lora_expert.py), so a
    freshly-activated expert should compute f(x) = 0 -- meaning
    activating it must not suddenly change the model's existing outputs
    for tokens that get routed to it, until it's actually trained.
    """
    level, _ = _build_level(num_experts=1, max_experts=4, hidden_dim=16)
    new_idx = level.expand()
    new_expert = level.experts[new_idx]

    x = torch.randn(3, 5, 16)
    out = new_expert(x)
    assert torch.allclose(out, torch.zeros_like(out)), (
        "Newly expanded LowRankExpert should output exactly zero (B initialized "
        "to zero), so activation doesn't perturb existing model behavior."
    )
