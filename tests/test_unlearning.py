"""
tests/test_unlearning.py
==========================
Verifies the unlearning additions: structural unlearn() (Module 4's
memory/expandable_memory.py) correctly deactivates+resets an expert, and
gradient_ascent_unlearn() (memory/unlearning.py) genuinely increases
forget-set loss while keeping retain-set loss roughly stable.
"""

import torch
import torch.nn.functional as F

from config.model_config import load_config, MemoryLevelConfig, MemoryConfig, ModelConfig, SleepLLMConfig
from models.backbone import SleepLLMBackbone
from memory.unlearning import gradient_ascent_unlearn
from evaluation.classifier_head import IntentClassifierHead
from evaluation.toy_intent_data import generate_intent_task


# ---------------------------------------------------------------------
# Structural unlearning (expand()'s counterpart)
# ---------------------------------------------------------------------
def _build_level():
    from memory.expandable_memory import ExpandableMemoryLevel
    cfg = MemoryLevelConfig(
        name="test", update_frequency=100, chunk_length=16,
        num_experts=1, expert_hidden_dim=32, lora_rank=4, max_experts=4,
    )
    return ExpandableMemoryLevel(cfg, hidden_dim=32)


def test_unlearn_deactivates_the_expert():
    level = _build_level()
    new_idx = level.expand()
    assert level.active_mask[new_idx] == True

    level.unlearn(new_idx)
    assert level.active_mask[new_idx] == False
    assert level.num_active_experts == 1  # back to just the original base expert


def test_unlearn_resets_expert_parameters():
    level = _build_level()
    new_idx = level.expand()
    expert = level.experts[new_idx]

    # Train it a bit so its weights are no longer at their zero-init state.
    optimizer = torch.optim.Adam(expert.parameters(), lr=0.1)
    x = torch.randn(2, 5, 32)
    for _ in range(5):
        optimizer.zero_grad()
        out = expert(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

    assert not torch.allclose(expert.B.weight, torch.zeros_like(expert.B.weight))

    level.unlearn(new_idx)
    # After unlearning, B should be back to exactly zero (LoRA-style reset).
    assert torch.allclose(expert.B.weight, torch.zeros_like(expert.B.weight))


def test_unlearn_raises_on_already_inactive_expert():
    level = _build_level()
    try:
        level.unlearn(2)  # slot 2 was never activated
        assert False, "should have raised RuntimeError"
    except RuntimeError:
        pass


def test_unlearned_slot_can_be_expanded_again():
    """A structurally-unlearned slot returns to the available capacity
    pool -- expand() should be able to reuse it."""
    level = _build_level()
    idx1 = level.expand()
    level.unlearn(idx1)
    idx2 = level.expand()
    assert idx2 == idx1  # the freed slot gets reused


# ---------------------------------------------------------------------
# Gradient-ascent unlearning
# ---------------------------------------------------------------------
def _forward_fn(model, inputs):
    return model(inputs)


def test_gradient_ascent_unlearn_increases_forget_loss():
    config = load_config("config/debug_laptop.yaml")
    torch.manual_seed(config.seed)
    backbone = SleepLLMBackbone(config)
    model = IntentClassifierHead(backbone, num_classes=6)

    forget_inputs, forget_labels = generate_intent_task(
        num_classes=2, examples_per_class=8, seq_len=12,
        vocab_size=config.model.vocab_size, class_offset=0, seed=1,
    )
    retain_inputs, retain_labels = generate_intent_task(
        num_classes=2, examples_per_class=8, seq_len=12,
        vocab_size=config.model.vocab_size, class_offset=2, seed=2,
    )

    # First, train normally on BOTH sets so the model actually knows
    # something worth unlearning (otherwise "forgetting" an untrained
    # random model is a meaningless test).
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_inputs = torch.cat([forget_inputs, retain_inputs])
    all_labels = torch.cat([forget_labels, retain_labels])
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(all_inputs), all_labels)
        loss.backward()
        optimizer.step()

    result = gradient_ascent_unlearn(
        model, _forward_fn, forget_inputs, forget_labels,
        retain_inputs, retain_labels, steps=15, lr=1e-3, retain_weight=1.0,
    )

    print(f"\nForget loss: {result.forget_loss_before:.4f} -> {result.forget_loss_after:.4f}")
    print(f"Retain loss: {result.retain_loss_before:.4f} -> {result.retain_loss_after:.4f}")

    # Core claim: unlearning made the model WORSE at the forget set.
    assert result.forget_loss_after > result.forget_loss_before

    # Retain performance shouldn't blow up -- allow some drift (this is
    # gradient ascent, an aggressive method) but it should stay well
    # below the forget set's degradation, proving the retain term is
    # doing its job.
    retain_increase = result.retain_loss_after - result.retain_loss_before
    forget_increase = result.forget_loss_after - result.forget_loss_before
    assert retain_increase < forget_increase
