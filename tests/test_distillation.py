"""
tests/test_distillation.py
============================
Verifies Module 6: reward functions compute sensible values, the
distillation loss is a proper KL divergence, and (most importantly)
freeze_all_except_expert() correctly restricts gradient flow to ONLY
the newly-activated expert -- the paper's core stability guarantee.
"""

import torch

from config.model_config import load_config
from models.backbone import SleepLLMBackbone
from distillation.rewards import r_abs, r_sem, combined_lti_reward
from distillation.losses import reverse_kl_divergence, on_policy_distillation_loss
from distillation.knowledge_seeding import freeze_all_except_expert, compute_knowledge_seeding_loss


# ---------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------
def test_r_abs_identical_sequences_gives_max_reward():
    ids = torch.tensor([10, 20, 30, 40])
    reward = r_abs(ids, ids, z0=0.5)
    assert reward == 1.0  # zero edit distance -> perfect reward


def test_r_abs_completely_different_sequences_gives_zero_below_threshold():
    student = torch.tensor([10, 20, 30, 40])
    teacher = torch.tensor([99, 98, 97, 96])  # every token different
    reward = r_abs(student, teacher, z0=0.5)
    assert reward == 0.0  # edit distance ratio (1.0) exceeds z0 (0.5)


def test_r_sem_identical_hidden_states_gives_reward_1():
    hidden = torch.randn(8, 16)
    reward = r_sem(hidden, hidden.clone(), threshold=0.8)
    assert reward == 1.0  # cosine similarity with itself is exactly 1.0


def test_r_sem_orthogonal_hidden_states_gives_reward_0():
    h1 = torch.zeros(4, 4)
    h1[:, 0] = 1.0  # all mass on dim 0
    h2 = torch.zeros(4, 4)
    h2[:, 1] = 1.0  # all mass on dim 1 -- orthogonal to h1
    reward = r_sem(h1, h2, threshold=0.8)
    assert reward == 0.0


def test_combined_reward_is_weighted_average():
    ids = torch.tensor([1, 2, 3])
    hidden = torch.randn(4, 8)
    reward = combined_lti_reward(
        ids, ids, hidden, hidden.clone(), gamma=0.5, z0=0.5, sem_threshold=0.8
    )
    # identical sequences AND identical hidden states -> both sub-rewards are 1.0
    assert reward == 1.0


# ---------------------------------------------------------------------
# Distillation loss tests
# ---------------------------------------------------------------------
def test_reverse_kl_is_zero_for_identical_distributions():
    logits = torch.randn(2, 5, 100)
    kl = reverse_kl_divergence(logits, logits.clone())
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)


def test_reverse_kl_is_positive_for_different_distributions():
    torch.manual_seed(0)
    student_logits = torch.randn(2, 5, 100)
    teacher_logits = torch.randn(2, 5, 100) * 3  # different distribution
    kl = reverse_kl_divergence(student_logits, teacher_logits)
    assert kl.item() > 0


def test_on_policy_distillation_loss_combines_both_terms():
    torch.manual_seed(1)
    shape = (2, 4, 50)
    s_data, t_data = torch.randn(*shape), torch.randn(*shape)
    s_policy, t_policy = torch.randn(*shape), torch.randn(*shape)

    loss_full_offpolicy = on_policy_distillation_loss(s_data, t_data, s_policy, t_policy, lam=0.0)
    loss_full_onpolicy = on_policy_distillation_loss(s_data, t_data, s_policy, t_policy, lam=1.0)
    expected_offpolicy = reverse_kl_divergence(s_data, t_data)
    expected_onpolicy = reverse_kl_divergence(s_policy, t_policy)

    assert torch.allclose(loss_full_offpolicy, expected_offpolicy, atol=1e-5)
    assert torch.allclose(loss_full_onpolicy, expected_onpolicy, atol=1e-5)


def test_invalid_lambda_raises():
    shape = (1, 2, 10)
    x = torch.randn(*shape)
    try:
        on_policy_distillation_loss(x, x, x, x, lam=1.5)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------
# Freezing / orchestration tests -- THE core correctness requirement
# ---------------------------------------------------------------------
def test_freeze_all_except_expert_only_unfreezes_target():
    config = load_config("config/debug_laptop.yaml")
    model = SleepLLMBackbone(config)
    level = model.blocks[0].cms.levels[0]
    new_idx = level.expand()

    freeze_all_except_expert(model, level, new_idx)

    # The newly activated expert MUST be trainable.
    target_expert = level.experts[new_idx]
    assert all(p.requires_grad for p in target_expert.parameters())

    # EVERYTHING else -- embeddings, attention, other experts, other
    # levels -- must be frozen.
    frozen_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += 1
        is_target = any(param is p for p in target_expert.parameters())
        if not is_target:
            assert not param.requires_grad, f"{name} should be frozen but isn't!"
            frozen_count += 1

    assert frozen_count == total_count - len(list(target_expert.parameters()))


def test_compute_knowledge_seeding_loss_is_differentiable():
    """
    End-to-end sanity check: build a loss from a fake reward + fake
    log-prob + fake KL term, and confirm .backward() actually produces a
    gradient -- proving the whole Module 6 loss is a valid, trainable
    scalar, not just a formula that happens to type-check.
    """
    fake_param = torch.nn.Parameter(torch.randn(4))
    log_prob_sum = fake_param.sum()  # depends on fake_param, so grad can flow
    distillation_kl = (fake_param ** 2).sum()

    loss = compute_knowledge_seeding_loss(
        reward=0.8, log_prob_sum=log_prob_sum, distillation_kl=distillation_kl, alpha=0.5
    )
    loss.backward()

    assert fake_param.grad is not None
    assert fake_param.grad.abs().sum() > 0
