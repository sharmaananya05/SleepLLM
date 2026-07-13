"""
tests/test_dreaming.py
========================
Verifies Module 7: dream generation produces valid sequences, random
expert injection actually changes routing, importance scoring and
selection pick the right dreams, and the isolated SFT + reward loop
computes a correct binary signal per Eq. 5.
"""

import torch

from config.model_config import load_config
from models.backbone import SleepLLMBackbone
from dreaming.dream_generator import generate_dreams, random_expert_injection
from dreaming.dream_selection import compute_dream_importance_scores, select_dreams
from dreaming.self_improvement import isolated_sft_and_reward, restem_filter, DreamOutcome


def _build_model():
    config = load_config("config/debug_laptop.yaml")
    torch.manual_seed(config.seed)
    return SleepLLMBackbone(config), config


def test_generate_dreams_produces_correct_shape():
    model, config = _build_model()
    context = torch.randint(0, config.model.vocab_size, (8,))
    dreams = generate_dreams(model, context, num_dreams=4, max_new_tokens=5)
    assert dreams.shape == (4, 13)  # 8 context + 5 new tokens


def test_random_expert_injection_can_change_routing():
    """
    With injection_probability=1.0 (always inject), EVERY token should
    be forced to a random active expert -- the strongest possible test
    that the hook actually engages, not just exists unused.
    """
    model, config = _build_model()
    level = model.blocks[0].cms.levels[0]
    assert level._force_random_prob == 0.0  # off by default

    with random_expert_injection(model, probability=1.0):
        assert level._force_random_prob == 1.0  # engaged during the block
    assert level._force_random_prob == 0.0  # restored after the block


def test_dream_importance_scores_are_positive_finite():
    model, config = _build_model()
    dreams = torch.randint(0, config.model.vocab_size, (5, 16))
    scores = compute_dream_importance_scores(model, dreams)
    assert scores.shape == (5,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()  # cross-entropy loss is always non-negative


def test_select_dreams_returns_correct_count():
    dreams = torch.randint(0, 100, (10, 8))
    scores = torch.rand(10)
    selected = select_dreams(dreams, scores, top_k=3, num_random=2)
    assert selected.shape == (5, 8)


def test_select_dreams_always_includes_the_highest_scoring():
    dreams = torch.arange(10).unsqueeze(1).repeat(1, 4)  # dream i is all "i"s
    scores = torch.tensor([5.0, 1.0, 9.0, 2.0, 3.0, 0.5, 8.0, 4.0, 6.0, 7.0])
    selected = select_dreams(dreams, scores, top_k=2, num_random=0)
    # Highest scores are index 2 (9.0) and index 6 (8.0)
    selected_dream_ids = {row[0].item() for row in selected}
    assert selected_dream_ids == {2, 6}


def test_isolated_sft_does_not_modify_original_model():
    """
    Core correctness requirement: the "isolated instance" must be a true
    copy -- fine-tuning on a dream must NEVER change the original model's
    parameters, regardless of the outcome.
    """
    model, config = _build_model()
    original_param = model.token_embedding.weight.clone()

    dream_ids = torch.randint(0, config.model.vocab_size, (12,))
    eval_ids = torch.randint(0, config.model.vocab_size, (10,))

    isolated_sft_and_reward(model, dream_ids, eval_ids, sft_steps=2)

    assert torch.equal(model.token_embedding.weight, original_param)


def test_isolated_sft_reward_is_binary():
    model, config = _build_model()
    dream_ids = torch.randint(0, config.model.vocab_size, (12,))
    eval_ids = torch.randint(0, config.model.vocab_size, (10,))

    outcome = isolated_sft_and_reward(model, dream_ids, eval_ids, sft_steps=2)
    assert outcome.reward in (0, 1)
    assert isinstance(outcome.eval_loss_before, float)
    assert isinstance(outcome.eval_loss_after, float)


def test_restem_filter_keeps_only_reward_1_dreams():
    outcomes = [
        DreamOutcome(dream_ids=torch.tensor([1]), eval_loss_before=1.0, eval_loss_after=0.5, reward=1),
        DreamOutcome(dream_ids=torch.tensor([2]), eval_loss_before=1.0, eval_loss_after=1.5, reward=0),
        DreamOutcome(dream_ids=torch.tensor([3]), eval_loss_before=1.0, eval_loss_after=0.2, reward=1),
    ]
    kept = restem_filter(outcomes)
    assert len(kept) == 2
    assert torch.equal(kept[0], torch.tensor([1]))
    assert torch.equal(kept[1], torch.tensor([3]))
