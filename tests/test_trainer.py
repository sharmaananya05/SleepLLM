"""
tests/test_trainer.py
=======================
Verifies Module 8 (trainer): wake steps actually reduce loss, sleep
events fire exactly at the scheduled steps and correctly grow the model,
and the freeze/unfreeze cycle around a sleep event doesn't leave the
model stuck partially frozen afterward.
"""

import torch

from config.model_config import load_config
from models.backbone import SleepLLMBackbone
from trainer.trainer import SleepLLMTrainer


def _build_trainer():
    config = load_config("config/debug_laptop.yaml")
    torch.manual_seed(config.seed)
    model = SleepLLMBackbone(config)
    trainer = SleepLLMTrainer(model, config, wake_lr=1e-3, ks_lr=1e-2, ks_steps=2)
    return trainer, config


def test_wake_step_reduces_loss_on_repeated_pattern():
    """
    A simple, strong sanity check for a training loop: if we train on
    the SAME easy, repeated token pattern for several steps, loss should
    clearly go down -- if it doesn't, something in the wake step
    (optimizer, gradient flow, loss computation) is broken.
    """
    trainer, config = _build_trainer()
    input_ids = torch.arange(16).unsqueeze(0) % 20  # easy repeating pattern
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    first_loss = trainer.wake_step(input_ids, target_ids)
    for _ in range(20):
        last_loss = trainer.wake_step(input_ids, target_ids)

    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
    )


def test_sleep_event_fires_at_scheduled_step_and_grows_expert_count():
    trainer, config = _build_trainer()
    high_freq_chunk = config.memory.levels[0].chunk_length  # 16 in debug config

    level = trainer.model.blocks[0].cms.levels[0]
    experts_before = level.num_active_experts

    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    target_ids = torch.randint(0, config.model.vocab_size, (2, 16))

    # Step forward exactly to the first scheduled consolidation step.
    result = None
    for _ in range(high_freq_chunk):
        result = trainer.train_step(input_ids, target_ids)

    assert len(result.sleep_events) > 0, "Expected a sleep event at the scheduled step"
    assert level.num_active_experts == experts_before + 1


def test_model_is_fully_trainable_again_after_a_sleep_event():
    """
    Core safety check: after a sleep event freezes most of the model,
    the NEXT wake step must be able to train everything again -- no
    parameter should be left permanently stuck at requires_grad=False.
    """
    trainer, config = _build_trainer()
    high_freq_chunk = config.memory.levels[0].chunk_length

    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    target_ids = torch.randint(0, config.model.vocab_size, (2, 16))

    for _ in range(high_freq_chunk):
        trainer.train_step(input_ids, target_ids)

    # A wake step call ALREADY re-enables all grads (first line of
    # wake_step) -- but let's verify directly without even calling it,
    # by checking sleep_step's own explicit restoration.
    assert all(p.requires_grad for p in trainer.model.parameters())


def test_no_sleep_events_before_first_scheduled_step():
    trainer, config = _build_trainer()
    high_freq_chunk = config.memory.levels[0].chunk_length

    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    target_ids = torch.randint(0, config.model.vocab_size, (2, 16))

    for _ in range(high_freq_chunk - 1):  # stop ONE step short
        result = trainer.train_step(input_ids, target_ids)
        assert result.sleep_events == []
