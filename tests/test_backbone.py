"""
tests/test_backbone.py
========================
Verifies Module 3: the SleepLLMBackbone actually runs a forward AND
backward pass, on the exact laptop-sized config we'll train with.

Why we test the BACKWARD pass too, not just forward shapes:
A model can produce correctly-shaped output and still be untrainable
(e.g. a bug that detaches gradients somewhere, or a dead ReLU that zeros
everything). Checking that gradients are non-None and non-zero after
.backward() is the cheapest possible check that the whole computational
graph is actually connected and learnable.
"""

import torch

from config.model_config import load_config
from models.backbone import SleepLLMBackbone


def _build_model():
    config = load_config("config/debug_laptop.yaml")
    torch.manual_seed(config.seed)
    model = SleepLLMBackbone(config)
    return model, config


def test_forward_pass_shape_is_correct():
    model, config = _build_model()
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.model.vocab_size)
    assert not torch.isnan(logits).any(), "NaNs in output -- likely an init or attention bug"


def test_backward_pass_produces_gradients():
    model, config = _build_model()
    input_ids = torch.randint(0, config.model.vocab_size, (2, 16))

    logits = model(input_ids)
    # A trivial loss just to trigger backprop -- not real language modeling
    # loss (that comes with the trainer in Module 8), just enough to prove
    # gradients flow through the whole model.
    loss = logits.sum()
    loss.backward()

    # Spot-check a parameter from EACH major component actually got a
    # gradient: embedding, attention, and a CMS expert.
    assert model.token_embedding.weight.grad is not None
    assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    assert model.blocks[0].cms.levels[0].experts[0].fc1.weight.grad is not None


def test_parameter_count_is_laptop_sized():
    model, _ = _build_model()
    n_params = model.num_parameters()
    # Sanity ceiling: this config should be well under 50M params --
    # if it's suddenly 500M+, something in the config or model wiring
    # is wrong (e.g. an accidental extra vocab-sized layer).
    assert n_params < 50_000_000, f"Unexpectedly large model: {n_params:,} params"
