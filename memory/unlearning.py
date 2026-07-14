"""
memory/unlearning.py
=======================
Machine unlearning for SleepLLM -- NOT part of the original paper (the
paper's Sleep paradigm is about PREVENTING forgetting; unlearning is
about deliberately, targetedly CAUSING it for specific information).
This module implements two complementary methods, since this is exactly
the kind of addition your project title ("Memory Consolidation and
Unlearning in Intelligent Systems") requires beyond the paper's own scope.

Method 1 -- Structural unlearning (memory/expandable_memory.py's
unlearn() method): deactivate + reset a SPECIFIC expert known to store
a targeted memory. Fast, clean, exact -- but only works if you know
WHICH expert holds the memory (e.g. because Knowledge Seeding, Module 6,
consolidated it there deliberately).

Method 2 -- Gradient-ascent unlearning (this file): a standard technique
from the unlearning literature (e.g. approaches following "Who's Harry
Potter? Approximate Unlearning in LLMs", Eldan & Russinovich 2023, and
earlier gradient-ascent-based unlearning work) -- fine-tune the model to
INCREASE loss on a "forget set" (data representing what should be
unlearned) while a "retain set" term keeps other knowledge intact via
ordinary (decreasing) loss. Use this when the memory to unlearn is
diffused across many parameters rather than isolated in one expert.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnlearningResult:
    """Records the outcome of a gradient-ascent unlearning run."""
    forget_loss_before: float
    forget_loss_after: float
    retain_loss_before: float
    retain_loss_after: float


def gradient_ascent_unlearn(
    model: nn.Module,
    forward_fn,
    forget_inputs: torch.Tensor,
    forget_labels: torch.Tensor,
    retain_inputs: torch.Tensor,
    retain_labels: torch.Tensor,
    steps: int = 10,
    lr: float = 1e-3,
    retain_weight: float = 1.0,
) -> UnlearningResult:
    """
    Fine-tunes `model` IN PLACE to unlearn the forget set while
    preserving performance on the retain set.

    Args:
        model: any classifier-style model (e.g. IntentClassifierHead)
            whose CURRENTLY TRAINABLE parameters (requires_grad=True)
            will be updated. Callers should set up freezing BEFOREHAND
            (e.g. via freeze_all_except_expert, Module 6) if only a
            specific expert should be touched during unlearning --
            otherwise this updates every trainable parameter.
        forward_fn: a callable(model, inputs) -> logits -- injected
            rather than assuming model(inputs) directly, so this
            function works with ANY classifier interface (kept
            decoupled from IntentClassifierHead specifically).
        forget_inputs, forget_labels: the data representing the memory
            to unlearn (e.g. all examples of one specific intent class).
        retain_inputs, retain_labels: data that should NOT be affected
            (e.g. all other intent classes).
        steps: number of gradient steps.
        lr: learning rate.
        retain_weight: weight on the retain-set loss term. Paper-adjacent
            literature (e.g. gradient-ascent unlearning baselines) treats
            this as a tunable trade-off knob between forgetting strength
            and retained-knowledge preservation; NOT a value from "the
            paper" (which doesn't cover unlearning at all) -- default
            1.0 (equal weighting) is the standard neutral starting point,
            same convention as gamma/lambda/alpha elsewhere in this repo.

    Returns:
        UnlearningResult with before/after loss on both sets, so callers
        can verify: forget_loss should INCREASE (worse at the forget
        task = successfully unlearned), retain_loss should stay roughly
        FLAT (not increase much = other knowledge preserved).

    The core mechanism -- why "ascent" on the forget set:
    Ordinary training MINIMIZES loss (gradient descent). To make a model
    WORSE at something (unlearn it), we instead MAXIMIZE its loss on that
    data -- equivalent to minimizing the NEGATIVE loss. Combined with an
    ordinary (minimized) retain-set loss, the total objective is:
        total_loss = -forget_loss + retain_weight * retain_loss
    Minimizing this total pushes forget_loss UP (unlearning) while
    keeping retain_loss DOWN (preservation) -- both via standard gradient
    descent on one combined objective.

    Time complexity: O(steps) forward+backward passes over the combined
    forget+retain batches -- same order as any standard fine-tuning loop.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    def _loss(inputs, labels):
        logits = forward_fn(model, inputs)
        return F.cross_entropy(logits, labels)

    model.eval()
    with torch.no_grad():
        forget_loss_before = _loss(forget_inputs, forget_labels).item()
        retain_loss_before = _loss(retain_inputs, retain_labels).item()

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        forget_loss = _loss(forget_inputs, forget_labels)
        retain_loss = _loss(retain_inputs, retain_labels)
        total_loss = -forget_loss + retain_weight * retain_loss
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        forget_loss_after = _loss(forget_inputs, forget_labels).item()
        retain_loss_after = _loss(retain_inputs, retain_labels).item()

    return UnlearningResult(
        forget_loss_before=forget_loss_before,
        forget_loss_after=forget_loss_after,
        retain_loss_before=retain_loss_before,
        retain_loss_after=retain_loss_after,
    )
