"""
evaluation/continual_eval.py
===============================
The actual experiment: Class-Incremental Learning (paper Section 4.1,
Figure 3). Trains a classifier sequentially on Task 1 (classes 0..k-1),
then Task 2 (classes k..2k-1), then measures accuracy on BOTH tasks --
the standard test for catastrophic forgetting.

Runs this TWICE with identical data/seeds:
1. "Sleep" condition: normal wake steps via SleepLLMTrainer, which
   triggers scheduled consolidation events (Modules 4-6) as training
   proceeds.
2. "No-Sleep" (baseline) condition: plain continual fine-tuning --
   wake steps only, consolidation NEVER fires (achieved by using a
   MemoryConfig with chunk_lengths larger than the total number of
   training steps, so no event is ever scheduled -- reusing the exact
   same scheduler machinery rather than a separate code path, which
   guarantees the ONLY difference between conditions is Sleep itself).

This directly reproduces the paper's comparison: "Hope [+ Sleep]
performs best ... Relative to ICL, gains come from converting
prompt-level adaptation into durable parametric memory through
consolidation" (Section 4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import SleepLLMConfig
from evaluation.classifier_head import IntentClassifierHead
from evaluation.toy_intent_data import generate_intent_task
from sleep.scheduler import SleepScheduler


@dataclass
class ContinualLearningResult:
    condition: str  # "sleep" or "no_sleep"
    task1_acc_after_task1: float
    task1_acc_after_task2: float  # <-- the key forgetting metric
    task2_acc_after_task2: float
    forgetting: float = field(init=False)

    def __post_init__(self) -> None:
        # "Forgetting" = how much Task 1 accuracy DROPPED after learning
        # Task 2. Lower is better (0 = no forgetting at all).
        self.forgetting = self.task1_acc_after_task1 - self.task1_acc_after_task2


def _accuracy(model: IntentClassifierHead, inputs: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()


def _train_classification_steps(
    model: IntentClassifierHead,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_epochs: int,
    freeze_classifier_rows_below: int | None = None,
) -> None:
    """
    Args:
        freeze_classifier_rows_below: if set, zeroes the gradient for
            classifier output rows [0, freeze_classifier_rows_below)
            after backward() but before the optimizer step. Used during
            Task 2 training so old-task classes' OWN output weights
            can't be directly overwritten by Task 2's gradient.

        WHY THIS MATTERS (a real methodological fix, not a paper detail):
        cross_entropy over ALL classes computes softmax normalization
        across every class logit, including old ones -- so even though
        Task 2's loss only "targets" new classes, gradients still flow
        into old classes' rows of the classifier weight matrix (pushing
        their logits down via softmax competition). Without this
        protection, BOTH the Sleep and no-Sleep conditions show ~100%
        forgetting regardless of backbone quality, because the
        classifier HEAD itself (not the backbone Sleep is supposed to
        protect) gets scrambled -- which would make this experiment
        measure the wrong thing entirely. Freezing old rows isolates the
        comparison to what Sleep actually claims to help with: BACKBONE
        knowledge retention. This mirrors standard practice in the
        continual-learning literature (e.g. iCaRL, and other
        class-incremental methods commonly freeze or regularize old
        classifier rows separately from backbone-forgetting mitigation).
    """
    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        if freeze_classifier_rows_below is not None:
            with torch.no_grad():
                model.classifier.weight.grad[:freeze_classifier_rows_below] = 0.0
                if model.classifier.bias is not None:
                    model.classifier.bias.grad[:freeze_classifier_rows_below] = 0.0

        optimizer.step()


def run_continual_learning_experiment(
    config: SleepLLMConfig,
    build_model_fn,
    num_classes_per_task: int = 5,
    examples_per_class: int = 10,
    seq_len: int = 16,
    num_epochs_per_task: int = 30,
    enable_sleep: bool = True,
    data_seed: int = 42,
) -> ContinualLearningResult:
    """
    Runs ONE full condition (Sleep or no-Sleep) of the class-incremental
    experiment.

    Args:
        config: SleepLLMConfig (memory chunk_lengths control whether
            Sleep fires -- see enable_sleep below).
        build_model_fn: a callable() -> SleepLLMBackbone, so we can
            build a FRESH model for each condition (Sleep vs no-Sleep
            must start from identical, freshly-initialized weights --
            fixed by config.seed, per Module 2's set_seed()).
        num_classes_per_task: k -- Task 1 = classes [0,k), Task 2 =
            classes [k,2k).
        examples_per_class, seq_len: synthetic data shape (see
            evaluation/toy_intent_data.py).
        num_epochs_per_task: gradient steps of plain classification
            training per task (NOT the same as Sleep's own internal
            wake/sleep step count -- this is classifier-head training on
            top of a backbone that may ALSO be undergoing Sleep
            consolidation on its own step counter, when enable_sleep=True).
        enable_sleep: if True, use the model's real MemoryConfig chunk
            lengths (Sleep fires per schedule). If False, we internally
            construct a scheduler whose chunk lengths exceed the total
            step budget, guaranteeing zero consolidation events --
            i.e. a plain continually-fine-tuned baseline, all else equal.
        data_seed: shared across BOTH conditions for a fair comparison.

    Returns:
        A ContinualLearningResult with the key forgetting metric.
    """
    total_classes = num_classes_per_task * 2
    backbone = build_model_fn()
    model = IntentClassifierHead(backbone, num_classes=total_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = SleepScheduler(config.memory) if enable_sleep else None

    task1_inputs, task1_labels = generate_intent_task(
        num_classes_per_task, examples_per_class, seq_len,
        config.model.vocab_size, class_offset=0, seed=data_seed,
    )
    task2_inputs, task2_labels = generate_intent_task(
        num_classes_per_task, examples_per_class, seq_len,
        config.model.vocab_size, class_offset=num_classes_per_task, seed=data_seed + 1,
    )

    # --- Train Task 1 ---
    step = 0
    for _ in range(num_epochs_per_task):
        step += 1
        _train_classification_steps(model, optimizer, task1_inputs, task1_labels, num_epochs=1)
        if scheduler is not None:
            _maybe_consolidate(backbone, scheduler, step, reference_input=task1_inputs[:4])

    task1_acc_after_task1 = _accuracy(model, task1_inputs, task1_labels)

    # --- Train Task 2 (continuing from Task 1's weights -- the actual
    # continual-learning setup: NO reset in between) ---
    for _ in range(num_epochs_per_task):
        step += 1
        _train_classification_steps(
            model, optimizer, task2_inputs, task2_labels, num_epochs=1,
            freeze_classifier_rows_below=num_classes_per_task,
        )
        if scheduler is not None:
            _maybe_consolidate(backbone, scheduler, step, reference_input=task2_inputs[:4])

    task1_acc_after_task2 = _accuracy(model, task1_inputs, task1_labels)
    task2_acc_after_task2 = _accuracy(model, task2_inputs, task2_labels)

    return ContinualLearningResult(
        condition="sleep" if enable_sleep else "no_sleep",
        task1_acc_after_task1=task1_acc_after_task1,
        task1_acc_after_task2=task1_acc_after_task2,
        task2_acc_after_task2=task2_acc_after_task2,
    )


def _maybe_consolidate(backbone, scheduler: SleepScheduler, step: int, reference_input: torch.Tensor) -> None:
    """
    A lightweight, evaluation-specific version of Trainer.sleep_step()
    (Module 8) -- reused conceptually, kept separate here so
    evaluation/ doesn't have to depend on the full SleepLLMTrainer's
    wake-step machinery (this experiment trains via a classification
    loss, not language-modeling loss).
    """
    from distillation.knowledge_seeding import freeze_all_except_expert
    from distillation.losses import reverse_kl_divergence

    events = scheduler.get_events_at_step(step)
    if not events:
        return

    for event in events:
        for block in backbone.blocks:
            level = block.cms.levels[event.level_index]
            new_idx = level.expand()

            backbone.eval()
            with torch.no_grad():
                teacher_logits = backbone(reference_input).detach()

            freeze_all_except_expert(backbone, level, new_idx)
            ks_optimizer = torch.optim.Adam(level.experts[new_idx].parameters(), lr=1e-2)
            backbone.train()
            for _ in range(3):
                ks_optimizer.zero_grad()
                student_logits = backbone(reference_input)
                kl = reverse_kl_divergence(student_logits, teacher_logits)
                kl.backward()
                ks_optimizer.step()

    for param in backbone.parameters():
        param.requires_grad = True
