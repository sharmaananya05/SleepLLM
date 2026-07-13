"""
dreaming/self_improvement.py
==============================
The isolated fine-tune + reward step from paper Section 3.4, Eq. 5:

    "For each DREAM(i) in D we consider an isolated instance of the model
    and update its parameters via supervised finetuning (with LoRA):
    theta'(i) <- SFT(theta(i), DREAM(i)). Given the new fine-tuned model,
    ... we reward the generation of DREAM(i) based on LM_theta'(i)'s
    performance improvement over LM_theta(i):
        r(DREAM(i), tau(.), LM_theta(i)) = 1 if improves, 0 otherwise."

"The paper does not specify this implementation detail": exactly what
tau(.) (the downstream evaluation measure) is, in general -- it's
task-dependent. For a from-scratch educational implementation without a
fixed downstream benchmark wired in yet (that comes with Module 9,
evaluation/), we use held-out language-modeling loss on a small eval
sequence as a stand-in tau(.): "improvement" = eval loss goes down after
fine-tuning on the dream. This matches the paper's OWN Appendix B.3
description for the ARC setting ("report the fraction of dreams that
yield a correct answer"), just generalized to a loss-based signal since
we don't have ARC-style task labels here.

REST_EM SIMPLIFICATION (flagged): the paper cites the full ReSTEM
algorithm (Singh et al. 2024a), which iterates sample -> filter -> retrain
over multiple rounds. We implement ONE simplified iteration: generate
dreams, keep only the ones with reward=1 (i.e. that demonstrably helped),
and do a single SFT pass on the kept set. This captures ReSTEM's core
idea (train only on self-generated data that's been verified to help)
without its full iterative refinement loop -- a reasonable scope cut for
a first working version, not a paper-specified simplification.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.backbone import SleepLLMBackbone


@dataclass
class DreamOutcome:
    """Records what happened when we tried fine-tuning on one dream."""
    dream_ids: torch.Tensor
    eval_loss_before: float
    eval_loss_after: float
    reward: int  # 1 if improved, 0 otherwise -- Eq. 5


def _eval_loss(model: SleepLLMBackbone, eval_ids: torch.Tensor) -> float:
    """
    Held-out language modeling loss -- our tau(.) proxy (see module
    docstring). Lower is better.
    """
    model.eval()
    with torch.no_grad():
        logits = model(eval_ids[:, :-1])
        targets = eval_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss.item()


def isolated_sft_and_reward(
    model: SleepLLMBackbone,
    dream_ids: torch.Tensor,
    eval_ids: torch.Tensor,
    lr: float = 1e-3,
    sft_steps: int = 3,
) -> DreamOutcome:
    """
    Implements Eq. 5: fine-tune an ISOLATED COPY of the model on one
    dream, check whether held-out eval loss improved, assign binary
    reward.

    Args:
        model: the current SleepLLMBackbone. NOT modified by this
            function -- we deepcopy it, exactly matching the paper's
            "isolated instance" language, so the ORIGINAL model's
            parameters are completely unaffected by any single dream's
            fine-tune attempt. Only dreams that pass this test go on to
            actually train the real model (via Module 6's knowledge
            seeding machinery, or a follow-up ReSTEM aggregation step).
        dream_ids: (1, seq_len) or (seq_len,) -- one dream's token ids.
        eval_ids: (1, eval_len) held-out sequence for measuring tau(.).
        lr: learning rate for the isolated SFT steps. Not specified by
            the paper for this exact inner loop (Table 5 in the paper's
            appendix gives 5e-6 for the OUTER training loop, a very
            different setting with much larger models) -- we use a
            higher 1e-3 here since our isolated copy trains for only a
            handful of steps on tiny data and needs a visible effect
            within that budget to produce a meaningful reward signal.
        sft_steps: number of gradient steps on this single dream. Small
            on purpose -- Section 3.4 flags SEAL's (Zweiger et al. 2025)
            own limitation of being "limited to a small number of
            self-edits" due to SFT cost; we inherit the same constraint
            deliberately.

    Returns:
        A DreamOutcome recording before/after eval loss and the binary
        reward -- consumed by dream_selection-adjacent code to build the
        ReSTEM training set (only reward=1 dreams).

    Time complexity: O(sft_steps) forward+backward passes on a FULL
    COPY of the model -- this is the single most expensive part of
    Dreaming, which is why the paper (and we) keep sft_steps small and
    dream counts modest.
    Memory usage: requires a full duplicate of the model's parameters
    (deepcopy) -- at our 6.7M-parameter scale this is ~27MB, trivial;
    would NOT be trivial at the paper's Llama-3B/8B scale, which is
    exactly the kind of laptop-vs-cloud tradeoff your professor flagged.
    """
    if dream_ids.dim() == 1:
        dream_ids = dream_ids.unsqueeze(0)
    if eval_ids.dim() == 1:
        eval_ids = eval_ids.unsqueeze(0)

    eval_loss_before = _eval_loss(model, eval_ids)

    # Isolated copy: paper's theta(i) -> theta'(i). deepcopy ensures the
    # ORIGINAL model is untouched by this dream's trial fine-tune.
    isolated_model = copy.deepcopy(model)
    isolated_model.train()
    optimizer = torch.optim.Adam(isolated_model.parameters(), lr=lr)

    for _ in range(sft_steps):
        optimizer.zero_grad()
        logits = isolated_model(dream_ids[:, :-1])
        targets = dream_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    eval_loss_after = _eval_loss(isolated_model, eval_ids)

    # Eq. 5: reward = 1 if improved (LOWER eval loss = better for a loss
    # metric, so "improves" means eval_loss_after < eval_loss_before).
    reward = 1 if eval_loss_after < eval_loss_before else 0

    return DreamOutcome(
        dream_ids=dream_ids,
        eval_loss_before=eval_loss_before,
        eval_loss_after=eval_loss_after,
        reward=reward,
    )


def restem_filter(outcomes: list[DreamOutcome]) -> list[torch.Tensor]:
    """
    Simplified ReSTEM filtering step: keep only the dreams that
    DEMONSTRABLY helped (reward=1), discard the rest. The REAL model
    (not an isolated copy) would then be trained on this filtered set --
    that final training step reuses Module 6's on-policy distillation
    machinery (or a plain SFT step, for dreams not tied to a specific
    expert-consolidation event), left to the trainer (Module 8) to wire
    up, since it depends on which memory level/expert is the current
    consolidation target.
    """
    return [outcome.dream_ids for outcome in outcomes if outcome.reward == 1]
