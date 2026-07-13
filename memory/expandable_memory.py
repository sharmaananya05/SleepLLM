"""
memory/expandable_memory.py
=============================
ExpandableMemoryLevel: replaces Module 3's static MemoryLevel with one
that can GROW new low-rank experts at "sleep" time, following the paper's
exact prescription (Section 3.2, "Note on Implementation"):

    "Implementing the growing sparse modules can be extremely challenging
    if it requires a direct change in the dimensionality of tensors in
    the implementation. Alternatively, we can initially have those
    parameters in the model, but masked them in the forward and backward
    pass, before their initial activation in a sleep stage."

Design, matching that quote exactly:
- At construction, we PRE-ALLOCATE all `max_experts` expert slots (the
  first `num_experts` as dense Experts -- reused from models/cms.py --
  the rest as LowRankExpert "growth slots").
- An `active_mask` boolean buffer tracks which slots are "real" right
  now. Inactive slots:
    (a) receive ZERO routing probability (masked to -inf before softmax)
    (b) are SKIPPED in the forward loop entirely -- so they get no
        gradient and cost no compute, not just "zero contribution."
- `expand()` flips the mask for the NEXT inactive slot to True --
  the moment memory consolidation (Module 6) decides to grow this level.

This class REPLACES `MemoryLevel` from models/cms.py. We keep
models/cms.py's `Expert` class (dense expert) and import it here rather
than duplicating it -- one definition of "what a dense expert is,"
reused by both the static (Module 3) and growable (Module 4) paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MemoryLevelConfig
from memory.lora_expert import LowRankExpert
from models.expert import Expert


class ExpandableMemoryLevel(nn.Module):
    def __init__(self, level_config: MemoryLevelConfig, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = level_config
        self.hidden_dim = hidden_dim

        if level_config.num_experts > level_config.max_experts:
            raise ValueError(
                f"[{level_config.name}] num_experts ({level_config.num_experts}) "
                f"cannot exceed max_experts ({level_config.max_experts})"
            )

        self.router = nn.Linear(hidden_dim, level_config.max_experts)

        # Slots [0, num_experts) start ACTIVE as dense experts (the
        # model's initial capacity). Slots [num_experts, max_experts)
        # are PRE-BUILT low-rank experts, held INACTIVE until expand()
        # is called -- exactly the paper's "have the parameters, mask
        # them until activation" strategy.
        experts: list[nn.Module] = []
        for i in range(level_config.max_experts):
            if i < level_config.num_experts:
                experts.append(Expert(hidden_dim, level_config.expert_hidden_dim, dropout))
            else:
                experts.append(LowRankExpert(hidden_dim, level_config.lora_rank))
        self.experts = nn.ModuleList(experts)

        active_mask = torch.zeros(level_config.max_experts, dtype=torch.bool)
        active_mask[: level_config.num_experts] = True
        # persistent=True (default): SAVE this in checkpoints. Which
        # experts are active is essential model state -- reloading a
        # checkpoint without it would silently un-grow the model.
        self.register_buffer("active_mask", active_mask, persistent=True)

    @property
    def num_active_experts(self) -> int:
        return int(self.active_mask.sum().item())

    def expand(self) -> int:
        """
        Activate the next currently-inactive expert slot. This is what
        Module 6 (Knowledge Seeding) will call at a sleep-consolidation
        event, matching paper Section 3.2: "after each sleep time, the
        parameters of a subset of layers are growing."

        Returns:
            The index of the newly activated expert slot -- Module 6
            will need this to know WHICH expert's {A, B} to train during
            distillation.

        Raises:
            RuntimeError: if already at max_experts. The paper does not
            specify what happens when growth capacity is exhausted (this
            is a practical engineering limit, not a paper concept) --
            we fail loudly rather than silently ignoring the expand()
            call, since silently doing nothing would make a real bug
            (capacity exhausted mid-training) look like consolidation
            succeeded.
        """
        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_indices) == 0:
            raise RuntimeError(
                f"[{self.config.name}] Cannot expand: already at "
                f"max_experts={self.config.max_experts}. Increase "
                f"max_experts in the config if more growth is needed."
            )
        new_idx = int(inactive_indices[0].item())
        self.active_mask[new_idx] = True
        return new_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time complexity: O(seq_len * hidden_dim * num_active_experts) for
        routing, plus per-token compute for exactly ONE active expert
        (top-1 routing) -- INACTIVE slots contribute nothing to either
        time or memory cost, since we skip them in the loop below rather
        than computing-then-masking.
        """
        batch, seq_len, _ = x.shape

        router_logits = self.router(x)  # (batch, seq_len, max_experts)
        # Mask inactive slots to -inf BEFORE softmax, so they receive
        # exactly zero routing probability -- this is what guarantees
        # "not-yet-activated experts never receive knowledge or
        # gradients" (paper's plasticity/stability requirement).
        mask = self.active_mask.to(router_logits.device)
        router_logits = router_logits.masked_fill(~mask, float("-inf"))

        router_probs = F.softmax(router_logits, dim=-1)
        top1_prob, top1_idx = router_probs.max(dim=-1)

        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            if not self.active_mask[expert_id]:
                continue  # skip inactive slots entirely: no compute, no grad
            token_mask = (top1_idx == expert_id).unsqueeze(-1)
            if token_mask.any():
                expert_out = expert(x)
                output = output + expert_out * token_mask

        output = output * top1_prob.unsqueeze(-1)
        return output
