"""
models/cms.py
==============
The Continuum Memory System (CMS): a chain of frequency-tiered MLP
blocks, matching Eq. 1 (Section 2.2) and Section 3.2 of the paper.

As of Module 4, each level is an `ExpandableMemoryLevel`
(memory/expandable_memory.py) -- capable of growing new low-rank experts
at "sleep" time via masking, per the paper's exact "Note on
Implementation" (Section 3.2). Module 3's original static `MemoryLevel`
is superseded by this; `Expert` (the dense expert building block) now
lives in models/expert.py so both models/ and memory/ can share it
without a circular import.
"""

import torch
import torch.nn as nn

from config.model_config import MemoryLevelConfig
from memory.expandable_memory import ExpandableMemoryLevel


class ContinuumMemorySystem(nn.Module):
    """
    The full CMS chain: ExpandableMemoryLevel_1 -> ... -> ExpandableMemoryLevel_k,
    matching Eq. 1 of the paper. Levels are ordered fast -> slow (enforced
    by MemoryConfig's validation in config/model_config.py).
    """

    def __init__(self, level_configs: list[MemoryLevelConfig], hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [ExpandableMemoryLevel(cfg, hidden_dim, dropout) for cfg in level_configs]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chains levels per Eq. 1, with a residual connection around the
        whole chain (standard Transformer practice, not an explicit
        paper detail) to keep gradients flowing to earlier layers even
        through multiple MoE routing decisions.
        """
        residual = x
        for level in self.levels:
            x = level(x)
        return residual + x
