"""
memory/dense_expert.py
=========================
The dense "Expert" MLP -- moved here (from its original home in
models/expert.py) to fix a circular import between the `models` and
`memory` packages.

Why the circular import existed: `memory/expandable_memory.py` imported
`Expert` from `models.expert`, but importing ANY submodule of `models`
first triggers `models/__init__.py`, which imports `models.backbone`,
which imports `models.cms`, which imports `memory.expandable_memory` --
back to where we started, while it's still mid-import. This "worked" by
accident only when some OTHER test file imported `models.backbone`
FIRST (finishing that import chain before `memory` was ever touched),
which is exactly the kind of fragile, order-dependent bug that's
invisible until you run a test file in isolation (as we did here).

Fix: `memory` no longer depends on `models` at all. `models/cms.py`
depends on `memory` (one direction only) for its
`ExpandableMemoryLevel`, which internally uses this `Expert`. A thin
re-export is kept at `models/expert.py` for backward compatibility with
any code that imports it from its original location.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    One "dense" expert: a standard 2-layer MLP with GELU activation --
    identical in spirit to a Transformer's feed-forward block.

    Time complexity: O(seq_len * hidden_dim * expert_hidden_dim).
    Memory usage: 2 * hidden_dim * expert_hidden_dim parameters.
    """

    def __init__(self, hidden_dim: int, expert_hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
