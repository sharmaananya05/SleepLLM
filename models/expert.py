"""
models/expert.py
==================
The dense "Expert" MLP, extracted into its own file so both
models/cms.py (Module 3's static CMS) and memory/expandable_memory.py
(Module 4's growable CMS) can import ONE definition without creating a
circular import between the models/ and memory/ packages.
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
