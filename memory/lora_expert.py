"""
memory/lora_expert.py
=======================
The low-rank expert added during Memory Consolidation (paper Section 3.2):

    "we add a low-rank MLP parametrized by {A^(f_l*),s+1, B^(f_l*),s+1},
    where A^(f_l*) in R^(d x d_low) and B^(f_l*) in R^(d_low x d)
    (d_low << d), to the set of experts. These new parameters will be
    allocated for storing the new transferred knowledge from
    MLP^(f_l*-1)(.)"

Why low-rank (not a full dense expert) specifically for GROWTH:
A full dense expert at hidden_dim=128 costs ~33K params (see Module 3's
Expert class). A rank-4 low-rank expert costs only
2 * 128 * 4 = 1,024 params -- about 32x cheaper. Since a model may grow
MANY experts over repeated sleep cycles (paper Fig 7: growth happens
every C^(ell) steps), keeping each growth increment cheap is what makes
"the model can keep growing capacity over its lifetime" actually
practical rather than an exponential memory blowup.
"""

import torch
import torch.nn as nn


class LowRankExpert(nn.Module):
    """
    A single low-rank "growth" expert: output = (x @ A) @ B, i.e. a
    bottleneck MLP with NO nonlinearity in between (unlike the dense
    Expert in models/cms.py, which has a GELU between fc1/fc2).

    Why no nonlinearity here: this mirrors LoRA (Hu et al. 2022) exactly
    -- A @ B is a low-rank APPROXIMATION of a full-rank weight update,
    and LoRA's whole mathematical justification (that gradient updates
    to pretrained weights have low "intrinsic rank") applies to a LINEAR
    map, not a nonlinear one. Adding a nonlinearity here would break that
    connection to the well-studied LoRA theory this design borrows from.

    Args:
        hidden_dim: d in the paper's notation -- must match the CMS
            level's hidden_dim so outputs can be added into the same
            residual stream as the dense experts.
        lora_rank: d_low in the paper's notation. Must be << hidden_dim
            (enforced upstream by MemoryLevelConfig, not re-checked here
            to avoid duplicate validation logic).
    """

    def __init__(self, hidden_dim: int, lora_rank: int) -> None:
        super().__init__()
        self.A = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.B = nn.Linear(lora_rank, hidden_dim, bias=False)

        # LoRA-style initialization (Hu et al. 2022): A is initialized to
        # a small random matrix, B is initialized to EXACTLY ZERO. This
        # means a freshly-added low-rank expert computes f(x) = 0 at the
        # moment it's created -- it contributes NOTHING to the model's
        # output until it's actually trained during knowledge seeding
        # (Module 6). This is important: it means activating a new expert
        # slot never suddenly perturbs the model's existing behavior;
        # the perturbation only appears gradually as training updates B.
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim)

        Time complexity: O(seq_len * hidden_dim * lora_rank) -- linear in
        lora_rank, which is why keeping lora_rank small (paper: d_low << d)
        keeps growth cheap even after many consolidation events.
        Memory usage: 2 * hidden_dim * lora_rank parameters. At
        hidden_dim=128, lora_rank=4: just 1,024 params.
        """
        return self.B(self.A(x))
