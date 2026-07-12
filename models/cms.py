"""
models/cms.py
==============
The Continuum Memory System (CMS): a chain of frequency-tiered MLP blocks,
each a sparse Mixture-of-Experts, matching paper Section 2.2 and 3.2.

Paper cross-reference:
- Eq. 1: y_t = MLP^(f_k)(MLP^(f_k-1)(...MLP^(f_1)(x_t))) -- a CHAIN, not a
  parallel ensemble. Each level's OUTPUT feeds the next level's INPUT.
- Section 3.2: "the MLP blocks {MLP^(f_l)(.)} are sparse mixture of
  experts (MoEs) with a router R^(f_l): each MLP^(f_l)(.) includes a set
  of experts {W^(f_l),1, ..., W^(f_l),s_l}."

What this module does NOT yet do (deliberately -- that's Module 4):
- Growing new low-rank experts during a "sleep" consolidation event
  (Section 3.2's "Memory Consolidation: Parameter Expansion")
- Masking newly-added-but-not-yet-activated experts (the paper's "Note on
  Implementation": mask in forward/backward instead of resizing tensors)
Today we build the STATIC structure (a fixed number of experts per level,
as set in the YAML config) and a working router. Module 4 will add the
DYNAMIC growth/masking logic on top of this class.

For teaching purposes -- what is a Mixture-of-Experts (MoE)?
Instead of one big MLP processing every token, you have several smaller
MLPs ("experts"), and a small "router" network decides, per token, which
expert(s) should process it. This means the model can have a large TOTAL
parameter count (many experts) while only using a SMALL FRACTION of them
per token (sparse activation) -- more capacity without proportionally
more compute per token. Here we use "top-1" routing (each token goes to
exactly its single highest-scoring expert) -- the simplest, most common
MoE routing scheme (used by e.g. Switch Transformer).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import MemoryLevelConfig


class Expert(nn.Module):
    """
    One "dense" expert: a standard 2-layer MLP with a GELU activation,
    identical in spirit to a Transformer's feed-forward block.

    Time complexity: O(seq_len * hidden_dim * expert_hidden_dim) per
    forward pass for the tokens routed to this expert.
    Memory usage: 2 * hidden_dim * expert_hidden_dim parameters
    (the two Linear layers) -- e.g. at hidden_dim=128,
    expert_hidden_dim=128: ~33K params per expert. Trivially small.
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


class MemoryLevel(nn.Module):
    """
    One MLP^(f_ell)(.) block in the CMS chain: a router + a set of
    experts, matching Section 3.2's {W^(f_ell),1, ..., W^(f_ell),s_ell}.

    Why top-1 routing (not top-2 or "soft" weighted mixing of all
    experts): simplicity for a first implementation you need to explain
    line-by-line. Top-2 routing (used by e.g. Mixtral) generally improves
    quality slightly at 2x the per-token expert compute -- a documented
    "possible improvement" for later once the core Sleep mechanics work.
    """

    def __init__(self, level_config: MemoryLevelConfig, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.config = level_config
        self.hidden_dim = hidden_dim

        # The router: a single linear layer mapping each token's hidden
        # state to a score per expert. Section 3.2 calls this R^(f_ell).
        # We size it for max_experts up front (not just the CURRENT
        # num_experts) so that Module 4 can grow into pre-allocated
        # router capacity without reshaping this layer -- this mirrors
        # the paper's own "Note on Implementation" (Section 3.2): avoid
        # resizing tensors; mask instead.
        self.router = nn.Linear(hidden_dim, level_config.max_experts)

        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim, level_config.expert_hidden_dim, dropout)
                for _ in range(level_config.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) -- output of the previous
               level (or the Sequence Layer, for level 0).
        Returns:
            (batch, seq_len, hidden_dim)

        Time complexity: O(batch * seq_len * hidden_dim * num_experts)
        for routing logits (cheap), plus O(batch * seq_len * hidden_dim *
        expert_hidden_dim) for the actual expert compute PER TOKEN (each
        token only visits ONE expert, so total expert compute does not
        scale with num_experts -- that's the entire point of sparse MoE).
        """
        batch, seq_len, _ = x.shape
        num_active_experts = len(self.experts)

        # Only compute routing logits over CURRENTLY ACTIVE experts, not
        # the full max_experts capacity -- experts that don't exist yet
        # (future consolidation targets) must never receive routing mass.
        router_logits = self.router(x)[..., :num_active_experts]
        # (batch, seq_len, num_active_experts)

        router_probs = F.softmax(router_logits, dim=-1)
        top1_prob, top1_idx = router_probs.max(dim=-1)  # both (batch, seq_len)

        # Dispatch tokens to their chosen expert. For a first,
        # explainable implementation we loop over experts and mask -- an
        # O(num_experts) pass over the whole tensor rather than a
        # scatter/gather -- simple to read, and cheap at our tiny
        # num_experts counts (<=16 per config/model_config.py's
        # max_experts default).
        output = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            token_mask = (top1_idx == expert_id).unsqueeze(-1)  # (batch, seq_len, 1)
            if token_mask.any():
                expert_out = expert(x)
                output = output + expert_out * token_mask

        # Scale by the router's own confidence in its choice (standard
        # MoE practice, e.g. Switch Transformer) -- this makes the
        # router's probability directly affect the gradient it receives,
        # which is what lets it learn to route well in the first place.
        output = output * top1_prob.unsqueeze(-1)

        return output


class ContinuumMemorySystem(nn.Module):
    """
    The full CMS chain: MemoryLevel_1 -> MemoryLevel_2 -> ... -> MemoryLevel_k,
    matching Eq. 1 of the paper. Levels are ordered fast -> slow (as
    enforced by MemoryConfig's validation in config/model_config.py).
    """

    def __init__(self, level_configs: list[MemoryLevelConfig], hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            [MemoryLevel(cfg, hidden_dim, dropout) for cfg in level_configs]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chains levels exactly as Eq. 1 specifies: each level's output
        becomes the next level's input. We ALSO add a residual connection
        around the whole chain (standard Transformer practice, not an
        explicit paper detail) -- without it, gradients would have to
        flow through every MoE level's routing decision to reach earlier
        layers, which is a well-known source of training instability in
        deep MoE stacks.
        """
        residual = x
        for level in self.levels:
            x = level(x)
        return residual + x
