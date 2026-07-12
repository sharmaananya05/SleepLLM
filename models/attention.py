"""
models/attention.py
=====================
The "Sequence Layer" from Figure 1/2 of the paper: standard causal
multi-head self-attention, acting as the model's SHORT-TERM memory.

Paper cross-reference (Section 2.2):
"Attention module ... acts as an associative memory and conditions the
output on the past tokens in the context ... attention's update span is
the context length -- meaning that at the end of the context, its
corresponding parameters are updated and the acquired knowledge is
forgotten." In our terms: attention has effectively INFINITE update
frequency (Definition 1) -- it's the fastest, most fragile memory in the
whole system, which is exactly why the CMS chain that follows it exists:
to consolidate whatever attention picks up into progressively more
stable, slower-updating parameters.

"The paper does not specify this implementation detail":
The paper treats "sequence model (e.g., attention ... or other memory
modules or RNNs)" as a pluggable choice and focuses its contribution on
what happens AFTER it. We use standard causal self-attention because (a)
it's the most well-understood choice to validate the CMS/Sleep machinery
against, and (b) it lets us explain every line in a review without also
having to defend an exotic architecture choice.

For teaching purposes (you said you know Python but not LLM internals):
Self-attention lets each token "look at" every earlier token in the
sequence and decide how much to weight each one when building its own
representation. "Causal" means a token can only look at itself and
earlier tokens, never future ones -- required for autoregressive
(next-token-prediction) language modeling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention, the same core mechanism as GPT-2's
    attention block (Radford et al. 2019), built from scratch here (rather
    than imported) specifically so every line is inspectable and
    explainable in your project review.

    Args:
        hidden_dim: model width (d in the paper's notation).
        num_heads: number of attention heads. hidden_dim must be evenly
            divisible by num_heads (each head gets hidden_dim/num_heads
            dimensions -- we assert this below).
        max_seq_len: maximum sequence length, used to pre-build the causal
            mask once at construction time rather than rebuilding it on
            every forward pass (a small but standard efficiency choice).
        dropout: dropout probability applied to attention weights and the
            output projection, for regularization.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Paper's notation: Q, K, V projections. We compute all three with
        # ONE linear layer (3x the output width) instead of three separate
        # layers -- purely a speed optimization (one matmul instead of
        # three), mathematically identical to separate Q/K/V projections.
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precompute the causal mask ONCE. register_buffer means this
        # tensor moves with the model when you call .to(device), but is
        # NOT treated as a trainable parameter (no gradient, not saved
        # as part of the "learned" state_dict in the usual sense -- though
        # by default it IS included in state_dict; that's fine, it's cheap
        # and makes checkpoints self-contained).
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.bool(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
        Returns:
            (batch, seq_len, hidden_dim) -- same shape, attention-mixed.

        Time complexity: O(seq_len^2 * hidden_dim) -- the classic
        quadratic-in-sequence-length cost of self-attention. This is WHY
        the paper's CMS approach matters for long-context: attention
        itself doesn't scale to 10M tokens, but consolidating its output
        into fixed-size CMS parameters means the model doesn't need to
        re-attend over the full history forever.

        Memory usage: the attention weight matrix is
        (batch, num_heads, seq_len, seq_len) floats -- this is the
        dominant memory cost for long sequences. At our laptop config
        (seq_len=128), this is tiny; at the paper's tested 10M-token
        scale, this naive form would be completely infeasible (real
        long-context systems use techniques like FlashAttention or
        recurrent memory -- an explicit "possible improvement" for later,
        out of scope for this from-scratch educational implementation).
        """
        batch, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*hidden_dim)
        q, k, v = qkv.split(self.hidden_dim, dim=-1)

        # Reshape each of Q, K, V from (batch, seq_len, hidden_dim) into
        # (batch, num_heads, seq_len, head_dim) so each head attends
        # independently. transpose(1, 2) swaps the seq_len and num_heads
        # axes after the reshape.
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: Q @ K^T / sqrt(head_dim).
        # The 1/sqrt(head_dim) scaling (Vaswani et al. 2017) prevents the
        # dot products from growing too large in magnitude as head_dim
        # increases, which would otherwise push softmax into
        # near-one-hot, vanishing-gradient territory.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply the causal mask: positions where causal_mask is False
        # (i.e., "future" tokens relative to the query) get -inf, so
        # softmax assigns them ~0 probability.
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v  # (batch, num_heads, seq_len, head_dim)

        # Merge heads back into (batch, seq_len, hidden_dim).
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
