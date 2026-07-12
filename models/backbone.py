"""
models/backbone.py
====================
SleepLLMBackbone: the complete architecture from Figure 1/2 of the paper --
Token Embedding -> [Sequence Layer (attention) -> LayerNorm -> CMS chain
-> LayerNorm] x N -> Output head (next-token logits).

This is the model that gets trained during "wake" (normal language
modeling) and modified during "sleep" (memory consolidation, Module 4-6,
and dreaming, Module 7).
"""

import torch
import torch.nn as nn

from config.model_config import SleepLLMConfig
from models.attention import CausalSelfAttention
from models.cms import ContinuumMemorySystem


class SleepLLMBlock(nn.Module):
    """
    One full Transformer-style block: attention (Sequence Layer, the
    short-term memory) followed by the CMS chain (the frequency-tiered
    long-term memory), each wrapped with pre-LayerNorm and a residual
    connection -- the standard "pre-LN Transformer" recipe (GPT-2 style),
    chosen because pre-LN is known to train more stably than the original
    post-LN Transformer, especially in smaller models like ours.
    """

    def __init__(self, config: SleepLLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model.hidden_dim)
        self.attn = CausalSelfAttention(
            hidden_dim=config.model.hidden_dim,
            num_heads=config.model.num_attention_heads,
            max_seq_len=config.model.max_sequence_length,
            dropout=config.model.dropout,
        )
        self.ln2 = nn.LayerNorm(config.model.hidden_dim)
        self.cms = ContinuumMemorySystem(
            level_configs=config.memory.levels,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual pattern: x = x + Sublayer(LayerNorm(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.cms(self.ln2(x))
        return x


class SleepLLMBackbone(nn.Module):
    """
    The full model: embeddings -> N SleepLLMBlocks -> final LayerNorm ->
    output head.

    Why we tie the output head's weights to the input embedding
    (weight tying, Press & Wolf 2017 -- standard practice in GPT-2 and
    most modern LLMs): it roughly halves the parameter count contributed
    by the vocabulary (50257 x 128 = ~6.4M params, DOUBLED if untied),
    which matters a lot at our tiny scale, and empirically doesn't hurt
    (often helps) small-model quality.
    """

    def __init__(self, config: SleepLLMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.model.vocab_size, config.model.hidden_dim)
        self.position_embedding = nn.Embedding(config.model.max_sequence_length, config.model.hidden_dim)
        self.embed_dropout = nn.Dropout(config.model.dropout)

        self.blocks = nn.ModuleList(
            [SleepLLMBlock(config) for _ in range(config.model.num_sequence_layers)]
        )

        self.final_ln = nn.LayerNorm(config.model.hidden_dim)
        self.lm_head = nn.Linear(config.model.hidden_dim, config.model.vocab_size, bias=False)

        # Weight tying: the output head literally REUSES the embedding
        # matrix (not a copy -- the same tensor object), so gradients
        # from the language modeling loss flow back into token_embedding
        # directly through lm_head too.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Standard GPT-2-style initialization: small random weights for
        Linear/Embedding layers (std=0.02, matching Radford et al. 2019),
        biases at zero. Proper initialization matters a lot for training
        stability -- e.g., too-large initial weights can cause the
        attention softmax to saturate immediately, producing near-zero
        gradients from step 1 (a classic silent training bug).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) integer token ids.
        Returns:
            logits: (batch, seq_len, vocab_size) -- unnormalized scores
                over the vocabulary for the NEXT token at each position.

        Time complexity: dominated by attention (O(seq_len^2 * hidden_dim)
        per layer) and CMS experts (O(seq_len * hidden_dim *
        expert_hidden_dim) per active expert per layer), times
        num_sequence_layers.
        Memory usage: parameters are tiny at our config (see test output
        below); the dominant memory cost during training is activations
        (proportional to batch_size * seq_len * hidden_dim * num_layers)
        and the attention weight matrices.
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Convenience method -- you'll want this constantly for reporting model size."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
