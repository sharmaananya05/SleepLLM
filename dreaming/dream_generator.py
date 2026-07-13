"""
dreaming/dream_generator.py
=============================
Generates synthetic "dreams" from the model itself (paper Section 3.4):

    "our dreaming process starts with generating m >= 1 dreams with
    having C in context ... each router in MoE blocks additionally
    chooses a random expert and so incorporates random irrelevant
    knowledge to the dreaming, learning the underlying patterns that are
    hidden from model's sight."

Two things happen here:
1. Standard autoregressive sampling (temperature/top-k) from the model,
   given a context C.
2. RANDOM EXPERT INJECTION: while generating, each MoE router
   occasionally routes to a random active expert instead of its own
   top-1 choice -- implemented via a context manager that temporarily
   flips a flag on every ExpandableMemoryLevel in the model.
"""

import random
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from models.backbone import SleepLLMBackbone


@contextmanager
def random_expert_injection(model: SleepLLMBackbone, probability: float = 0.3):
    """
    Context manager: while active, every ExpandableMemoryLevel in the
    model routes each token to a RANDOM active expert (instead of its
    router's top-1 choice) with the given probability, per token.

    Args:
        probability: fraction of tokens that get randomly rerouted.
            NOT given a specific value by the paper ("The paper does not
            specify this implementation detail") -- 0.3 is a moderate
            default: high enough to meaningfully surface knowledge stored
            in rarely-used experts, low enough that generation still
            mostly reflects the model's actual learned routing (too high
            would make dreams closer to noise than useful synthetic data).

    How it works: we monkey-patch a `_force_random_prob` attribute onto
    each level for the duration of the `with` block, and restore it
    (defaulting to 0.0 / off) afterward -- guaranteed even if an
    exception occurs inside the block, since this is a context manager
    with a `finally`-equivalent teardown.
    """
    levels = []
    for module in model.modules():
        if hasattr(module, "active_mask"):  # duck-typing for ExpandableMemoryLevel
            levels.append(module)
            module._force_random_prob = probability

    try:
        yield
    finally:
        for level in levels:
            level._force_random_prob = 0.0


@torch.no_grad()
def generate_dreams(
    model: SleepLLMBackbone,
    context_ids: torch.Tensor,
    num_dreams: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    injection_probability: float = 0.3,
) -> torch.Tensor:
    """
    Generate `num_dreams` synthetic continuations of `context_ids`.

    Args:
        model: the SleepLLMBackbone (in eval mode is recommended by the
            caller, though we don't force it here -- some experiments
            may want dropout active during dreaming for diversity, which
            is a legitimate design choice left to the caller).
        context_ids: (seq_len,) 1D tensor -- the context C from the
            paper. Same context is used to seed all `num_dreams` samples
            (they diverge due to sampling randomness + expert injection).
        num_dreams: m in the paper's notation.
        max_new_tokens: how many new tokens to generate per dream.
        temperature: sampling temperature -- higher = more random. The
            paper doesn't specify a value; 1.0 (unmodified distribution)
            is the standard neutral default.
        top_k: restrict sampling to the top_k most likely tokens at each
            step -- standard practice to avoid sampling wildly
            improbable tokens that would make dreams incoherent noise
            rather than useful synthetic training data.
        injection_probability: passed to random_expert_injection().

    Returns:
        (num_dreams, seq_len + max_new_tokens) tensor of token ids.

    Time complexity: O(num_dreams * max_new_tokens) forward passes
    (autoregressive generation is inherently sequential per token) --
    this is the most expensive part of the Sleep pipeline per paper's
    own Appendix B.3 ("For each task, we sample 60 dreams and reject
    45"), which is why dream COUNT is a key cost/quality knob to tune
    down first if training is too slow on your laptop.
    """
    model.eval()
    batch_size = num_dreams
    seq_len = context_ids.shape[0]

    # Replicate the context across the batch dimension: all dreams start
    # from the same prompt, diverging via sampling + expert injection.
    generated = context_ids.unsqueeze(0).repeat(batch_size, 1)  # (num_dreams, seq_len)

    with random_expert_injection(model, probability=injection_probability):
        for _ in range(max_new_tokens):
            # Only feed the most recent max_sequence_length tokens if
            # we've grown past the model's context window -- a simple
            # sliding-window guard, since our tiny debug config caps
            # max_sequence_length at 128.
            window = generated[:, -model.config.model.max_sequence_length :]
            logits = model(window)  # (batch, cur_len, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering: zero out (set to -inf) all but the top_k
            # logits before sampling, per token.
            top_k_actual = min(top_k, next_token_logits.shape[-1])
            top_values, _ = torch.topk(next_token_logits, top_k_actual, dim=-1)
            min_top_value = top_values[:, -1].unsqueeze(-1)
            filtered_logits = next_token_logits.masked_fill(
                next_token_logits < min_top_value, float("-inf")
            )

            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            generated = torch.cat([generated, next_token], dim=1)

    return generated
