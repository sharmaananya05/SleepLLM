"""
dreaming/dream_selection.py
=============================
Selects which generated dreams are worth training on (paper Section 3.4):

    "we reject some of the generated dreams and only keep the samples
    with the most potential ... for each dream, DREAM(i), we assign an
    importance score w(i) and select Top-k dreams with highest
    importance score along with b random samples to maintain diversity."

SCOPED SIMPLIFICATION (flagged explicitly, per project instructions):
The paper defines the importance score as a GRADIENT norm:
    g_DR(i) = grad_theta L_SFT(DREAM(i), theta)
This requires a full backward pass per CANDIDATE dream, before you've
even decided which ones are worth training on -- i.e. you pay the full
training cost for every dream just to find out which ones to discard.
We instead score dreams by their LANGUAGE MODELING LOSS under the
current model (forward-pass only, no backward needed for scoring) as a
cheaper proxy: loss and gradient magnitude are strongly correlated in
practice (high-loss examples generally produce larger gradients), and
avoiding a wasted backward pass on the ~75% of dreams the paper's own
Appendix B.3 example rejects (45 of 60) is a meaningful compute saving
on laptop hardware. This is a documented engineering trade-off, not a
paper-specified detail.
"""

import random

import torch
import torch.nn.functional as F


def compute_dream_importance_scores(
    model: torch.nn.Module,
    dream_token_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        model: the SleepLLMBackbone.
        dream_token_ids: (num_dreams, seq_len) -- output of
            dream_generator.generate_dreams().

    Returns:
        (num_dreams,) tensor of importance scores -- HIGHER means more
        "surprising" to the model (higher loss), matching the intuition
        that high-gradient-norm examples (what the paper actually wants)
        tend to be ones the model currently fits poorly.

    Time complexity: O(num_dreams * seq_len * hidden_dim) for ONE forward
    pass over all dreams (batched) -- versus O(num_dreams) BACKWARD
    passes for the paper's exact gradient-norm version. This is the
    entire point of the simplification: an order of magnitude cheaper.
    """
    model.eval()
    with torch.no_grad():
        logits = model(dream_token_ids[:, :-1])  # predict token t from tokens < t
        targets = dream_token_ids[:, 1:]

        # Per-token cross-entropy, then mean per dream (not summed -- so
        # dream LENGTH doesn't bias the score; a longer dream shouldn't
        # automatically look "more important" just by accumulating loss).
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)  # (num_dreams, seq_len - 1)

        per_dream_loss = losses.mean(dim=1)  # (num_dreams,)

    return per_dream_loss


def select_dreams(
    dream_token_ids: torch.Tensor,
    importance_scores: torch.Tensor,
    top_k: int,
    num_random: int,
) -> torch.Tensor:
    """
    Select the top_k highest-importance dreams PLUS num_random additional
    random dreams for diversity, per the paper's exact selection rule.

    Args:
        dream_token_ids: (num_dreams, seq_len)
        importance_scores: (num_dreams,) from compute_dream_importance_scores()
        top_k: number of highest-scoring dreams to always keep.
        num_random: number of ADDITIONAL random dreams from the
            remaining pool, for diversity (paper: "along with b random
            samples to maintain diversity" -- b = num_random here).

    Returns:
        (top_k + num_random, seq_len) tensor of selected dream token ids
        (fewer if num_dreams < top_k + num_random).

    Time complexity: O(num_dreams log num_dreams) for the top-k sort,
    negligible compared to generation/scoring cost.
    """
    num_dreams = dream_token_ids.shape[0]
    top_k = min(top_k, num_dreams)

    sorted_indices = torch.argsort(importance_scores, descending=True)
    top_k_indices = sorted_indices[:top_k].tolist()

    remaining_indices = sorted_indices[top_k:].tolist()
    num_random = min(num_random, len(remaining_indices))
    random_indices = random.sample(remaining_indices, num_random) if num_random > 0 else []

    selected_indices = top_k_indices + random_indices
    return dream_token_ids[selected_indices]
