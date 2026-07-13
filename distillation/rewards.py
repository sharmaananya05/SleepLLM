"""
distillation/rewards.py
=========================
The reward functions for Learning-to-Imitate (LTI), paper Section 3.3,
Equations 3-4:

    r(dhat, d, LM) = gamma * r_sem(dhat, d, LM) + (1-gamma) * r_abs(dhat, d, LM)

- r_abs: exact formula given by the paper (Eq. 4), Levenshtein-distance
  based token-level similarity.
- r_sem: "The paper does not specify this implementation detail" -- it
  says only "a reward model that is frozen and rewards the student with
  1 (resp. 0), if the semantic of dhat and d are the same." No specific
  reward model architecture is given. We use the frozen TEACHER model's
  own mean-pooled hidden states (cosine similarity, thresholded) as a
  semantic-similarity proxy -- this keeps the "frozen judge" property the
  paper requires, without needing a separate trained classifier or
  external API, which would be impractical for a laptop-only project.
"""

import Levenshtein
import torch
import torch.nn.functional as F


def r_abs(student_ids: torch.Tensor, teacher_ids: torch.Tensor, z0: float = 0.5) -> float:
    """
    Absolute (Levenshtein-based) reward, EXACTLY matching paper Eq. 4:

        r_abs = 1 - z(dhat, d) / max(|dhat|, |d|)   if z(dhat, d) <= z0
                0                                     otherwise

    where z(.,.) is Levenshtein edit distance.

    Args:
        student_ids: 1D tensor of token ids, the student's generated
            continuation (dhat in the paper).
        teacher_ids: 1D tensor of token ids, the teacher's generated
            continuation (d in the paper).
        z0: similarity threshold. NOT specified numerically by the paper
            -- "The paper does not specify this implementation detail."
            We default to 0.5 (at most half the tokens may differ,
            edit-distance-wise, for ANY partial credit) as a reasonable
            middle ground: too strict (small z0) gives near-zero reward
            almost always early in training (no learning signal); too
            loose (large z0) rewards very dissimilar sequences.

    Time complexity: O(len(student_ids) * len(teacher_ids)) -- standard
    Levenshtein DP. For short generated sequences (our debug config caps
    seq_len at 128), this is negligible.
    """
    student_list = student_ids.tolist()
    teacher_list = teacher_ids.tolist()

    # python-Levenshtein operates on strings, not lists of ints -- we map
    # each token id to a unique unicode character so Levenshtein.distance
    # treats each TOKEN as one edit-unit, not each digit of its id.
    student_str = "".join(chr(t) for t in student_list)
    teacher_str = "".join(chr(t) for t in teacher_list)

    edit_dist = Levenshtein.distance(student_str, teacher_str)
    max_len = max(len(student_list), len(teacher_list), 1)  # avoid div-by-0
    normalized_dist = edit_dist / max_len

    if normalized_dist <= z0:
        return 1.0 - normalized_dist
    return 0.0


def r_sem(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    threshold: float = 0.8,
) -> float:
    """
    Semantic-similarity reward: cosine similarity between mean-pooled
    hidden states, thresholded to a binary-ish {0, 1} signal, per the
    paper's description of r_sem as rewarding "1 (resp. 0) if the
    semantic ... are the same (resp. otherwise)."

    Args:
        student_hidden: (seq_len, hidden_dim) -- the model's own hidden
            states (NOT logits) for the student's generated continuation.
            "Own hidden states" because we're using the frozen teacher
            model itself as the semantic judge (see module docstring).
        teacher_hidden: (seq_len, hidden_dim) -- same, for the teacher's
            generated continuation.
        threshold: cosine similarity threshold above which we consider
            the two sequences "semantically the same." Not specified by
            the paper -- 0.8 is a common default for embedding-similarity
            thresholds in the sentence-similarity literature, but this is
            an engineering choice, not a paper-derived number.

    Returns:
        1.0 if mean-pooled cosine similarity >= threshold, else 0.0.
    """
    student_pooled = student_hidden.mean(dim=0)  # (hidden_dim,)
    teacher_pooled = teacher_hidden.mean(dim=0)

    cosine_sim = F.cosine_similarity(
        student_pooled.unsqueeze(0), teacher_pooled.unsqueeze(0)
    ).item()

    return 1.0 if cosine_sim >= threshold else 0.0


def combined_lti_reward(
    student_ids: torch.Tensor,
    teacher_ids: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    gamma: float = 0.5,
    z0: float = 0.5,
    sem_threshold: float = 0.8,
) -> float:
    """
    The full LTI reward, paper Eq. 3:
        r = gamma * r_sem + (1 - gamma) * r_abs

    gamma: NOT specified numerically by the paper -- we default to 0.5
    (equal weighting) as the most neutral, defensible starting point
    absent a specified value. Worth tuning as a hyperparameter later.
    """
    sem = r_sem(student_hidden, teacher_hidden, threshold=sem_threshold)
    abs_reward = r_abs(student_ids, teacher_ids, z0=z0)
    return gamma * sem + (1 - gamma) * abs_reward
