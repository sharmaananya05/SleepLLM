"""
data/dataset.py
=================
The data pipeline for "wake" phase training: turns raw token ids into
fixed-length (input, target) chunks for standard next-token-prediction
language modeling, plus an optional WikiText-2 loader for real training
on your laptop.

Design choice: this module is TOKENIZER-AGNOSTIC. `TextChunkDataset`
just takes a 1D tensor of already-tokenized ids -- it doesn't care
whether they came from GPT-2's BPE tokenizer, a custom one, or synthetic
test data. This means:
1. We can unit-test the CHUNKING logic (this file) completely offline,
   without needing internet access to download a tokenizer or dataset.
2. On your laptop (which has internet), you plug in a real
   AutoTokenizer.from_pretrained("gpt2") and load_wikitext2_token_ids()
   below to get real training data.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader


class TextChunkDataset(Dataset):
    """
    Wraps a long 1D sequence of token ids into non-overlapping
    (input_ids, target_ids) chunks of length seq_len, for standard
    next-token-prediction training.

    Args:
        token_ids: 1D LongTensor of the full tokenized corpus.
        seq_len: length of each training chunk (should match, or be <=,
            config.model.max_sequence_length).

    Why NON-overlapping chunks (not a sliding window)?
    Sliding windows (stride < seq_len) give more training examples per
    token of raw text, at the cost of redundant computation (the model
    sees overlapping context many times). For a laptop-scale first
    training loop, non-overlapping chunks are simpler to reason about
    and faster per epoch -- a documented engineering choice, not a paper
    requirement (the paper doesn't specify a chunking scheme for
    "wake"-phase language modeling at all, since that's just standard
    LM pretraining, not a Sleep-specific contribution).
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int) -> None:
        if token_ids.dim() != 1:
            raise ValueError(f"token_ids must be 1D, got shape {token_ids.shape}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        self.token_ids = token_ids
        self.seq_len = seq_len
        # We need seq_len+1 tokens per example (input = tokens[i:i+seq_len],
        # target = tokens[i+1:i+seq_len+1]), so the last usable start index
        # is len(token_ids) - seq_len - 1.
        self.num_chunks = max(0, (len(token_ids) - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Time complexity: O(seq_len) per item (tensor slicing) --
        negligible; PyTorch's DataLoader parallelizes this across
        workers if num_workers > 0.
        """
        start = idx * self.seq_len
        input_ids = self.token_ids[start : start + self.seq_len]
        target_ids = self.token_ids[start + 1 : start + self.seq_len + 1]
        return input_ids, target_ids


def build_dataloader(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience wrapper: TextChunkDataset -> a ready-to-iterate DataLoader."""
    dataset = TextChunkDataset(token_ids, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def load_wikitext2_token_ids(tokenizer, split: str = "train", max_tokens: int | None = None) -> torch.Tensor:
    """
    Downloads WikiText-2 (via Hugging Face `datasets`) and tokenizes it
    with the given tokenizer. Requires internet access -- run this on
    your laptop (not in a network-restricted sandbox).

    Args:
        tokenizer: any object with an `encode(text) -> list[int]` method
            (e.g. a Hugging Face AutoTokenizer, or GPT2Tokenizer).
        split: "train", "validation", or "test".
        max_tokens: optionally truncate to the first N tokens -- useful
            on a laptop to keep a training run short while testing the
            pipeline, before committing to a full epoch.

    Returns:
        1D LongTensor of token ids, ready for TextChunkDataset.

    Why WikiText-2 specifically (an engineering choice, not paper-driven):
    it's small (~4MB), freely licensed, requires no authentication, and
    is a standard, well-understood language modeling benchmark --
    appropriate for proving the Sleep training LOOP works correctly,
    though it is NOT one of the paper's own evaluation datasets (those
    are CLINC/Banking/DBpedia for classification and LongHealth/QASPER/
    MK-NIAH for long-context -- Module 9 will address matching the
    paper's actual benchmarks).
    """
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(row["text"] for row in raw if row["text"].strip())

    token_ids = tokenizer.encode(text)
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    return torch.tensor(token_ids, dtype=torch.long)
