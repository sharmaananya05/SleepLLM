"""
tests/test_data.py
====================
Verifies Module 8 (data pipeline): chunking is correct and offline-testable.
NOTE: load_wikitext2_token_ids() is NOT tested here -- it requires
internet access to download WikiText-2, which this test suite must run
without (see data/dataset.py's docstring). Test that function manually
on your laptop instead.
"""

import torch

from data.dataset import TextChunkDataset, build_dataloader


def test_chunking_produces_correct_shapes():
    token_ids = torch.arange(100)  # tokens 0..99
    dataset = TextChunkDataset(token_ids, seq_len=10)
    input_ids, target_ids = dataset[0]
    assert input_ids.shape == (10,)
    assert target_ids.shape == (10,)


def test_target_is_input_shifted_by_one():
    token_ids = torch.arange(20)
    dataset = TextChunkDataset(token_ids, seq_len=5)
    input_ids, target_ids = dataset[0]
    assert torch.equal(input_ids, torch.tensor([0, 1, 2, 3, 4]))
    assert torch.equal(target_ids, torch.tensor([1, 2, 3, 4, 5]))


def test_num_chunks_is_computed_correctly():
    token_ids = torch.arange(101)  # 101 tokens
    dataset = TextChunkDataset(token_ids, seq_len=10)
    # (101 - 1) // 10 = 10 chunks
    assert len(dataset) == 10


def test_dataloader_yields_correct_batch_shape():
    token_ids = torch.arange(200)
    loader = build_dataloader(token_ids, seq_len=8, batch_size=4, shuffle=False)
    input_batch, target_batch = next(iter(loader))
    assert input_batch.shape == (4, 8)
    assert target_batch.shape == (4, 8)


def test_rejects_non_1d_token_ids():
    bad_ids = torch.randint(0, 100, (2, 50))  # 2D, should be 1D
    try:
        TextChunkDataset(bad_ids, seq_len=10)
        assert False, "should have raised ValueError"
    except ValueError:
        pass
