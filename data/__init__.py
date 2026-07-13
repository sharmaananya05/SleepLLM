"""
data package
=============
Module 8 (part 1): the wake-phase data pipeline. TextChunkDataset and
build_dataloader are tokenizer-agnostic and fully testable offline;
load_wikitext2_token_ids() requires internet and real usage on your
laptop, not in an automated test.
"""

from data.dataset import TextChunkDataset, build_dataloader, load_wikitext2_token_ids

__all__ = ["TextChunkDataset", "build_dataloader", "load_wikitext2_token_ids"]
