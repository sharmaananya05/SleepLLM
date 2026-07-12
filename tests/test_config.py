"""
tests/test_config.py
=====================
Unit tests for configs/model_config.py.

How to run:
    cd SleepLLM
    pytest tests/test_config.py -v

What "good" test design looks like here (and why):
- We test the HAPPY PATH (valid YAML loads into the right objects) AND the
  FAILURE PATHS (invalid configs raise clear errors) -- research code that
  is only tested on the happy path is how you get a silent bug 3 weeks into
  training that invalidates your results.
- We don't test PyTorch/model behavior here (there's no model yet!) -- this
  file's ONLY job is to validate Module 1 (configuration) in isolation.
  This is a core software-engineering principle: unit tests should test ONE
  unit. Model-level tests belong in tests/test_models.py (a later module).
"""

import pytest

from config.model_config import (
    MemoryConfig,
    MemoryLevelConfig,
    load_config,
)


def test_load_debug_config_succeeds():
    """The shipped debug_laptop.yaml must load without error."""
    cfg = load_config("config/debug_laptop.yaml")

    assert cfg.experiment_name == "sleepllm_debug_laptop"
    assert cfg.model.hidden_dim == 128
    assert cfg.memory.num_levels == 2
    assert cfg.memory.chunk_lengths == [16, 64]


def test_memory_config_rejects_non_increasing_chunk_lengths():
    """
    Paper Section 2.2: chunk lengths must be ordered fast -> slow
    (C^(ell) strictly increasing). A config that violates this should fail
    at construction time, not silently produce a broken sleep schedule.
    """
    bad_levels = [
        MemoryLevelConfig(name="a", update_frequency=100, chunk_length=64),
        MemoryLevelConfig(name="b", update_frequency=10, chunk_length=32),  # smaller!
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        MemoryConfig(levels=bad_levels)


def test_memory_config_rejects_non_divisible_chunk_lengths():
    """
    Paper Section 2.2 assumption: C^(ell) divisible by C^(ell-1).
    64 does not evenly divide 100, so this must raise.
    """
    bad_levels = [
        MemoryLevelConfig(name="a", update_frequency=100, chunk_length=64),
        MemoryLevelConfig(name="b", update_frequency=10, chunk_length=100),
    ]
    with pytest.raises(ValueError, match="divisible"):
        MemoryConfig(levels=bad_levels)


def test_missing_config_file_raises_filenotfounderror():
    with pytest.raises(FileNotFoundError):
        load_config("config/this_file_does_not_exist.yaml")
