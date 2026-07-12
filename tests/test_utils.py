"""
tests/test_utils.py
====================
Verifies Module 2: seeding reproducibility + logger correctness.
"""

import logging

import torch

from utils.seed import set_seed
from utils.logging_utils import get_logger


def test_set_seed_gives_reproducible_torch_randn():
    """Same seed -> identical tensor, every time. This IS the whole point."""
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_different_seeds_give_different_results():
    """Sanity check the OTHER direction: seeding isn't a no-op."""
    set_seed(1)
    a = torch.randn(5)
    set_seed(2)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_get_logger_returns_same_instance_for_same_name(tmp_path):
    """Calling get_logger twice with the same name must not duplicate handlers."""
    log_file = tmp_path / "test.log"
    logger1 = get_logger("sleepllm.test_module", log_file=log_file)
    n_handlers_after_first_call = len(logger1.handlers)

    logger2 = get_logger("sleepllm.test_module", log_file=log_file)
    n_handlers_after_second_call = len(logger2.handlers)

    assert logger1 is logger2
    assert n_handlers_after_first_call == n_handlers_after_second_call

    logger1.info("hello from test")
    assert log_file.exists()
    assert "hello from test" in log_file.read_text()
