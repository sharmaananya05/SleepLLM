"""
utils/seed.py
==============
Sets every source of randomness in the stack to a fixed seed, so that
re-running an experiment with the same config produces IDENTICAL results.

Why this matters for a research project:
If your professor asks "run it again and show me the loss curve," and it
comes out different every time, that's not proof of anything -- it could
be a bug, or luck. Fixing all RNGs turns "did my change help?" into a
well-posed question.

The four independent RNGs in a typical PyTorch project:
1. Python's built-in `random` module      -- used by some data shuffling
2. NumPy's RNG                            -- used by many data pipelines
3. PyTorch's CPU RNG                      -- weight init, CPU-side sampling
4. PyTorch's CUDA RNG                     -- GPU-side sampling, dropout, etc.
Missing #4 is the most common bug: code works "reproducibly" on CPU, then
mysteriously isn't reproducible once you move to a GPU (which is exactly
what will happen when you move to Lightning AI).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Seed every RNG source used in this project.

    Args:
        seed: The seed value (from SleepLLMConfig.seed).
        deterministic: If True, also force PyTorch's CUDA backend into
            deterministic mode (cuDNN algorithms that give bit-identical
            results run by run). This can be ~10-20% SLOWER than
            non-deterministic mode, because PyTorch has to avoid certain
            highly-optimized-but-nondeterministic CUDA kernels. Trade-off:
            set True while debugging/writing your report (you want
            identical numbers to point to), set False during final,
            long training runs on Lightning AI if you need every bit of
            speed and don't need bit-identical reruns.

    Time complexity: O(1) -- this just flips some global flags and RNG
    states, independent of model or data size.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # seeds ALL GPUs, safe even with 1 GPU

    # PYTHONHASHSEED affects e.g. dict/set iteration order across
    # processes -- rarely matters here, but costs nothing to fix.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cudnn.benchmark=True (the default in many tutorials) lets cuDNN
        # auto-tune the fastest convolution algorithm for your exact input
        # shape -- but that auto-tuning process is itself non-deterministic
        # and can pick different algorithms across runs. We turn it off
        # here for reproducibility; SleepLLM barely uses convolutions
        # anyway (it's Transformer-based), so the speed cost is negligible.
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
