"""
utils/logging_utils.py
========================
A single, project-wide logging setup, used by every module (trainer,
sleep scheduler, dreaming, distillation) instead of scattered print()
calls.

Why not just use print()?
--------------------------
1. print() has no severity levels -- you can't say "only show me warnings
   and errors" when scanning a 6-hour Lightning AI log.
2. print() has no timestamps -- you can't tell how long a sleep-phase
   consolidation step took relative to the surrounding wake-phase steps.
3. print() can't easily be redirected to BOTH the console (for you,
   watching live) AND a file (for your project report / debugging later)
   at the same time.

How researchers normally do this:
Most PyTorch research repos either use Python's built-in `logging` module
(what we do here -- zero extra dependencies) or a wrapper like Weights &
Biases / TensorBoard for run-tracking. We start with `logging` because
it's transparent enough to explain line-by-line; adding W&B later is a
natural "possible improvement" once you're running many experiments on
Lightning AI and want to compare runs in a dashboard instead of reading
raw log files.
"""

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Args:
        name: Usually `__name__` of the calling module, e.g.
            "sleep.scheduler" -- this shows up in every log line so you
            know WHICH module produced it.
        log_file: If given, log lines are ALSO written to this file
            (in addition to the console). Pass e.g.
            "checkpoints/sleepllm_debug/train.log".
        level: Minimum severity to show. logging.INFO is the standard
            default; use logging.DEBUG while actively debugging a module
            to see much more detail (e.g. every consolidation trigger).

    Returns:
        A standard `logging.Logger` -- use it like:
            logger = get_logger(__name__)
            logger.info("Sleep phase started at step %d", step)
            logger.warning("Chunk length not divisible by ...")
            logger.error("Consolidation failed: %s", exc)

    Why `logger.getLogger(name)` and not just creating a new object every
    time: Python's logging module caches loggers by name internally, so
    calling get_logger("sleep.scheduler") from multiple files all returns
    the SAME underlying logger object/configuration -- this avoids
    duplicate handlers (which would print every message twice, a very
    common logging bug).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. this function was called twice for the
        # same module name) -- don't add duplicate handlers, which would
        # cause every log line to print multiple times.
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler: what you see live in your VS Code terminal.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler: a permanent record for your project report.
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent messages from ALSO being handled by the Python root logger
    # (which would otherwise cause double-printing in some environments,
    # e.g. Jupyter notebooks on Lightning AI Studio).
    logger.propagate = False

    return logger
