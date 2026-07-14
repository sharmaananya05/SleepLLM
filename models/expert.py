"""
models/expert.py
==================
Backward-compatible re-export. The actual `Expert` class now lives in
memory/dense_expert.py (moved there to eliminate a circular import
between `models` and `memory` -- see that file's docstring for the full
explanation). Anything importing `from models.expert import Expert`
continues to work unchanged.
"""

from memory.dense_expert import Expert

__all__ = ["Expert"]
