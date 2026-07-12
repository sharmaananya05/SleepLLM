"""
configs package
================
Central place for all experiment configuration dataclasses and loaders.

Why a package (not a single file)?
-----------------------------------
As SleepLLM grows, we'll have separate config groups (model architecture,
sleep schedule, dreaming/RL hyperparameters, training loop, data). Keeping
them in one `configs/` package -- one file per concern -- means you can
import just what you need:

    from config.model_config import SleepLLMConfig

and it keeps any single file from becoming a 1000-line God-object.
"""

from config.model_config import (
    ModelConfig,
    MemoryLevelConfig,
    MemoryConfig,
    SleepLLMConfig,
    load_config,
)

__all__ = [
    "ModelConfig",
    "MemoryLevelConfig",
    "MemoryConfig",
    "SleepLLMConfig",
    "load_config",
]
