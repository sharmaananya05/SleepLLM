"""
trainer package
=================
Module 8 (part 2): SleepLLMTrainer -- wires wake-phase language modeling
together with scheduled sleep-phase memory consolidation, using every
prior module (config, models, memory, sleep, distillation).
"""

from trainer.trainer import SleepLLMTrainer, TrainStepResult, SleepEventResult

__all__ = ["SleepLLMTrainer", "TrainStepResult", "SleepEventResult"]
