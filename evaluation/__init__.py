"""
evaluation package
=====================
Module 9: proves the paper's core claim on a laptop-scale reproduction --
Class-Incremental Learning (Section 4.1, Figure 3). Compares a Sleep-
enabled backbone against a plain continually-fine-tuned baseline on the
same synthetic intent-classification task sequence, measuring
catastrophic forgetting directly.
"""

from evaluation.toy_intent_data import generate_intent_task
from evaluation.classifier_head import IntentClassifierHead
from evaluation.continual_eval import run_continual_learning_experiment, ContinualLearningResult

__all__ = [
    "generate_intent_task",
    "IntentClassifierHead",
    "run_continual_learning_experiment",
    "ContinualLearningResult",
]
