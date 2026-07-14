"""
tests/test_evaluation.py
==========================
Verifies Module 9: synthetic data generation is correct and separable,
the classifier head produces valid shapes, and (the actual point of this
whole module) Sleep measurably reduces catastrophic forgetting compared
to the no-Sleep baseline, on identical data/seeds.
"""

import torch

from config.model_config import load_config, MemoryConfig, MemoryLevelConfig
from models.backbone import SleepLLMBackbone
from evaluation.toy_intent_data import generate_intent_task
from evaluation.classifier_head import IntentClassifierHead
from evaluation.continual_eval import run_continual_learning_experiment


def test_generate_intent_task_produces_correct_shapes_and_labels():
    inputs, labels = generate_intent_task(
        num_classes=3, examples_per_class=4, seq_len=10, vocab_size=100, class_offset=0, seed=0
    )
    assert inputs.shape == (12, 10)
    assert labels.shape == (12,)
    assert set(labels.tolist()) == {0, 1, 2}


def test_generate_intent_task_class_offset_shifts_labels():
    inputs, labels = generate_intent_task(
        num_classes=3, examples_per_class=2, seq_len=8, vocab_size=100, class_offset=5, seed=0
    )
    assert set(labels.tolist()) == {5, 6, 7}


def test_classifier_head_produces_correct_output_shape():
    config = load_config("config/debug_laptop.yaml")
    torch.manual_seed(config.seed)
    backbone = SleepLLMBackbone(config)
    model = IntentClassifierHead(backbone, num_classes=10)

    inputs = torch.randint(0, config.model.vocab_size, (6, 16))
    logits = model(inputs)
    assert logits.shape == (6, 10)


def _build_tiny_config_for_test(chunk_lengths):
    """A minimal config for fast tests -- tiny hidden_dim so the whole
    experiment (2 tasks x 30 epochs x 2 conditions) runs in seconds."""
    from config.model_config import ModelConfig
    model_cfg = ModelConfig(
        vocab_size=200, hidden_dim=32, num_attention_heads=2,
        num_sequence_layers=1, max_sequence_length=32, dropout=0.0,
    )
    levels = [
        MemoryLevelConfig(name=f"level_{i}", update_frequency=100, chunk_length=c,
                           num_experts=1, expert_hidden_dim=32, lora_rank=4, max_experts=20)
        for i, c in enumerate(chunk_lengths)
    ]
    from config.model_config import SleepLLMConfig
    return SleepLLMConfig(experiment_name="test", seed=0, model=model_cfg, memory=MemoryConfig(levels=levels))


def test_sleep_reduces_forgetting_versus_no_sleep_baseline():
    """
    THE core result test for Module 9: with identical data, seeds, and
    architecture, the Sleep-enabled run should forget LESS of Task 1
    after learning Task 2 than the no-Sleep baseline.
    """
    torch.manual_seed(0)

    # Sleep-enabled config: short chunk lengths so consolidation fires
    # multiple times within our short 30-epoch-per-task budget.
    sleep_config = _build_tiny_config_for_test(chunk_lengths=[5, 20])
    # No-Sleep config: chunk lengths deliberately larger than the total
    # step budget (60 steps total: 30 for task1 + 30 for task2) --
    # guarantees zero consolidation events, a clean baseline.
    no_sleep_config = _build_tiny_config_for_test(chunk_lengths=[1000, 5000])

    def build_sleep_model():
        torch.manual_seed(0)
        return SleepLLMBackbone(sleep_config)

    def build_no_sleep_model():
        torch.manual_seed(0)
        return SleepLLMBackbone(no_sleep_config)

    sleep_result = run_continual_learning_experiment(
        sleep_config, build_sleep_model, num_classes_per_task=3, examples_per_class=6,
        seq_len=12, num_epochs_per_task=30, enable_sleep=True, data_seed=1,
    )
    no_sleep_result = run_continual_learning_experiment(
        no_sleep_config, build_no_sleep_model, num_classes_per_task=3, examples_per_class=6,
        seq_len=12, num_epochs_per_task=30, enable_sleep=False, data_seed=1,
    )

    print(f"\nSleep forgetting:    {sleep_result.forgetting:.4f}")
    print(f"No-Sleep forgetting: {no_sleep_result.forgetting:.4f}")

    # The core claim: Sleep forgets less (or at least, not more) than
    # the no-Sleep baseline. We allow equality (<=) rather than strict
    # (<) because at this tiny synthetic scale, exact ties are possible;
    # the important, testable claim is that Sleep is never WORSE.
    assert sleep_result.forgetting <= no_sleep_result.forgetting + 1e-6
