"""
main.py
=========
Module 10: the single command-line entry point for the whole SleepLLM
project. Every module built in Modules 1-9 is reachable from here --
this file contains NO new logic of its own, only wiring + a CLI, so
that reviewers (and you, in a demo) can run the whole system with one
command instead of hand-writing a Python script each time.

Usage:
    python main.py train --config config/debug_laptop.yaml --steps 100
    python main.py demo-continual-learning
    python main.py demo-unlearning
"""

from __future__ import annotations

import argparse

import torch

from config.model_config import load_config
from models.backbone import SleepLLMBackbone
from trainer.trainer import SleepLLMTrainer
from utils.seed import set_seed
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def cmd_train(args: argparse.Namespace) -> None:
    """
    Runs the wake/sleep training loop (Module 8) on synthetic token data
    for a fixed number of steps -- a quick way to SEE the Sleep paradigm
    running end-to-end without needing a real dataset wired in.

    For real training on real text (e.g. WikiText-2), see
    data/dataset.py's load_wikitext2_token_ids() and wire it in here in
    place of the synthetic torch.randint() batches below -- left as a
    documented extension point rather than hardcoded, since the right
    dataset depends on what you're trying to demonstrate (see
    evaluation/ for the class-incremental-learning-specific pipeline).
    """
    config = load_config(args.config)
    set_seed(config.seed)
    logger.info("Loaded config '%s' (experiment: %s)", args.config, config.experiment_name)

    model = SleepLLMBackbone(config)
    logger.info("Built model: %s parameters", f"{model.num_parameters():,}")

    trainer = SleepLLMTrainer(model, config, wake_lr=args.lr)

    for step in range(1, args.steps + 1):
        input_ids = torch.randint(0, config.model.vocab_size, (args.batch_size, args.seq_len))
        target_ids = torch.randint(0, config.model.vocab_size, (args.batch_size, args.seq_len))
        result = trainer.train_step(input_ids, target_ids)

        if step % args.log_every == 0 or result.sleep_events:
            sleep_info = f" | sleep_events={len(result.sleep_events)}" if result.sleep_events else ""
            logger.info("step=%d wake_loss=%.4f%s", result.step, result.wake_loss, sleep_info)

    logger.info("Training finished. Final active experts per level:")
    for block_idx, block in enumerate(model.blocks):
        for level in block.cms.levels:
            logger.info("  block=%d level=%s active_experts=%d", block_idx, level.config.name, level.num_active_experts)


def cmd_demo_continual_learning(args: argparse.Namespace) -> None:
    """
    Runs Module 9's class-incremental-learning experiment: trains on
    Task 1, then Task 2, and reports how much Task 1 knowledge each
    condition (Sleep-enabled vs. plain continual fine-tuning) retained --
    this is the project's core "results" demonstration.
    """
    from config.model_config import SleepLLMConfig, ModelConfig, MemoryConfig, MemoryLevelConfig
    from evaluation.continual_eval import run_continual_learning_experiment

    model_cfg = ModelConfig(
        vocab_size=200, hidden_dim=64, num_attention_heads=4,
        num_sequence_layers=1, max_sequence_length=32, dropout=0.0,
    )

    def make_config(chunk_lengths):
        levels = [
            MemoryLevelConfig(name=f"level_{i}", update_frequency=100, chunk_length=c,
                               num_experts=1, expert_hidden_dim=64, lora_rank=4, max_experts=20)
            for i, c in enumerate(chunk_lengths)
        ]
        return SleepLLMConfig(experiment_name="demo", seed=0, model=model_cfg, memory=MemoryConfig(levels=levels))

    sleep_config = make_config([5, 20])
    no_sleep_config = make_config([10000, 50000])  # effectively never fires

    def build_sleep_model():
        set_seed(0)
        return SleepLLMBackbone(sleep_config)

    def build_no_sleep_model():
        set_seed(0)
        return SleepLLMBackbone(no_sleep_config)

    logger.info("Running Sleep-enabled condition...")
    sleep_result = run_continual_learning_experiment(
        sleep_config, build_sleep_model, num_classes_per_task=3, examples_per_class=6,
        seq_len=12, num_epochs_per_task=30, enable_sleep=True, data_seed=1,
    )
    logger.info("Running no-Sleep baseline condition...")
    no_sleep_result = run_continual_learning_experiment(
        no_sleep_config, build_no_sleep_model, num_classes_per_task=3, examples_per_class=6,
        seq_len=12, num_epochs_per_task=30, enable_sleep=False, data_seed=1,
    )

    print("\n" + "=" * 60)
    print("CLASS-INCREMENTAL LEARNING RESULTS")
    print("=" * 60)
    for label, result in [("Sleep", sleep_result), ("No-Sleep (baseline)", no_sleep_result)]:
        print(f"\n{label}:")
        print(f"  Task 1 accuracy right after Task 1:  {result.task1_acc_after_task1:.2%}")
        print(f"  Task 1 accuracy after also learning Task 2: {result.task1_acc_after_task2:.2%}")
        print(f"  Task 2 accuracy after Task 2:         {result.task2_acc_after_task2:.2%}")
        print(f"  Forgetting (lower is better):          {result.forgetting:.4f}")
    print("=" * 60)


def cmd_demo_unlearning(args: argparse.Namespace) -> None:
    """
    Demonstrates both unlearning methods on a small trained classifier:
    (1) structural unlearning -- deactivate a specific memory expert,
    (2) gradient-ascent unlearning -- fine-tune to forget a target class
    while preserving performance on everything else.
    """
    import torch.nn.functional as F
    from config.model_config import load_config
    from evaluation.classifier_head import IntentClassifierHead
    from evaluation.toy_intent_data import generate_intent_task
    from memory.unlearning import gradient_ascent_unlearn

    config = load_config(args.config)
    set_seed(config.seed)
    backbone = SleepLLMBackbone(config)
    model = IntentClassifierHead(backbone, num_classes=6)

    forget_inputs, forget_labels = generate_intent_task(
        num_classes=2, examples_per_class=8, seq_len=12,
        vocab_size=config.model.vocab_size, class_offset=0, seed=1,
    )
    retain_inputs, retain_labels = generate_intent_task(
        num_classes=2, examples_per_class=8, seq_len=12,
        vocab_size=config.model.vocab_size, class_offset=2, seed=2,
    )

    logger.info("Training classifier on both 'forget' and 'retain' classes...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    all_inputs = torch.cat([forget_inputs, retain_inputs])
    all_labels = torch.cat([forget_labels, retain_labels])
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(all_inputs), all_labels)
        loss.backward()
        optimizer.step()

    logger.info("Running gradient-ascent unlearning on the 'forget' classes...")
    result = gradient_ascent_unlearn(
        model, lambda m, x: m(x), forget_inputs, forget_labels,
        retain_inputs, retain_labels, steps=15, lr=1e-3,
    )

    print("\n" + "=" * 60)
    print("UNLEARNING RESULTS")
    print("=" * 60)
    print(f"Forget-set loss:  {result.forget_loss_before:.4f} -> {result.forget_loss_after:.4f}  (should INCREASE)")
    print(f"Retain-set loss:  {result.retain_loss_before:.4f} -> {result.retain_loss_after:.4f}  (should stay LOW)")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SleepLLM: memory consolidation and unlearning for language models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the wake/sleep training loop")
    train_parser.add_argument("--config", default="config/debug_laptop.yaml")
    train_parser.add_argument("--steps", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--seq-len", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.set_defaults(func=cmd_train)

    demo_cl_parser = subparsers.add_parser("demo-continual-learning", help="Run the Sleep vs no-Sleep forgetting comparison")
    demo_cl_parser.set_defaults(func=cmd_demo_continual_learning)

    demo_ul_parser = subparsers.add_parser("demo-unlearning", help="Run the unlearning demonstration")
    demo_ul_parser.add_argument("--config", default="config/debug_laptop.yaml")
    demo_ul_parser.set_defaults(func=cmd_demo_unlearning)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
