"""
evaluation/toy_intent_data.py
================================
A small, SYNTHETIC, offline intent-classification dataset, standing in
for CLINC150 (paper Section 4.1, Figure 3) so Module 9 can run and be
tested without needing internet access to download the real CLINC150
dataset.

"The paper does not specify this implementation detail" (for OUR
scaled-down reproduction, not the paper itself): the paper trains
Llama-3B/8B on real CLINC150/Banking/DBpedia. At our laptop scale (a
6.7M-parameter model, no pretrained language understanding), using the
REAL CLINC150 text would mostly test whether our tiny from-scratch model
can do English intent classification at all -- a much harder, noisier
problem than what we actually want to demonstrate here, which is
narrower: does Sleep protect old TASK knowledge better than plain
continual fine-tuning, when both are trained on the SAME distribution
shift. A synthetic task with clean, controllable class-token patterns
isolates that specific question cleanly. Swapping in real CLINC150 text
(via Hugging Face `datasets`, same pattern as data/dataset.py's
WikiText-2 loader) is a natural documented "possible improvement" once
this scaffold is confirmed to work.

Design: each "intent" is represented by a short signature pattern of
token ids repeated with noise -- e.g. intent 3's examples all start with
token id (100 + 3*10) followed by random filler tokens. A classifier
that has genuinely learned intent 3 should recognize this signature
regardless of the random filler; this cleanly mimics "intents have a
consistent underlying pattern amid surface variation," which is the
actual property class-incremental learning experiments care about.
"""

from __future__ import annotations

import torch


def generate_intent_task(
    num_classes: int,
    examples_per_class: int,
    seq_len: int,
    vocab_size: int,
    class_offset: int = 0,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates one "task" (a batch of classes) of synthetic intent data.

    Args:
        num_classes: how many distinct intents in this task.
        examples_per_class: how many examples per intent.
        seq_len: token sequence length per example.
        vocab_size: must match the model's config.model.vocab_size --
            filler tokens are sampled from the FULL vocab, but the
            signature token is reserved in a low, fixed range (see
            below) to guarantee it never collides with random filler
            noise for reasonable vocab sizes.
        class_offset: shifts intent LABELS (not token ids) up by this
            amount -- used so "task 2" of a continual sequence has
            labels 5-9 instead of colliding with "task 1"'s 0-4, exactly
            matching class-INCREMENTAL learning's setup (new tasks add
            NEW classes, don't overwrite old ones).
        seed: for reproducibility -- IMPORTANT for a fair Sleep vs
            no-Sleep comparison: both conditions must see EXACTLY the
            same data, so this must be fixed and reused across both
            runs (Module 9's continual_eval.py enforces this).

    Returns:
        (inputs, labels): inputs is (num_classes*examples_per_class, seq_len)
        token ids; labels is (num_classes*examples_per_class,) int labels
        in [class_offset, class_offset+num_classes).
    """
    generator = torch.Generator().manual_seed(seed)
    all_inputs = []
    all_labels = []

    for local_class_id in range(num_classes):
        # Reserve a small, fixed signature token per class: guaranteed
        # distinguishable, low-index range (well below any reasonable
        # vocab_size), so classes never accidentally share a signature.
        signature_token = 10 + local_class_id

        for _ in range(examples_per_class):
            filler = torch.randint(
                0, vocab_size, (seq_len - 1,), generator=generator
            )
            example = torch.cat([torch.tensor([signature_token]), filler])
            all_inputs.append(example)
            all_labels.append(local_class_id + class_offset)

    inputs = torch.stack(all_inputs)
    labels = torch.tensor(all_labels, dtype=torch.long)

    # Shuffle so classes aren't grouped in order (a classifier shouldn't
    # be able to cheat off of batch position).
    perm = torch.randperm(len(labels), generator=generator)
    return inputs[perm], labels[perm]
