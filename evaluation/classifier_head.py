"""
evaluation/classifier_head.py
================================
A lightweight linear classification head on top of the SleepLLMBackbone,
for turning our language-model backbone into an intent classifier
(paper Section 4.1's Class-Incremental Learning setup).

"The paper does not specify this implementation detail" for our
scaled-down version: the paper classifies using Llama-3B/8B's native
generative capabilities (likely prompting the LM to output a class
label as text). At our tiny from-scratch scale, that's not a fair test
(the model has no pretrained language understanding to leverage for
that). We instead add a standard linear probe on MEAN-POOLED final
hidden states -- the standard, simple way to adapt a Transformer
backbone for classification (e.g. as in the original BERT paper's
classification head) -- appropriate for isolating the specific question
Module 9 asks: does Sleep protect old-task knowledge better than plain
continual fine-tuning?
"""

import torch
import torch.nn as nn

from models.backbone import SleepLLMBackbone


class IntentClassifierHead(nn.Module):
    """
    Wraps a SleepLLMBackbone with a linear classification head. The
    backbone itself is exactly what Modules 3-8 built and tested --
    nothing about the Sleep machinery changes here; we're just adding an
    output head suited to classification instead of next-token
    prediction.

    Args:
        backbone: a SleepLLMBackbone instance.
        num_classes: total number of intent classes across ALL tasks the
            classifier will ever see (grows with class-incremental
            learning, but we size the head up front to the FINAL total --
            same "pre-allocate, don't reshape" principle as
            MemoryLevel's max_experts in Module 3/4).
    """

    def __init__(self, backbone: SleepLLMBackbone, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.model.hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            (batch, num_classes) classification logits.

        Implementation note: we need the backbone's HIDDEN STATES (not
        its vocabulary logits) to classify, but SleepLLMBackbone.forward()
        as built in Module 3 only returns final vocab-projected logits.
        Rather than modifying Module 3's tested interface, we
        recompute the embedding+block forward pass here explicitly --
        a small amount of duplication, clearly justified because it
        keeps SleepLLMBackbone's own tested contract (call it, get LM
        logits) completely stable for every other module that depends
        on it (trainer, dreaming, distillation all assume this).
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        x = self.backbone.token_embedding(input_ids) + self.backbone.position_embedding(positions)
        x = self.backbone.embed_dropout(x)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.final_ln(x)  # (batch, seq_len, hidden_dim)

        pooled = x.mean(dim=1)  # mean-pool over sequence -> (batch, hidden_dim)
        return self.classifier(pooled)
