"""
trainer/trainer.py
====================
The training loop that ties every previous module together: "wake" phase
(normal language modeling on real data) interleaved with "sleep" phase
(memory consolidation via expand() + Knowledge Seeding), scheduled by
SleepScheduler -- this is the paper's core claim made concrete:
"a continual learner need to have different stages of activeness ...
Active or Wake Time, and Sleep Time" (Section 1/Figure 1).

KEY DESIGN DECISION -- how we get "teacher" and "student" WITHOUT a full
model deepcopy (unlike dreaming/self_improvement.py's isolated SFT, which
genuinely needs a separate copy):
Recall from memory/lora_expert.py: a newly expand()-ed expert's B matrix
is initialized to EXACTLY ZERO, so it contributes f(x)=0 to the model's
output the instant it's created. This means, at the very moment expand()
is called, calling model(x) produces IDENTICAL output to what the model
would have produced before expansion (LM_theta from the paper). We
EXPLOIT this: we capture that first forward pass (before any Knowledge
Seeding training step touches the new expert) as our frozen "teacher"
reference, and every subsequent student forward pass is compared against
it -- without ever needing a second copy of the whole model in memory.
This is a genuine memory-saving design choice enabled by the paper's own
zero-init strategy, worth highlighting in your review.

ANOTHER ASSUMPTION (flagged): our SleepScheduler (Module 5) is built on
ONE shared MemoryConfig, so a consolidation event at a given step applies
to the SAME memory level index across EVERY block in the backbone
simultaneously (e.g. all 2 blocks' "high_freq" level consolidate
together). The paper does not explicitly discuss whether different
Transformer layers should have independent sleep schedules or a shared
one -- we assume SHARED, which is simpler to reason about and matches
how Figure 7's single global schedule is presented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from config.model_config import SleepLLMConfig
from models.backbone import SleepLLMBackbone
from sleep.scheduler import SleepScheduler
from distillation.losses import reverse_kl_divergence
from distillation.knowledge_seeding import freeze_all_except_expert
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SleepEventResult:
    """Records what happened during one consolidation event, for logging/testing."""
    step: int
    block_index: int
    level_name: str
    new_expert_index: int
    final_kl_loss: float


@dataclass
class TrainStepResult:
    step: int
    wake_loss: float
    sleep_events: list[SleepEventResult] = field(default_factory=list)


class SleepLLMTrainer:
    """
    Orchestrates wake-phase training + scheduled sleep-phase consolidation.

    Args:
        model: a SleepLLMBackbone.
        config: the SleepLLMConfig used to build it (needed for
            memory-level chunk lengths, via SleepScheduler).
        wake_lr: learning rate for the FULL-model optimizer used during
            wake steps.
        ks_lr: learning rate for the Knowledge Seeding optimizer, applied
            ONLY to a newly-activated expert's parameters during a sleep
            event.
        ks_steps: number of gradient steps to run per consolidation
            event (paper doesn't specify an exact count for this
            sub-loop -- kept small, matching the same "keep sleep cheap"
            spirit as dreaming's sft_steps).
    """

    def __init__(
        self,
        model: SleepLLMBackbone,
        config: SleepLLMConfig,
        wake_lr: float = 3e-4,
        ks_lr: float = 1e-3,
        ks_steps: int = 5,
    ) -> None:
        self.model = model
        self.config = config
        self.scheduler = SleepScheduler(config.memory)
        self.ks_lr = ks_lr
        self.ks_steps = ks_steps
        self.step_count = 0

        self.wake_optimizer = torch.optim.Adam(model.parameters(), lr=wake_lr)

    def wake_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
        """
        One standard language-modeling training step: forward, cross-
        entropy loss, backward, optimizer step -- with ALL parameters
        trainable (any freezing from a PRIOR sleep event is explicitly
        undone here, first thing, so wake phases always train the full
        currently-active model).

        Time complexity: O(seq_len^2 * hidden_dim) for attention +
        O(seq_len * hidden_dim * expert_hidden_dim) per active expert,
        times num_sequence_layers -- same cost profile as the plain
        forward pass tested in Module 3.
        """
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        self.wake_optimizer.zero_grad()

        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1)
        )
        loss.backward()
        self.wake_optimizer.step()

        return loss.item()

    def _consolidate_one_level(
        self, block_index: int, level_index: int, reference_input: torch.Tensor
    ) -> SleepEventResult:
        """
        Runs ONE level's consolidation: expand() + Knowledge Seeding
        distillation, using the zero-init teacher trick described in the
        module docstring.
        """
        block = self.model.blocks[block_index]
        level = block.cms.levels[level_index]

        new_idx = level.expand()

        # Capture the teacher reference NOW, before any training touches
        # the new expert -- at this instant its LowRankExpert.B is still
        # exactly zero (memory/lora_expert.py), so this forward pass
        # equals the pre-expansion model's output exactly.
        self.model.eval()
        with torch.no_grad():
            teacher_logits = self.model(reference_input).detach()

        freeze_all_except_expert(self.model, level, new_idx)
        ks_optimizer = torch.optim.Adam(level.experts[new_idx].parameters(), lr=self.ks_lr)

        self.model.train()
        final_kl = 0.0
        for _ in range(self.ks_steps):
            ks_optimizer.zero_grad()
            student_logits = self.model(reference_input)
            kl = reverse_kl_divergence(student_logits, teacher_logits)
            kl.backward()
            ks_optimizer.step()
            final_kl = kl.item()

        logger.info(
            "Consolidated block=%d level=%s new_expert=%d final_kl=%.4f",
            block_index, level.config.name, new_idx, final_kl,
        )

        return SleepEventResult(
            step=self.step_count,
            block_index=block_index,
            level_name=level.config.name,
            new_expert_index=new_idx,
            final_kl_loss=final_kl,
        )

    def sleep_step(self, reference_input: torch.Tensor) -> list[SleepEventResult]:
        """
        Checks the schedule for the CURRENT step, and runs consolidation
        for every (block, level) pair the schedule says should fire --
        per the "shared schedule across blocks" assumption documented
        above.

        Args:
            reference_input: a batch of token ids used both to capture
                the teacher reference and to train the student during
                Knowledge Seeding. In a full pipeline this would
                typically be a recent wake-phase batch (representing
                "what was just learned in-context") -- the paper's own
                framing of consolidating RECENT short-term memory.
        """
        events = self.scheduler.get_events_at_step(self.step_count)
        results = []
        for event in events:
            for block_index in range(len(self.model.blocks)):
                result = self._consolidate_one_level(block_index, event.level_index, reference_input)
                results.append(result)

        # Restore full trainability for the NEXT wake step -- sleep
        # events must never leave the model partially frozen afterward.
        for param in self.model.parameters():
            param.requires_grad = True

        return results

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> TrainStepResult:
        """
        One full step of the paper's wake/sleep lifecycle: increment the
        step counter, run a wake step on the given batch, then check/run
        any scheduled sleep events using that SAME batch as the
        consolidation reference.
        """
        self.step_count += 1
        wake_loss = self.wake_step(input_ids, target_ids)
        sleep_events = self.sleep_step(reference_input=input_ids)
        return TrainStepResult(step=self.step_count, wake_loss=wake_loss, sleep_events=sleep_events)
