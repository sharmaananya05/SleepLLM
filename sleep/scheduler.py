"""
sleep/scheduler.py
====================
Decides WHEN each memory level in the CMS chain should trigger a
consolidation event, based purely on the training step count and each
level's chunk_length -- matching paper Section 3.2 exactly:

    "given the list of chunk lengths {C^(1), ..., C^(k)}, the sleep
    process and so memory consolidation happens only at
    {C^(1)xb, ..., C^(k)xb} steps for all b in N."

This module contains NO model code and NO knowledge of what a
"consolidation event" actually DOES (that's Module 6, distillation/) --
its only job is the scheduling arithmetic: given a step number, which
levels (if any) should fire right now. This separation matters for
testability: we can verify the SCHEDULE is correct (right steps trigger
right levels) completely independently of whether the actual distillation
math (Module 6) is implemented yet or correct.
"""

from dataclasses import dataclass

from config.model_config import MemoryConfig


@dataclass
class ConsolidationEvent:
    """
    Describes ONE level's consolidation trigger at a given step.

    Attributes:
        level_index: index into MemoryConfig.levels / ContinuumMemorySystem.levels
            of the level that should consolidate right now.
        level_name: human-readable name, for logging.
        step: the training step this event fires at.
    """
    level_index: int
    level_name: str
    step: int


class SleepScheduler:
    """
    Given a MemoryConfig (the ordered list of chunk lengths), tells the
    training loop (Module 8) which memory levels need to consolidate at
    any given step.

    Why a class (not just a function)?
    We cache each level's chunk_length once at construction instead of
    re-reading the config every step -- trivial here, but it also gives
    us a natural place to add STATE later (e.g. tracking how many times
    each level has consolidated so far, useful for logging/debugging sleep
    schedules over a long training run) without changing the calling code
    in the trainer.
    """

    def __init__(self, memory_config: MemoryConfig) -> None:
        self.memory_config = memory_config

    def get_events_at_step(self, step: int) -> list[ConsolidationEvent]:
        """
        Returns every level that should consolidate at this EXACT step.

        Args:
            step: current training step, 1-indexed (step 1 = the very
                first optimizer update). We use 1-indexing (not 0) so
                that "step % chunk_length == 0" has an intuitive meaning:
                step 16 is the 16th update, matching how a human would
                count "every 16 steps."

        Returns:
            A list of ConsolidationEvent, one per level whose
            chunk_length evenly divides `step`. Can be EMPTY (most steps
            trigger nothing), can contain MULTIPLE events (e.g. at
            step=64 in our debug config, BOTH high_freq (chunk_length=16)
            AND low_freq (chunk_length=64) trigger simultaneously, since
            64 is divisible by both 16 and 64).

        Time complexity: O(L) where L = number of memory levels (typically
        2-5) -- checks each level's chunk_length once. Negligible cost,
        called once per training step.

        Order matters: we return events in level order (fast -> slow),
        matching paper Section 3.2's requirement that faster memories
        consolidate INTO slower ones -- so if the training loop processes
        events in the returned order, it naturally consolidates
        high_freq -> low_freq before any later, even-slower level, never
        the reverse.
        """
        if step <= 0:
            raise ValueError(f"step must be positive (1-indexed), got {step}")

        events = []
        for idx, level in enumerate(self.memory_config.levels):
            if step % level.chunk_length == 0:
                events.append(
                    ConsolidationEvent(level_index=idx, level_name=level.name, step=step)
                )
        return events

    def next_event_step(self, after_step: int) -> int:
        """
        Convenience method: what's the NEXT step (strictly after
        `after_step`) at which ANY level triggers? Useful for a training
        loop that wants to skip ahead / log "next sleep phase in N steps"
        without checking every single intermediate step one by one.

        Time complexity: O(min(chunk_lengths)) in the worst case (has to
        walk forward step by step) -- fine for our chunk lengths (tens to
        low thousands), would need a smarter approach (e.g. LCM-based) if
        chunk lengths were huge, which is a reasonable "possible
        improvement" flagged for scaling to the paper's actual settings
        (e.g. Figure 7's 1k/5k/10k schedule).
        """
        smallest_chunk = min(lvl.chunk_length for lvl in self.memory_config.levels)
        candidate = after_step + 1
        # Search at most `smallest_chunk` steps ahead -- guaranteed to
        # find a hit within that window since the fastest level triggers
        # at least that often.
        for _ in range(smallest_chunk):
            for level in self.memory_config.levels:
                if candidate % level.chunk_length == 0:
                    return candidate
            candidate += 1
        raise RuntimeError("Could not find next event step -- this should be unreachable.")
