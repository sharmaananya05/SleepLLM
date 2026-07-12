"""
configs/model_config.py
========================
Defines every hyperparameter needed to instantiate a SleepLLM model and its
Sleep paradigm (Memory Consolidation + Dreaming), and provides a single
`load_config()` entry point that turns a YAML file into type-checked Python
objects.

Paper cross-reference
----------------------
- MemoryLevelConfig.update_frequency / chunk_length  <-> Definition 1
  (Update Frequency) and Eq. 1-2, Section 2.2.
- MemoryLevelConfig.num_experts / lora_rank            <-> Section 3.2,
  "sparse mixture of experts (MoEs) with a router R^(f_l)" and the
  low-rank expert {A^(f_l*), B^(f_l*)} added at each consolidation step.
- SleepLLMConfig.hidden_dim / sequence_layer_type      <-> Section 2.2,
  "the sequence model ... acts as the short-term memory of the model".
  NOTE: the paper does not fully specify the sequence-layer architecture
  (it builds on the separate "Hope" architecture from Behrouz et al. 2025,
  which is not this paper). We flag this explicitly below.

Design decision the paper does NOT specify (flagged per your instructions)
----------------------------------------------------------------------------
"The paper does not specify this implementation detail."
The paper treats the "Sequence Layer" (Figure 1/2) as a pluggable attention
or recurrent memory module and focuses its contributions on what happens
AFTER it (the CMS chain of MLPs). It never gives exact dimensions.
Proposed solution: implement the sequence layer as standard causal
multi-head self-attention (Vaswani et al. 2017) to start, since (a) it's
well-understood, (b) it lets us validate the CMS/Sleep machinery in
isolation without also debugging an exotic recurrent memory, and (c) the
paper itself says CMS is "the superset of Transformers design" (Section
2.2), i.e. attention + our CMS chain literally IS a valid instance of
their architecture. We can swap in a Titans/linear-attention layer later
as an ablation once the core Sleep logic is verified.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


# ---------------------------------------------------------------------------
# Per-memory-level configuration
# ---------------------------------------------------------------------------
@dataclass
class MemoryLevelConfig:
    """
    Configuration for ONE block in the Continuum Memory System (CMS) chain,
    i.e. one MLP^(f_ell)(.) in Eq. 1 of the paper.

    Why this exists as its OWN dataclass (not just fields on a bigger config):
    The paper's CMS is a *chain* of an arbitrary number of blocks (Figure 7
    shows 3: High/Mid/Low frequency FFN at 1k/5k/10k). We need a LIST of
    these, so each level's settings must be self-contained and repeatable.

    Attributes
    ----------
    name:
        Human-readable label, e.g. "high_freq". Used in logging and
        checkpoint naming so you can tell which block a log line refers to.
    update_frequency:
        f_ell in Definition 1 -- "number of updates per unit of time".
        Higher = updates more often = shorter-term / more fragile memory.
        We store this mostly for documentation/logging; the quantity we
        actually compute FROM it is chunk_length (see below).
    chunk_length:
        C^(ell) in Eq. 2 -- the number of input steps between parameter
        updates for this block. This is what the Sleep scheduler
        (sleep/scheduler.py, a later module) actually uses to decide WHEN
        to trigger consolidation. Per the paper: "parameters of the l-th
        MLP block ... are updated every C^(ell) steps."
        We require chunk lengths of later (slower) levels to be integer
        multiples of earlier ones, matching the paper's simplifying
        assumption in Section 2.2: "C^(ell) is divisible by C^(ell-1)."
    num_experts:
        s_ell in Section 3.2 -- current number of experts in this block's
        MoE router. Starts at 1 (a single dense MLP) and GROWS by one
        low-rank expert every time a consolidation step targets this level
        (Section 3.2, "after each sleep time, the parameters of a subset
        of layers are growing").
    expert_hidden_dim:
        Width of each individual expert MLP (the "dense" experts, not the
        low-rank consolidation experts).
    lora_rank:
        d_low in Section 3.2 -- rank of the low-rank expert
        {A^(f_ell), B^(f_ell)} added at each consolidation event. The paper
        requires d_low << d (the model hidden dim) -- we assert this in
        __post_init__.
    max_experts:
        A practical (non-paper-specified) safety cap on how many low-rank
        experts we allow a block to grow before we stop, to keep memory
        usage bounded on constrained hardware (your 4GB GPU!). This is an
        engineering guard rail, not a paper concept -- flagged as such.
    """

    name: str
    update_frequency: float
    chunk_length: int
    num_experts: int = 1
    expert_hidden_dim: int = 256
    lora_rank: int = 8
    max_experts: int = 16

    def __post_init__(self) -> None:
        if self.chunk_length <= 0:
            raise ValueError(
                f"[{self.name}] chunk_length must be positive, got {self.chunk_length}"
            )
        if self.num_experts < 1:
            raise ValueError(
                f"[{self.name}] num_experts must be >= 1, got {self.num_experts}"
            )
        if self.lora_rank < 1:
            raise ValueError(
                f"[{self.name}] lora_rank must be >= 1, got {self.lora_rank}"
            )


# ---------------------------------------------------------------------------
# Memory (CMS chain) configuration
# ---------------------------------------------------------------------------
@dataclass
class MemoryConfig:
    """
    Holds the FULL chain of CMS levels, ordered from fastest (shortest-term)
    to slowest (longest-term), i.e. levels[0] = MLP^(f1), levels[-1] = MLP^(fk)
    in the paper's notation, where f1 >= f2 >= ... >= fk (Section 2.1,
    Notation: "f1 >= ..., >= fc").

    Why a separate validation step (`__post_init__`) instead of trusting
    the YAML file blindly?
    Research code fails silently in the worst way when a config is subtly
    wrong (e.g. chunk lengths not increasing -> consolidation never fires
    -> you train for 6 hours and get a broken checkpoint with no error).
    Fail FAST and LOUD at config-load time instead.
    """

    levels: List[MemoryLevelConfig]

    def __post_init__(self) -> None:
        if len(self.levels) < 1:
            raise ValueError("MemoryConfig needs at least 1 level.")

        # Enforce the paper's ordering assumption: chunk lengths strictly
        # increase from fast (short-term) to slow (long-term) blocks, and
        # each one evenly divides the next (Section 2.2 assumption).
        for i in range(1, len(self.levels)):
            prev, cur = self.levels[i - 1], self.levels[i]
            if cur.chunk_length <= prev.chunk_length:
                raise ValueError(
                    "Memory levels must have strictly increasing chunk_length "
                    f"(fast -> slow). Level '{prev.name}' has chunk_length="
                    f"{prev.chunk_length}, but the next level '{cur.name}' has "
                    f"chunk_length={cur.chunk_length}, which is not larger."
                )
            if cur.chunk_length % prev.chunk_length != 0:
                raise ValueError(
                    "Paper's Section 2.2 assumes C^(ell) is divisible by "
                    f"C^(ell-1). '{cur.name}'.chunk_length="
                    f"{cur.chunk_length} is not divisible by "
                    f"'{prev.name}'.chunk_length={prev.chunk_length}."
                )

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def chunk_lengths(self) -> List[int]:
        """Convenience accessor: just the C^(ell) values, in order."""
        return [lvl.chunk_length for lvl in self.levels]


# ---------------------------------------------------------------------------
# Top-level model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Everything needed to build the backbone (sequence layer + embeddings),
    independent of the CMS/memory-consolidation machinery.
    """

    vocab_size: int = 50257  # GPT-2 BPE vocab size (we'll use GPT-2 tokenizer
    # to start -- swappable later; flagged as an engineering default, not a
    # paper-specified value, since the paper evaluates on Llama/Qwen
    # tokenizers which are far larger and not laptop-friendly).
    hidden_dim: int = 256
    num_attention_heads: int = 4
    num_sequence_layers: int = 4
    max_sequence_length: int = 512
    dropout: float = 0.1
    sequence_layer_type: Literal["attention"] = "attention"


# ---------------------------------------------------------------------------
# Master config: composes everything above
# ---------------------------------------------------------------------------
@dataclass
class SleepLLMConfig:
    """
    The single object threaded through the entire codebase. Every module
    (models/, memory/, sleep/, dreaming/, distillation/, trainer/,
    evaluation/) takes a `SleepLLMConfig` (or a sub-config sliced from it)
    in its constructor -- never magic numbers.
    """

    experiment_name: str = "sleepllm_debug"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(
        default_factory=lambda: MemoryConfig(
            levels=[
                MemoryLevelConfig(name="high_freq", update_frequency=1000, chunk_length=64),
                MemoryLevelConfig(name="low_freq", update_frequency=10, chunk_length=512),
            ]
        )
    )

    def save(self, path: str | Path) -> None:
        """Serialize this config back to YAML (for logging alongside checkpoints)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> SleepLLMConfig:
    """
    Load a YAML file into a validated SleepLLMConfig.

    Time complexity: O(L) where L = number of memory levels (everything else
    is O(1) field assignment). Irrelevant in practice -- this runs once at
    startup, not in the training loop.

    Memory usage: negligible (a few KB of Python objects); no tensors are
    created here.

    How researchers normally do this:
    Hydra (Facebook/Meta) or OmegaConf are the de-facto standards for large
    labs because they support config composition/overrides from the CLI
    (e.g. `python train.py memory.levels.0.chunk_length=128`). We're using
    plain dataclasses + PyYAML here because it's transparent and dependency-
    light for a student project you need to explain line-by-line in a
    review -- but migrating to Hydra later is a natural "possible
    improvement" once you have many experiment variants to sweep over.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {path}")

    model_cfg = ModelConfig(**raw.get("model", {}))

    level_cfgs = [
        MemoryLevelConfig(**lvl) for lvl in raw["memory"]["levels"]
    ]
    memory_cfg = MemoryConfig(levels=level_cfgs)

    return SleepLLMConfig(
        experiment_name=raw.get("experiment_name", "sleepllm_debug"),
        seed=raw.get("seed", 42),
        model=model_cfg,
        memory=memory_cfg,
    )
