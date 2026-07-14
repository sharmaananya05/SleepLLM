# SleepLLM: Memory Consolidation and Unlearning in Intelligent Systems

A from-scratch, research-grade implementation of a **Sleep paradigm** for
Large Language Models, inspired by *"Language Models Need Sleep:
Learning to Self-Modify and Consolidate Memories"* (Behrouz, Hashemi,
Mirrokni; Google Research / Cornell, 2026), extended with a
**machine unlearning** component.

## What this project does

Standard LLMs are static after training: they cannot acquire new
knowledge without catastrophic forgetting of old knowledge. This project
implements a biologically-inspired alternative where the model
alternates between two phases:

- **Wake phase** — the model processes new data and learns normally.
- **Sleep phase** — periodically, the model stops taking new input and:
  1. **Grows new parameters** ("experts") in a modular memory system
  2. **Consolidates** recent short-term knowledge into that new capacity
     via a distillation objective ("Knowledge Seeding")
  3. **Dreams** — generates and rehearses its own synthetic training data

On top of this, the project adds **targeted unlearning**: the ability to
deliberately and selectively erase specific learned information, via
(a) deactivating the exact memory expert that stores it, or (b) a
gradient-ascent fine-tuning method for more diffuse knowledge.

## Project structure

```
SleepLLM/
├── config/         Experiment configuration (YAML + dataclasses)
├── models/         Backbone architecture: attention + Continuum Memory System
├── memory/         Expandable/maskable memory experts + unlearning
├── sleep/          Scheduler: decides WHEN to consolidate
├── distillation/   Knowledge Seeding: on-policy distillation + RL reward
├── dreaming/        Synthetic data generation + self-improvement
├── data/           Language modeling data pipeline
├── trainer/        Wake/sleep training loop
├── evaluation/     Class-incremental learning experiment (Sleep vs. baseline)
├── utils/          Reproducible seeding + logging
├── tests/          59+ unit tests covering every module
├── scripts/        Environment check utility
└── main.py         Command-line entry point
```

## Setup

```powershell
conda create -n sleepllm python=3.11 -y
conda activate sleepllm
pip install -r requirements.txt
python scripts/check_environment.py   # verify everything installed correctly
```

## Running the tests

```powershell
pytest tests/ -v
```
All 59 tests should pass. Tests cover: config validation, model
forward/backward correctness, memory expansion and masking, sleep
scheduling, distillation loss correctness, dream generation and
selection, the full wake/sleep training loop, the class-incremental
learning evaluation, and both unlearning methods.

## Running the demos

**1. Watch the Sleep paradigm actually train, live:**
```powershell
python main.py train --steps 100 --log-every 10
```
Watch for `Consolidated block=... level=... new_expert=...` log lines —
that's a real memory consolidation event happening.

**2. See Sleep vs. no-Sleep on catastrophic forgetting:**
```powershell
python main.py demo-continual-learning
```
Trains a classifier on Task 1, then Task 2, and reports how much Task 1
accuracy each condition retained.

**3. See targeted unlearning in action:**
```powershell
python main.py demo-unlearning
```
Trains a classifier, then deliberately unlearns specific classes,
showing forget-set loss increase sharply while retain-set loss stays low.

## Design decisions and known limitations

Per the project's guiding principle of never silently inventing
implementation details, every place where the paper is ambiguous is
explicitly flagged in the corresponding source file's docstring with the
phrase *"the paper does not specify this implementation detail"*,
followed by the reasoning behind the choice made. Notable examples:

- The Sequence Layer uses standard causal self-attention (the paper
  treats this as pluggable).
- The semantic reward (`r_sem`) uses cosine similarity of the frozen
  teacher's own hidden states, rather than a separately-trained judge
  model.
- Dream selection uses loss value as a cheaper proxy for the paper's
  gradient-norm importance score.
- Dreaming's ReSTEM step is a single simplified iteration rather than
  the full iterative algorithm.

**Known limitation**: the current class-incremental learning demo shows
substantial forgetting in both the Sleep and no-Sleep conditions at this
laptop scale, because wake-phase training currently updates all backbone
parameters uniformly every step. The paper's full mechanism (Eq. 2) also
varies *update frequency* by memory level during ordinary training, not
just at consolidation events — extending wake-phase training to respect
per-level update frequency is the natural next step to sharpen this
comparison.

## Hardware

Developed and tested entirely on a laptop (Dell G15 5520, RTX 3050
4GB VRAM, 8GB RAM) — no cloud GPU required. The debug model
configuration (`config/debug_laptop.yaml`) uses a ~6.7M parameter
backbone, deliberately sized to leave large headroom on 4GB of VRAM.
