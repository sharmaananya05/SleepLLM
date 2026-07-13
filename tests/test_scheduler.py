"""
tests/test_scheduler.py
=========================
Verifies Module 5: the sleep schedule fires at exactly the steps the
paper specifies (Section 3.2), including the "faster memory consolidates
multiple times before slower memory's own update" scenario the paper
describes explicitly:

    "consider a memory with update frequency of 1K followed by a memory
    with update frequency of 10K: in this case, the faster memory is
    updated 10 times before the update of slower memory block."

We test this with our debug config's simpler 16/64 ratio (4x, not 10x --
same principle, smaller numbers so the test runs instantly and is easy to
verify by hand).
"""

from config.model_config import MemoryConfig, MemoryLevelConfig
from sleep.scheduler import SleepScheduler


def _build_scheduler():
    memory_config = MemoryConfig(
        levels=[
            MemoryLevelConfig(name="high_freq", update_frequency=1000, chunk_length=16),
            MemoryLevelConfig(name="low_freq", update_frequency=10, chunk_length=64),
        ]
    )
    return SleepScheduler(memory_config)


def test_no_events_on_steps_not_divisible_by_any_chunk_length():
    scheduler = _build_scheduler()
    for step in [1, 5, 15, 17, 33, 50, 63]:
        assert scheduler.get_events_at_step(step) == []


def test_high_freq_fires_every_16_steps():
    scheduler = _build_scheduler()
    for step in [16, 32, 48, 80, 96]:  # NOT 64 -- that's tested separately below
        events = scheduler.get_events_at_step(step)
        assert len(events) == 1
        assert events[0].level_name == "high_freq"


def test_both_levels_fire_together_at_step_64():
    """
    This is the paper's exact scenario: step 64 is divisible by BOTH 16
    (high_freq) and 64 (low_freq) -- both levels should trigger
    simultaneously, with high_freq listed FIRST (fast -> slow order),
    since it must consolidate INTO low_freq before low_freq's own update.
    """
    scheduler = _build_scheduler()
    events = scheduler.get_events_at_step(64)
    assert len(events) == 2
    assert events[0].level_name == "high_freq"
    assert events[1].level_name == "low_freq"


def test_high_freq_consolidates_4_times_before_each_low_freq_update():
    """
    Paper's "10 consolidation steps before slower memory's own update"
    scenario, scaled to our config's 64/16=4 ratio. Over steps 1-64,
    high_freq should fire at 16, 32, 48, 64 (4 times) while low_freq
    fires only once, at 64.
    """
    scheduler = _build_scheduler()
    high_freq_fires = 0
    low_freq_fires = 0
    for step in range(1, 65):
        for event in scheduler.get_events_at_step(step):
            if event.level_name == "high_freq":
                high_freq_fires += 1
            elif event.level_name == "low_freq":
                low_freq_fires += 1

    assert high_freq_fires == 4
    assert low_freq_fires == 1


def test_next_event_step_finds_the_nearest_upcoming_trigger():
    scheduler = _build_scheduler()
    assert scheduler.next_event_step(after_step=1) == 16
    assert scheduler.next_event_step(after_step=16) == 32
    assert scheduler.next_event_step(after_step=60) == 64


def test_zero_or_negative_step_raises():
    scheduler = _build_scheduler()
    try:
        scheduler.get_events_at_step(0)
        assert False, "should have raised ValueError for step=0"
    except ValueError:
        pass
