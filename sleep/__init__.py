"""
sleep package
==============
Module 5: the Sleep Scheduler -- pure scheduling logic deciding WHEN
memory consolidation events fire, based on each CMS level's chunk_length
(paper Section 3.2). Contains no model or training code.
"""

from sleep.scheduler import SleepScheduler, ConsolidationEvent

__all__ = ["SleepScheduler", "ConsolidationEvent"]
