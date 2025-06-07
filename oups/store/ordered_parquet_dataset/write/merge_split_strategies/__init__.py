#!/usr/bin/env python3
"""
Merge and split strategies for ordered atomic regions (OARs).

This module provides implementations of strategies for merging and splitting
ordered atomic regions when writing DataFrame data alongside existing Parquet files.

Available Strategies:
- NRowsMergeSplitStrategy: Uses a target number of rows to determine optimal splits
- TimePeriodMergeSplitStrategy: Uses time periods to determine optimal splits

"""

# Concrete strategy implementations
from .n_rows_strategy import NRowsMergeSplitStrategy
from .time_period_strategy import TimePeriodMergeSplitStrategy


__all__ = [
    "NRowsMergeSplitStrategy",
    "TimePeriodMergeSplitStrategy",
]
