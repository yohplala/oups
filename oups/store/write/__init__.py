#!/usr/bin/env python3
"""
Write operations for oups library.

This package provides utilities and strategies for writing ordered data to Parquet
files.

"""

from .write import KEY_DUPLICATES_ON
from .write import KEY_MAX_N_OFF_TARGET_RGS
from .write import KEY_ROW_GROUP_TARGET_SIZE
from .write import write


__all__ = [
    "write",
    "KEY_DUPLICATES_ON",
    "KEY_MAX_N_OFF_TARGET_RGS",
    "KEY_ROW_GROUP_TARGET_SIZE",
]
