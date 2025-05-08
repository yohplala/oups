#!/usr/bin/env python3
"""
Write operations for oups library.

This package provides utilities and strategies for writing ordered data to Parquet
files.

"""

from .write import KEY_ROW_GROUP_TARGET_SIZE
from .write import write_ordered


__all__ = [
    "write_ordered",
    "KEY_ROW_GROUP_TARGET_SIZE",
]
