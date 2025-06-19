#!/usr/bin/env python3
"""
Created on Mon Jun 16 20:00:00 2025.

@author: yoh

OrderedParquetDataset caching utilities for Store operations.

"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, List


if TYPE_CHECKING:
    from oups.store.store import Store


@contextmanager
def cached_datasets(
    store: "Store",
    keys: List[dataclass],
):
    """
    Context manager for caching OrderedParquetDataset objects.

    Parameters
    ----------
    store : Store
        Store instance to get datasets from
    keys : List[dataclass]
        List of dataset keys to cache

    Yields
    ------
    Dict[dataclass, OrderedParquetDataset]
        Dictionary mapping keys to cached dataset objects

    """
    cache = {}
    try:
        cache = {key: store[key] for key in keys}
        yield cache
    finally:
        # Explicitly trigger OrderedParquetDataset deletion to release the lock.
        for key in keys:
            del cache[key]
