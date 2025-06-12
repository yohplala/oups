#!/usr/bin/env python3
"""
Created on Sun May 18 16:00:00 2025.

@author: yoh

File-based Lock Decorators.

This module provides NFS-safe locking for OrderedParquetDataset operations using
flufl.lock. It implements exclusive lock, with lock files placed alongside the
dataset directory and _opdmd file.

Files Structure:

parent_directory/
├── my_dataset1/                    # Dataset directory
│   ├── file_0000.parquet
│   └── file_0001.parquet
├── my_dataset1_opdmd               # Metadata file
└── my_dataset1.lock                # Exclusive lock file

"""
from functools import wraps
from typing import Optional

from flufl.lock import Lock


LOCK_EXTENSION = ".lock"


def exclusive_lock(timeout: Optional[int] = 20, lifetime: Optional[int] = 40):
    """
    Ensure exclusive access to OrdedredParquetDataset.

    Only one decorated method can run at a time, and it will
    block all other exclusive_lock methods.

    Parameters
    ----------
    timeout : Optional[int], default 40
        Maximum time to wait for lock acquisition in seconds.
    lifetime : Optional[int], default 60
        Maximum lock lifetime.

    Example
    -------
    @exclusive_lock(timeout=40)
    def exclusive(self, **kwargs):
        # exclusive operations here.
        pass

    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create lock file path.
            lock_file = self.dirpath.parent / f"{self.dirpath.name}{LOCK_EXTENSION}"
            # Ensure parent directory exists.
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            # Use flufl.lock context manager directly.
            with Lock(str(lock_file), lifetime=lifetime, default_timeout=timeout):
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
