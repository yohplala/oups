#!/usr/bin/env python3
"""
Tests for OrderedParquetDataset locking functionality.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

from oups.store.ordered_parquet_dataset.lock import exclusive_lock


class MockDataset:
    """
    Mock dataset class for testing the decorator.
    """

    def __init__(self, dirpath: Union[str, Path]):
        self.dirpath = Path(dirpath).resolve()

    @exclusive_lock(timeout=2, lifetime=10)
    def slow_operation(self, sleep_time=1):
        """
        Dummy operation that takes some time.
        """
        time.sleep(sleep_time)
        return "completed"

    @exclusive_lock(timeout=1, lifetime=10)
    def quick_timeout_operation(self):
        """
        Operation with short timeout for testing timeouts.
        """
        return "completed"


def test_exclusive_lock_blocks_concurrent_access(tmp_path):
    """
    Test that exclusive lock prevents concurrent access.
    """
    dataset_path = tmp_path / "test_dataset"
    dataset_path.mkdir()

    def worker_task(sleep_time):
        """
        Worker that tries to access the same dataset.
        """
        mock = MockDataset(str(dataset_path))
        start_time = time.time()
        result = mock.slow_operation(sleep_time=sleep_time)
        end_time = time.time()
        return result, start_time, end_time

    # Run two tasks concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(worker_task, 0.5)  # Task 1: sleeps 0.5s
        future2 = executor.submit(worker_task, 0.1)  # Task 2: sleeps 0.1s

        result1, start1, end1 = future1.result()
        result2, start2, end2 = future2.result()

    # Both should complete successfully
    assert result1 == "completed"
    assert result2 == "completed"

    # Tasks should not overlap (exclusive access)
    # One should finish before the other starts
    total_time = max(end1, end2) - min(start1, start2)
    expected_min_time = 0.5 + 0.1  # Both sleep times combined
    # Tasks should run sequentially, not concurrently.
    assert total_time >= expected_min_time
