#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh

"""
import pytest
from pandas import DataFrame
from pandas import date_range

from oups.defines import KEY_FILE_IDS
from oups.defines import KEY_N_ROWS
from oups.store import OrderedParquetDataset
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset.read_only import (
    ReadOnlyOrderedParquetDataset,
)


df_ref = DataFrame(
    {
        "timestamp": date_range(start="2021/01/01 08:00", periods=8, freq="2h"),
        "temperature": [8.4, 5.3, 4.9, 2.3, 1.0, 0.5, 0.2, 0.1],
    },
)


def test_readonly_from_instance(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd._write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_metadata_file=True,
    )
    assert not isinstance(opd, ReadOnlyOrderedParquetDataset)
    opd_sub1 = opd[1]
    # Check that the read-only instance has the expected attributes (excluding lock management)
    assert opd.__dict__.keys() == opd_sub1.__dict__.keys()
    # Using slice notation to preserve DataFrame format.
    assert opd_sub1.row_group_stats.equals(opd._row_group_stats.iloc[1:2])
    assert opd_sub1.key_value_metadata == opd.key_value_metadata
    assert opd_sub1.ordered_on == opd.ordered_on
    assert len(opd_sub1) == 1
    assert isinstance(opd_sub1, ReadOnlyOrderedParquetDataset)
    opd_sub2 = opd[1:3]
    assert opd.__dict__.keys() == opd_sub2.__dict__.keys()
    assert opd_sub2.row_group_stats.equals(opd.row_group_stats.iloc[1:3])
    assert opd_sub2.key_value_metadata == opd.key_value_metadata
    assert opd_sub2.ordered_on == opd.ordered_on
    assert len(opd_sub2) == 2
    # Check subset of a subset.
    opd_sub3 = opd_sub2[0]
    assert len(opd_sub3) == 1


def test_readonly_max_file_id(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd._write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_metadata_file=True,
    )
    max_file_id_ref = len(df_ref) - 1
    assert opd[2:4].row_group_stats.loc[:, KEY_FILE_IDS].max() == 3
    # Check 'max_file_id' is correctly computed on a subset.
    assert opd[2:4].max_file_id == max_file_id_ref


def test_readonly_to_pandas(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd._write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_metadata_file=True,
    )
    df_res = opd[2:4].to_pandas()
    assert df_ref.iloc[2:4].equals(df_res)


def test_readonly_lock_preservation(tmp_path):
    """
    Test that lock is preserved through parent reference in
    ReadOnlyOrderedParquetDataset.

    This test verifies that:
    1. A ReadOnlyOrderedParquetDataset keeps the parent's lock alive even after
       parent deletion.
    2. The lock is only released when the ReadOnlyOrderedParquetDataset is
       deleted

    """
    df = DataFrame(
        {
            "timestamp": date_range("2021-01-01", periods=5, freq="1h"),
            "value": [1, 2, 3, 4, 5],
        },
    )
    dirpath = tmp_path / "test_dataset"
    # Create original dataset
    opd_original = OrderedParquetDataset(dirpath, ordered_on="timestamp")
    opd_original.write(df=df, row_group_target_size="1h")
    # Create read-only subset
    opd_readonly = opd_original[1:3]
    with pytest.raises(TimeoutError, match="failed to acquire lock"):
        OrderedParquetDataset(dirpath, lock_timeout=1)
    # Delete the original dataset, lock should still be held by readonly
    # instance.
    del opd_original
    # Attempt to create a new dataset should fail due to lock being held by
    # readonly instance.
    with pytest.raises(TimeoutError, match="failed to acquire lock"):
        OrderedParquetDataset(dirpath, lock_timeout=1)
    # Verify readonly dataset still works
    df_readonly = opd_readonly.to_pandas()
    assert len(df_readonly) == 2  # Should have 2 rows (subset [1:3])
    # Delete the readonly dataset, this should release the lock.
    del opd_readonly
    # Now creating a new dataset object should succeed immediately
    opd_new = OrderedParquetDataset(dirpath, lock_timeout=1)
    df_new = opd_new.to_pandas()
    assert df_new.equals(df)
    # Clean up
    del opd_new


def test_exception_readonly_permission_errors(tmp_path):
    """
    Test that ReadOnlyParquetDataset properly blocks modification operations.

    This test verifies that all modification methods raise PermissionError with
    appropriate error messages when called on a read-only dataset.

    """
    # Setup - create a dataset and get a read-only subset
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    opd.write(df=df_ref, row_group_target_size="2h")
    readonly_opd = opd[1:3]

    # Test __setattr__ - should prevent any attribute modification
    with pytest.raises(
        PermissionError,
        match="cannot set attribute 'some_attr' on a read-only dataset",
    ):
        readonly_opd.some_attr = "value"

    with pytest.raises(
        PermissionError,
        match="cannot set attribute 'ordered_on' on a read-only dataset",
    ):
        readonly_opd.ordered_on = "different_column"

    # Test remove_from_disk() - should prevent clearing dataset from disk.
    with pytest.raises(
        PermissionError,
        match="cannot call 'remove_from_disk' on a read-only dataset",
    ):
        readonly_opd.remove_from_disk()

    # Test write() - should prevent writing data.
    with pytest.raises(PermissionError, match="cannot call 'write' on a read-only dataset"):
        readonly_opd.write(df=df_ref)

    # Test _align_file_ids() - should prevent file ID alignment.
    with pytest.raises(
        PermissionError,
        match="cannot call '_align_file_ids' on a read-only dataset",
    ):
        readonly_opd._align_file_ids()

    # Test _remove_row_group_files() - should prevent file removal
    with pytest.raises(
        PermissionError,
        match="cannot call '_remove_row_group_files' on a read-only dataset",
    ):
        readonly_opd._remove_row_group_files(file_ids=[0])

    # Test _sort_row_groups() - should prevent sorting.
    with pytest.raises(
        PermissionError,
        match="cannot call '_sort_row_groups' on a read-only dataset",
    ):
        readonly_opd._sort_row_groups()

    # Test _write_metadata_file() - should prevent metadata writing.
    with pytest.raises(
        PermissionError,
        match="cannot call '_write_metadata_file' on a read-only dataset",
    ):
        readonly_opd._write_metadata_file()

    # Test _write_row_group_files() - should prevent row group writing
    with pytest.raises(
        PermissionError,
        match="cannot call '_write_row_group_files' on a read-only dataset",
    ):
        readonly_opd._write_row_group_files([df_ref])

    # Verify that read operations still work.
    df_result = readonly_opd.to_pandas()
    assert len(df_result) == 2  # Should have 2 rows from the 2-row groups subset (indices 1 and 2)
    assert readonly_opd.ordered_on == "timestamp"
    assert len(readonly_opd) == 2


def test_readonly_defensive_copying(tmp_path):
    """
    Test that ReadOnlyParquetDataset properties are defensively copied.

    This test verifies that the cached properties key_value_metadata and row_group_stats
    return deep copies that are isolated from the underlying data and can be safely
    modified without affecting the original dataset.

    """
    # Setup dataset with custom metadata
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    custom_metadata = {"custom_key": "custom_value", "another_key": "another_value"}
    opd.write(df=df_ref, row_group_target_size="2h", key_value_metadata=custom_metadata)
    readonly_opd = opd[1:3]
    # Test defensive copying of key_value_metadata
    # The cached property should return a deep copy that's isolated from the original
    cached_metadata = readonly_opd.key_value_metadata
    # Modify the cached metadata
    cached_metadata["new_key"] = "new_value"
    cached_metadata["custom_key"] = "modified_value"
    # Verify the underlying _key_value_metadata is unchanged
    assert "new_key" not in readonly_opd._key_value_metadata
    assert readonly_opd._key_value_metadata["custom_key"] == "custom_value"
    # Test defensive copying of row_group_stats
    # The cached property should return a deep copy that's isolated from the original
    cached_stats = readonly_opd.row_group_stats
    # Modify the cached stats
    cached_stats.loc[cached_stats.index[0], KEY_N_ROWS] = 999
    cached_stats["new_column"] = "new_value"
    # Verify the underlying _row_group_stats is unchanged
    assert "new_column" not in readonly_opd._row_group_stats.columns
    assert (
        readonly_opd._row_group_stats.loc[readonly_opd._row_group_stats.index[0], KEY_N_ROWS] != 999
    )


def test_readonly_max_file_id_directory_scan(tmp_path):
    """
    Test that ReadOnlyParquetDataset.max_file_id scans directory for accuracy.

    This test verifies that ReadOnlyParquetDataset computes max_file_id by scanning the
    directory rather than using cached row_group_stats, which ensures accuracy even if
    the subset doesn't contain the highest file_id.

    """
    # Setup dataset with multiple files
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    opd.write(df=df_ref, row_group_target_size="2h")
    # Get the full dataset's max_file_id for reference
    full_max_file_id = opd.max_file_id
    assert full_max_file_id == len(df_ref) - 1
    # Create a read-only subset that doesn't include the highest file_ids
    readonly_opd = opd[1:4]  # This contains file_ids 1, 2, 3
    # The subset's row_group_stats.max() would only be 3
    subset_stats_max = readonly_opd.row_group_stats[KEY_FILE_IDS].max()
    assert subset_stats_max == 3
    # But max_file_id should scan the directory and return the actual max
    assert readonly_opd.max_file_id == full_max_file_id
    assert readonly_opd.max_file_id == 7
    # Test with empty subset
    empty_readonly = opd[0:0]  # Empty slice
    assert len(empty_readonly.row_group_stats) == 0
    assert empty_readonly.max_file_id == full_max_file_id  # Scans directory
    # Test edge case: non-existent directory
    nonexistent_path = tmp_path / "nonexistent"
    opd_nonexistent = OrderedParquetDataset(nonexistent_path, ordered_on="timestamp")
    readonly_nonexistent = opd_nonexistent[:]
    # Should return -1 for non-existent directory
    assert readonly_nonexistent.max_file_id == -1
