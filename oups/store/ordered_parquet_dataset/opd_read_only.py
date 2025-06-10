#!/usr/bin/env python3
"""
Created on Tue Jun 10 2025 20:00:00.

@author: yoh

"""
from copy import deepcopy
from os import scandir
from os.path import exists
from re import compile
from typing import Any, List, Union

from pandas import DataFrame

from oups.defines import PARQUET_FILE_EXTENSION
from oups.defines import PARQUET_FILE_PREFIX


# Find in names of parquet files the integer matching "**file_*.parquet" as 'i'.
FILE_ID_FROM_REGEX = compile(rf".*{PARQUET_FILE_PREFIX}(?P<i>[\d]+){PARQUET_FILE_EXTENSION}$")


def file_ids_in_directory(dirpath: str) -> List[int]:
    """
    Return an unordered list of file ids of parquet files in directory.

    Find the integer matching "**file_*.parquet" in referenced path and returns them as
    a list.

    """
    return [
        int(FILE_ID_FROM_REGEX.match(entry.name)["i"])
        for entry in scandir(dirpath)
        if entry.is_file()
    ]


class ReadOnlyOrderedParquetDatasetProxy:
    """
    A proxy object returned by 'OrderedParquetDataset.__getitem__'.

    This proxy enforces read-only access by wrapping an 'OrderedParquetDataset'
    instance and restricting method calls.

    Parameters
    ----------
    original_opd : OrderedParquetDataset
        The original OrderedParquetDataset instance to wrap.

    Available properties
    --------------------
    - __len__() : Number of row groups
    - dirpath : Directory path
    - key_value_metadata : Key-value metadata
    - is_newly_initialized : Initialization status
    - max_file_id : Maximum file ID (as if complete dataset)
    - ordered_on : Column name for ordering
    - row_group_stats : Row group statistics

    Available methods
    -----------------
    - to_pandas() : Read data as DataFrame

    Restricted operations
    --------------------
    - write() : Cannot write data
    - __getitem__() : Cannot create further subsets
    - __setattr__() : Cannot modify attributes
    - All private methods that modify state

    Examples
    --------
    >>> opd = OrderedParquetDataset('path/to/data')
    >>> readonly_subset = opd[0:5]  # Returns ReadOnlyOrderedParquetDatasetProxy
    >>> df = readonly_subset.to_pandas()  # OK
    >>> readonly_subset.write(new_data)   # Raises PermissionError

    """

    def __init__(self, original_opd):
        """
        Initialize ReadOnlyOrderedParquetDatasetProxy.

        Parameters
        ----------
        original_opd : OrderedParquetDataset
            The original OrderedParquetDataset instance to wrap.

        """
        # Original OPD instance is stored, but its internal state is treated as
        # read-only through this proxy.
        self._original_opd = original_opd

    def __len__(self) -> int:
        """
        Return number of row groups in the dataset.
        """
        return len(self._original_opd)

    @property
    def dirpath(self) -> str:
        """
        Directory path of the dataset.
        """
        return self._original_opd.dirpath

    @property
    def is_newly_initialized(self) -> bool:
        """
        Whether this dataset was newly initialized.
        """
        return self._original_opd.is_newly_initialized

    @property
    def ordered_on(self) -> str:
        """
        Column name for ordering.
        """
        return self._original_opd.ordered_on

    @property
    def key_value_metadata(self) -> dict:
        """
        Key-value metadata (deep copy).
        """
        return deepcopy(self._original_opd.key_value_metadata)

    @property
    def max_file_id(self) -> int:
        """
        Return refreshed max_file_id attribute by scanning file on disk.
        """
        if exists(self.dirpath):
            file_ids = file_ids_in_directory(self.dirpath)
            return max(file_ids) if file_ids else -1
        else:
            return -1

    @property
    def row_group_stats(self) -> DataFrame:
        """
        Allows read access to row_group_stats attribute through the proxy.
        """
        return self._original_opd.row_group_stats.copy(deep=True)

    def to_pandas(self) -> DataFrame:
        """
        Allow reading the dataset through the proxy.
        """
        # Original 'opd' has shared locks.
        return self._original_opd.to_pandas()

    # Explicitly forbid other methods by raising PermissionError
    # These methods are internal to OrderedParquetDataset (prefixed with _)
    # and cannot be callable directly on a read-only proxy.
    def __getitem__(self, item: Union[int, slice]):
        """
        Forbid creating further subsets from a read-only proxy.
        """
        raise PermissionError(
            "Cannot create further subsets from a read-only proxy.",
        )

    def __setattr__(self, name: str, value: Any):
        """
        Forbid setting attributes on a read-only proxy.
        """
        if name == "_original_opd" and not hasattr(self, "_original_opd"):
            super().__setattr__(name, value)
        else:
            raise PermissionError(
                f"Cannot set attribute '{name}' on a read-only proxy.",
            )

    def _align_file_ids(self, *args, **kwargs):
        """
        Forbid calling '_align_file_ids' on a read-only proxy.
        """
        raise PermissionError(
            "Cannot call '_align_file_ids' on a read-only proxy.",
        )

    def _remove_row_group_files(self, *args, **kwargs):
        """
        Forbid calling '_remove_row_group_files' on a read-only proxy.
        """
        raise PermissionError(
            "Cannot call '_remove_row_group_files' on a read-only proxy.",
        )

    def _sort_row_groups(self, *args, **kwargs):
        """
        Forbid calling '_sort_row_groups' on a read-only proxy.
        """
        raise PermissionError(
            "Cannot call '_sort_row_groups' on a read-only proxy.",
        )

    def _write_metadata_file(self, *args, **kwargs):
        """
        Forbid calling '_write_metadata_file' on a read-only proxy.
        """
        raise PermissionError(
            "Cannot call '_write_metadata_file' on a read-only proxy.",
        )

    def _write_row_group_files(self, *args, **kwargs):
        """
        Forbid calling '_write_row_group_files' on a read-only proxy.
        """
        raise PermissionError(
            "Cannot call '_write_row_group_files' on a read-only proxy.",
        )

    def write(self, *args, **kwargs):
        """
        Forbid calling 'write' on a read-only proxy.
        """
        raise PermissionError("Cannot call 'write' on a read-only proxy.")
