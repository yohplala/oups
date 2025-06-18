#!/usr/bin/env python3
"""
Created on Tue Jun 10 2025 20:00:00.

@author: yoh

"""
from copy import deepcopy
from functools import cached_property
from os import scandir
from os.path import exists
from re import compile
from typing import Any, List

from pandas import DataFrame

from oups.defines import PARQUET_FILE_EXTENSION
from oups.defines import PARQUET_FILE_PREFIX
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset.base import OrderedParquetDataset


# Find in names of parquet files the integer matching "**file_*.parquet" as 'i'.
FILE_ID_FROM_REGEX = compile(rf".*{PARQUET_FILE_PREFIX}(?P<i>[\d]+){PARQUET_FILE_EXTENSION}$")
CACHED_PROPERTIES = {"key_value_metadata", "row_group_stats"}


def file_ids_in_directory(dirpath: str) -> List[int]:
    """
    Return an unordered list of file ids of parquet files in directory.

    Find the integer matching "**file_*.parquet" in referenced path and returns them as
    a list.

    Parameters
    ----------
    dirpath : str
        Directory path to scan for parquet files.

    Returns
    -------
    List[int]
        List of file IDs found in the directory.

    """
    return [
        int(FILE_ID_FROM_REGEX.match(entry.name)["i"])
        for entry in scandir(dirpath)
        if entry.is_file()
    ]


class ReadOnlyOrderedParquetDataset(OrderedParquetDataset):
    """
    Read-only version of OrderedParquetDataset.

    This class inherits all functionality from BaseOrderedParquetDataset but
    blocks modification methods and provides defensive copying for mutable
    properties to enforce read-only access.

    Available Properties
    -------------------
    dirpath : str
        Directory path from where to load data.
    is_newly_initialized : bool
        True if this dataset instance was just created and has no existing
        metadata file. False if the dataset was loaded from existing files.
    key_value_metadata : Dict[str, str]
        Key-value metadata (cached deep copy to prevent modification).
    max_file_id : int
        Maximum file id in current directory (scans directory for accuracy).
    ordered_on : str
        Column name to order row groups by.
    row_group_stats : DataFrame
        Row group statistics (cached deep copy to prevent modification).

    Available Methods
    ----------------
    __getitem__()
        Select among the row-groups using integer/slicing (inherited from base).
    __len__()
        Return number of row groups in the dataset.
    to_pandas()
        Return data as a pandas dataframe.

    Restricted Operations
    --------------------
    __setattr__()
        Cannot modify any attributes.
    write()
        Cannot write data to disk.
    _align_file_ids()
        Cannot align file ids.
    _remove_row_group_files()
        Cannot remove row group files.
    _sort_row_groups()
        Cannot sort row groups.
    _write_metadata_file()
        Cannot write metadata to disk.
    _write_row_group_files()
        Cannot write row group files.

    Examples
    --------
    >>> from oups.store.ordered_parquet_dataset import OrderedParquetDataset
    >>> opd = OrderedParquetDataset('path/to/data')
    >>> readonly_opd = opd[0:5]  # Returns ReadOnlyOrderedParquetDataset
    >>> further_subset = readonly_opd[1:3]  # OK - further slicing allowed
    >>> df = readonly_opd.to_pandas()  # OK - read operations allowed
    >>> readonly_opd.write(data=new_data)  # Raises PermissionError

    """

    @classmethod
    def _from_instance(cls, opd):
        """
        Create ReadOnlyOrderedParquetDataset from BaseOrderedParquetDataset instance.

        This internal constructor bypasses normal __init__ to create a read-only
        version of any existing BaseOrderedParquetDataset instance.

        Parameters
        ----------
        opd : BaseOrderedParquetDataset
            Any BaseOrderedParquetDataset instance to make read-only.

        Returns
        -------
        ReadOnlyOrderedParquetDataset
            A read-only version of the input instance.

        """
        instance = cls.__new__(cls)
        # Copy __dict__ but exclude cached property values to ensure
        # new instance computes properties based on its own data.
        # Also do not copy '_lock' to maintain lock management at parent level.
        instance_dict = {
            key: value for key, value in opd.__dict__.items() if key not in CACHED_PROPERTIES
        }
        # Use object.__setattr__ to bypass the custom __setattr__ method.
        object.__setattr__(instance, "__dict__", instance_dict)
        # Increment reference count since new instance shares the lock.
        opd._lock._ref_count += 1
        return instance

    @cached_property
    def key_value_metadata(self) -> dict:
        """
        Key-value metadata (cached deep copy to prevent modification).
        """
        return deepcopy(self._key_value_metadata)

    @cached_property
    def row_group_stats(self) -> DataFrame:
        """
        Row group statistics (cached deep copy to prevent modification).
        """
        return self._row_group_stats.copy(deep=True)

    @property
    def max_file_id(self) -> int:
        """
        Return maximum file id by scanning directory for accuracy.
        """
        if exists(self.dirpath):
            file_ids = file_ids_in_directory(self.dirpath)
            return max(file_ids) if file_ids else -1
        else:
            return -1

    # Modification operations are blocked.
    def __setattr__(self, name: str, value: Any):
        """
        Block all attribute modification on read-only dataset.
        """
        raise PermissionError(f"cannot set attribute '{name}' on a read-only dataset.")

    def write(self, **kwargs):
        """
        Block write operations.
        """
        raise PermissionError("cannot call 'write' on a read-only dataset.")

    def _align_file_ids(self):
        """
        Block file ID alignment.
        """
        raise PermissionError("cannot call '_align_file_ids' on a read-only dataset.")

    def _remove_row_group_files(self, file_ids, sort_row_groups=True, key_value_metadata=None):
        """
        Block row group file removal.

        Parameters
        ----------
        file_ids : List[int]
            File ids to remove.
        sort_row_groups : Optional[bool], default True
            If `True`, sort row groups after removing files.
        key_value_metadata : Optional[Dict[str, str]], default None
            User-defined key-value metadata to write in metadata file.

        """
        raise PermissionError("cannot call '_remove_row_group_files' on a read-only dataset.")

    def _sort_row_groups(self):
        """
        Block row group sorting.
        """
        raise PermissionError("cannot call '_sort_row_groups' on a read-only dataset.")

    def _write_metadata_file(self, key_value_metadata=None):
        """
        Block metadata file writing.

        Parameters
        ----------
        key_value_metadata : Optional[Dict[str, str]], default None
            User-defined key-value metadata to write in metadata file.

        """
        raise PermissionError("cannot call '_write_metadata_file' on a read-only dataset.")

    def _write_row_group_files(
        self,
        dfs,
        write_metadata_file=True,
        key_value_metadata=None,
        **kwargs,
    ):
        """
        Block row group file writing.

        Parameters
        ----------
        dfs : Iterable[DataFrame]
            DataFrames to write.
        write_metadata_file : bool, default True
            If `True`, write metadata file.
        key_value_metadata : Optional[Dict[str, str]], default None
            User-defined key-value metadata to write in metadata file.

        """
        raise PermissionError("cannot call '_write_row_group_files' on a read-only dataset.")
