#!/usr/bin/env python3
"""
Created on Tue Jun 10 22:30:00 2025.

@author: yoh

"""
from functools import cached_property
from itertools import chain
from os import remove
from os import rename
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

from numpy import iinfo
from numpy import isin
from numpy import uint16
from numpy import uint32
from pandas import DataFrame
from pandas import Series
from pandas import concat

from oups.defines import DIR_SEP
from oups.defines import KEY_FILE_IDS
from oups.defines import KEY_N_ROWS
from oups.defines import KEY_ORDERED_ON
from oups.defines import KEY_ORDERED_ON_MAXS
from oups.defines import KEY_ORDERED_ON_MINS
from oups.defines import PARQUET_FILE_EXTENSION
from oups.defines import PARQUET_FILE_PREFIX
from oups.store.filepath_utils import strip_path_tail
from oups.store.ordered_parquet_dataset.metadata_filename import get_md_filepath
from oups.store.ordered_parquet_dataset.parquet_adapter import ParquetAdapter
from oups.store.ordered_parquet_dataset.write import write


if TYPE_CHECKING:
    from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_read_only import (
        ReadOnlyOrderedParquetDataset,
    )


# Do not change this order, it is expected by OrderedParquetDataset.write_row_group_files()
RGS_STATS_COLUMNS = [KEY_FILE_IDS, KEY_N_ROWS, KEY_ORDERED_ON_MINS, KEY_ORDERED_ON_MAXS]
RGS_STATS_BASE_DTYPES = {
    KEY_N_ROWS: uint32,
    KEY_FILE_IDS: uint16,
}


parquet_adapter = ParquetAdapter(use_arro3=False)


def get_parquet_filepaths(dirpath: str, file_id: Union[int, Series], file_id_n_digits: int) -> str:
    """
    Get standardized parquet file path(s).

    Parameters
    ----------
    dirpath : str
        The directory path to use in the filename.
    file_id : int or Series[int]
        The file ID to use in the filename. If a Series, a list of file paths
        is returned.
    file_id_n_digits : int, optional
        Number of digits to use for 'file_id' in filename.

    Returns
    -------
    Union[str, List[str]]
        The formatted file path(s).

    """
    return (
        (
            f"{dirpath}{DIR_SEP}{PARQUET_FILE_PREFIX}"
            + file_id.astype("string").str.zfill(file_id_n_digits)
            + PARQUET_FILE_EXTENSION
        ).to_list()
        if isinstance(file_id, Series)
        else f"{dirpath}{DIR_SEP}{PARQUET_FILE_PREFIX}{file_id:0{file_id_n_digits}}{PARQUET_FILE_EXTENSION}"
    )


def validate_ordered_on_match(base_ordered_on: str, new_ordered_on: str):
    """
    Check if 'new_ordered_on' is equal to 'base_ordered_on'.

    Raise ValueError if 'new_ordered_on' is not equal to 'base_ordered_on'.

    """
    if base_ordered_on != new_ordered_on:
        raise ValueError(
            f"'ordered_on' parameter value '{new_ordered_on}' does not match "
            f"'{base_ordered_on}' in record dataset.",
        )


class BaseOrderedParquetDataset:
    """
    Base class for Ordered Parquet Dataset with shared functionality.

    This class contains all shared attributes, properties, and methods between
    the full OrderedParquetDataset and its read-only version. Derived classes
    implement specific behaviors for __getitem__ and modification methods.

    Attributes
    ----------
    _file_ids_n_digits : int
        Number of digits to use for 'file_id' in filename. It is kept as an
        attribute to avoid recomputing it at each call to
        'get_parquet_file_name()'.
    _max_allowed_file_id : int
        Maximum allowed file id. Kept as hidden attribute to avoid
        recomputing it at each call in 'write_row_group_files()'.
    _max_n_rows : int
        Maximum allowed number of rows in a row group. Kept as hidden
        attribute to avoid recomputing it at each call in
        'write_row_group_files()'.
    dirpath : str
        Directory path from where to load data.
    is_newly_initialized : bool
        True if this dataset instance was just created and has no existing
        metadata file. False if the dataset was loaded from existing files.
    key_value_metadata : Dict[str, str]
        Key-value metadata, from user and including 'ordered_on' column name.
    max_file_id : int
        Maximum file id in current directory.
    ordered_on : str
        Column name to order row groups by. Can be set either at opd
        instantiation or in 'kwargs' of 'write()' method. Once set, it cannot
        be changed.
    row_group_stats : DataFrame
        Row groups statistics,
          - "ordered_on_min", min value in 'ordered_on' column for this group,
          - "ordered_on_max", max value in 'ordered_on' column for this group,
          - "n_rows": number of rows per row group,
          - "file_id": an int indicating the file id for this group.

    Methods
    -------
    to_pandas()
        Return data as a pandas dataframe.
    write()
        Write data to disk, merging with existing data.
    __getitem__(self, item: Union[int, slice]) -> 'ReadOnlyOrderedParquetDataset'
        Select among the row-groups using integer/slicing.
    __len__()
        Return number of row groups in the dataset.
    _align_file_ids()
        Align file ids to row group position in the dataset.
    _remove_row_group_files()
        Remove row group files from disk. Row group indexes are also removed
        from row_group_stats.
    _sort_row_groups()
        Sort row groups according their min value in 'ordered_on' column.
    _write_metadata_file()
        Write metadata to disk.
    _write_row_group_files()
        Write row group as files to disk. One row group per file.

    Notes
    -----
    - There is one row group per file.
    - Dataset metadata are written in a separate file in parquet format, located
      at the same level than the dataset directory (not within the directory).
      This way, if provided the directory path, another parquet reader can read
      the dataset without being confused by this metadata file.
    - File ids (in file names) have the same number of digits. This is to ensure
      that files can be read in the correct order by other parquet readers.

    """

    def __init__(self, dirpath: str, ordered_on: str = None):
        """
        Initialize BaseOrderedParquetDataset.

        Parameters
        ----------
        dirpath : str
            Directory path from where to load data.
        ordered_on : Optional[str], default None
            Column name to order row groups by. If not initialized, it can also
            be provided in 'kwargs' of 'write()' method.

        """
        self._dirpath = dirpath
        try:
            self._row_group_stats, self._key_value_metadata = parquet_adapter.read_parquet(
                get_md_filepath(self.dirpath),
                return_key_value_metadata=True,
            )
            if ordered_on:
                validate_ordered_on_match(
                    base_ordered_on=self._key_value_metadata[KEY_ORDERED_ON],
                    new_ordered_on=ordered_on,
                )
            self._is_newly_initialized = False
        except FileNotFoundError:
            # Using an empty Dataframe so that it can be written in the case
            # user is only using '_write_metadata_file()' without adding row
            # groups.
            self._row_group_stats = DataFrame(columns=RGS_STATS_COLUMNS).astype(
                RGS_STATS_BASE_DTYPES,
            )
            self._key_value_metadata = {KEY_ORDERED_ON: ordered_on}
            self._is_newly_initialized = True
        # While opd is in memory, 'ordered_on' is kept as a private attribute,
        # with the idea that it is an immutable dataset property, while the
        # content of 'self._key_value_metadata' is mutable.
        self._ordered_on = self._key_value_metadata.pop(KEY_ORDERED_ON)

    def __len__(self):
        """
        Return number of row groups in the dataset.
        """
        return len(self.row_group_stats)

    @cached_property
    def _max_allowed_file_id(self):
        """
        Return maximum allowed file id.
        """
        return iinfo(self.row_group_stats[KEY_FILE_IDS].dtype).max

    @cached_property
    def _file_id_n_digits(self):
        """
        Return number of digits imposed to format file ids in file names.
        """
        return len(str(self._max_allowed_file_id))

    @cached_property
    def _max_n_rows(self):
        """
        Return maximum allowed number of rows in a row group.
        """
        return iinfo(self.row_group_stats[KEY_N_ROWS].dtype).max

    @property
    def dirpath(self):
        """
        Return directory path.
        """
        return self._dirpath

    @property
    def is_newly_initialized(self):
        """
        Return True if this dataset has no existing metadata file.
        """
        return self._is_newly_initialized

    @property
    def key_value_metadata(self):
        """
        Return key-value metadata.
        """
        return self._key_value_metadata

    @property
    def ordered_on(self):
        """
        Return column name to order row groups by.
        """
        return self._ordered_on

    @property
    def row_group_stats(self):
        """
        Return row group statistics.
        """
        return self._row_group_stats

    @property
    def max_file_id(self):
        """
        Return maximum file id in current directory.

        If not row group in directory, return -1.

        """
        # Get max 'file_id' from 'self.row_group_stats'.
        return -1 if self.row_group_stats.empty else int(self.row_group_stats[KEY_FILE_IDS].max())

    def to_pandas(self) -> DataFrame:
        """
        Return data as a pandas dataframe.

        Returns
        -------
        DataFrame
            Dataframe.

        """
        return parquet_adapter.read_parquet(
            get_parquet_filepaths(
                self.dirpath,
                self.row_group_stats[KEY_FILE_IDS],
                self._file_id_n_digits,
            ),
            return_key_value_metadata=False,
        )

    def _align_file_ids(self):
        """
        Align file ids to row group position in the dataset and rename files.

        This method ensures that file ids match their row group positions while:
        1. Minimizing the number of renames.
        2. Avoiding conflicts where target filenames are already taken.
        3. Using temporary filenames when necessary to handle circular
           dependencies.

        """
        # Build mapping of current file ids to desired new ids.
        mask_ids_to_rename = self.row_group_stats.loc[:, KEY_FILE_IDS] != self.row_group_stats.index
        current_ids_to_rename = self.row_group_stats.loc[mask_ids_to_rename, KEY_FILE_IDS]
        if len(current_ids_to_rename) == 0:
            return
        # Initialize 'temp_id' to be used when no direct rename is possible.
        temp_id = self.max_file_id + 1
        new_ids = current_ids_to_rename.index.astype(RGS_STATS_BASE_DTYPES[KEY_FILE_IDS])
        current_to_new = dict(zip(current_ids_to_rename, new_ids))
        # Set of ids already being used by files in directory.
        # Before renaming, we will check the 'new_id' is not already taken.
        ids_already_in_use = set(current_ids_to_rename)
        # Process renames
        while current_to_new:
            # Find a current_id whose new_id is not taken by another current_id.
            for current_id, new_id in list(current_to_new.items()):
                if new_id not in ids_already_in_use:
                    # Safe to rename directly
                    rename(
                        get_parquet_filepaths(self.dirpath, current_id, self._file_id_n_digits),
                        get_parquet_filepaths(self.dirpath, new_id, self._file_id_n_digits),
                    )
                    del current_to_new[current_id]
                    ids_already_in_use.discard(current_id)
                else:
                    # No direct renames possible, need to use temporary id.
                    current_to_new[current_id] = temp_id
                    # Add at bottom of dict the correct mapping.
                    current_to_new[temp_id] = new_id
                    temp_id += 1
                    # Restart the loop.
                    break
        # Set new ids.
        self._row_group_stats.loc[mask_ids_to_rename, KEY_FILE_IDS] = new_ids

    def _remove_row_group_files(
        self,
        file_ids: List[int],
        sort_row_groups: Optional[bool] = True,
        key_value_metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Remove row group files from disk.

        Row group indexes are also removed from 'self.row_group_stats'.

        Parameters
        ----------
        file_ids : List[int]
            File ids to remove.
        sort_row_groups : Optional[bool], default True
            If `True`, sort row groups after removing files.
        key_value_metadata : Optional[Dict[str, str]], default None
            User-defined key-value metadata to write in metadata file.

        Notes
        -----
        After file removal, and optional row group sorting, '_align_file_ids()'
        and '_write_metadata_file()' methods are called, as a result of the
        following reasoning.
        It is anticipated that 'file_ids' may be generated from row group
        indexes. If definition of 'file_ids' from row group indexes occurs in a
        loop where '_remove_row_group_files()' is called, and that row group
        indexes are defined before execution of the loop, then row group indexes
        may not be valid anylonger at a next iteration.
        To mitigate this issue, '_align_file_ids()' and '_write_metadata_file()'
        methods are called, aligning then row group stats in memory and on disk
        ('_opdmd' file) with the existing row group files on disk.

        """
        if not file_ids:
            return
        # Remove files from disk.
        for file_id in file_ids:
            remove(get_parquet_filepaths(self.dirpath, file_id, self._file_id_n_digits))
        # Remove corresponding file ids from 'self.row_group_stats'.
        mask_rows_to_keep = isin(
            self.row_group_stats.loc[:, KEY_FILE_IDS].to_numpy(),
            file_ids,
            invert=True,
        )
        self._row_group_stats = self.row_group_stats.loc[mask_rows_to_keep, :].reset_index(
            drop=True,
        )
        if sort_row_groups:
            self._sort_row_groups()
        self._align_file_ids()
        self._write_metadata_file(key_value_metadata=key_value_metadata)

    def _sort_row_groups(self):
        """
        Sort row groups according their min value in 'ordered_on' column.
        """
        self._row_group_stats.sort_values(by=KEY_ORDERED_ON_MINS, inplace=True, ignore_index=True)

    def write(self, **kwargs):
        """
        Write data to disk.

        This method relies on 'oups.store.write.write()' function.

        Parameters
        ----------
        **kwargs : dict
            Keywords in 'kwargs' are forwarded to `oups.store.write.write()`.

        """
        if self.ordered_on is None:
            if KEY_ORDERED_ON in kwargs:
                self._ordered_on = kwargs.pop(KEY_ORDERED_ON)
            else:
                raise ValueError("'ordered_on' parameter is required.")
        elif KEY_ORDERED_ON in kwargs:
            validate_ordered_on_match(
                base_ordered_on=self.ordered_on,
                new_ordered_on=kwargs.pop(KEY_ORDERED_ON),
            )
        write(self, ordered_on=self.ordered_on, **kwargs)

    def _write_metadata_file(self, key_value_metadata: Dict[str, str] = None):
        """
        Write metadata to disk.

        Metadata are 2 different types of data,
          - ``self.key_value_metadata``, a dict which (key, value) pairs can be
            set by user, and which also contain ``self.ordered_on`` parameter.
            It is retrieved from ``OUPS_METADATA_KEY`` key.
          - ``self.row_group_stats``, a DataFrame which contains row groups
            statistics.

        Parameters
        ----------
        key_value_metadata : Dict[str, str], optional
            User-defined key-value metadata to write, or update in dataset.

        Notes
        -----
        Update strategy of oups specific metadata depends if key found in
        ``OUPS_METADATA`` metadata is also found in already existing metadata,
        as well as its value.
          - If not found in existing, it is added.
          - If found in existing, it is updated.
          - If its value is `None`, it is not added, and if found in existing,
            it is removed from existing.

        Albeit a parquet file, opdmd file is not compressed.

        """
        existing_md = self._key_value_metadata
        if key_value_metadata:
            for key, value in key_value_metadata.items():
                if key in existing_md:
                    if value is None:
                        # Case 'remove'.
                        del existing_md[key]
                    else:
                        # Case 'update'.
                        existing_md[key] = value
                elif value:
                    # Case 'add'.
                    existing_md[key] = value
        if self._is_newly_initialized:
            Path(strip_path_tail(str(self.dirpath))).mkdir(parents=True, exist_ok=True)
        parquet_adapter.write_parquet(
            path=get_md_filepath(self.dirpath),
            df=self.row_group_stats,
            key_value_metadata=existing_md | {KEY_ORDERED_ON: self.ordered_on},
        )
        self._is_newly_initialized = False

    def _write_row_group_files(
        self,
        dfs: Iterable[DataFrame],
        write_metadata_file: bool = True,
        key_value_metadata: Dict[str, str] = None,
        **kwargs,
    ):
        """
        Write row groups as files to disk. One row group per file.

        Parameters
        ----------
        dfs : Iterable[DataFrame]
            Dataframes to write.
        write_metadata_file : bool, optional
            If `True`, write opd metadata file to disk.
        key_value_metadata : Dict[str, str], optional
            User-defined key-value metadata to write, if 'write_metadata_file'
            is `True`.
        **kwargs : dict
            Additional parameters to pass to 'ParquetAdapter.write_parquet()'.

        """
        iter_dfs = iter(dfs)
        try:
            first_df = next(iter_dfs)
        except StopIteration:
            return
        if self.ordered_on not in first_df.columns:
            raise ValueError(
                f"'ordered_on' column '{self.ordered_on}' is not in dataframe columns.",
            )
        buffer = []
        dtype_limit_exceeded = False
        if len(self.row_group_stats) == 0:
            Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        for file_id, df in enumerate(chain([first_df], iter_dfs), start=self.max_file_id + 1):
            if file_id > self._max_allowed_file_id or len(df) > self._max_n_rows:
                dtype_limit_exceeded = True
                break
            buffer.append(
                (
                    file_id,  # file_ids
                    len(df),  # n_rows
                    df.loc[:, self.ordered_on].iloc[0],  # ordered_on_mins
                    df.loc[:, self.ordered_on].iloc[-1],  # ordered_on_maxs
                ),
            )
            parquet_adapter.write_parquet(
                path=get_parquet_filepaths(self.dirpath, file_id, self._file_id_n_digits),
                df=df,
                **kwargs,
            )
        self._row_group_stats = concat(
            [
                None if self.row_group_stats.empty else self.row_group_stats,
                DataFrame(data=buffer, columns=RGS_STATS_COLUMNS).astype(RGS_STATS_BASE_DTYPES),
            ],
            ignore_index=True,
            copy=False,
        )
        if write_metadata_file or dtype_limit_exceeded:
            self._write_metadata_file(key_value_metadata=key_value_metadata)
        if dtype_limit_exceeded:
            if file_id > self._max_allowed_file_id:
                raise ValueError(
                    f"file id '{file_id}' exceeds max value "
                    f"{self._max_allowed_file_id}. Metadata has been written "
                    "before the exception has been raised.",
                )
            else:
                raise ValueError(
                    f"number of rows {len(df)} exceeds max value "
                    f"{self._max_n_rows}. Metadata has been written before the "
                    "exception has been raised.",
                )

    def __getitem__(self, item: Union[int, slice]) -> "ReadOnlyOrderedParquetDataset":
        """
        Select among the row-groups using integer/slicing.

        Parameters
        ----------
        item : int or slice
            Integer or slice to select row groups.

        Returns
        -------
        ReadOnlyOrderedParquetDataset
            A new read-only dataset with the selected row groups.

        """
        # To preserve DataFrame format when selecting single row
        new_row_group_stats = (
            self.row_group_stats.iloc[item : item + 1]
            if isinstance(item, int)
            else self.row_group_stats.iloc[item]
        )

        # Create new instance
        new_opd_subset = object.__new__(BaseOrderedParquetDataset)
        new_opd_subset.__dict__ = self.__dict__ | {
            "_row_group_stats": new_row_group_stats,
        }
        # Lazy import to avoid circular dependency
        from oups.store.ordered_parquet_dataset.ordered_parquet_dataset_read_only import (
            ReadOnlyOrderedParquetDataset,
        )

        return ReadOnlyOrderedParquetDataset._from_instance(new_opd_subset)
