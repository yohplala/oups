#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh

"""
from copy import deepcopy
from functools import cached_property
from itertools import chain
from os import path as os_path
from os import remove
from os import scandir
from pathlib import Path
from pickle import dumps
from pickle import loads
from typing import Dict, Iterable, List

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from fastparquet.api import statistics
from fastparquet.util import update_custom_metadata
from numpy import iinfo
from numpy import ones
from numpy import uint16
from numpy import uint32
from pandas import DataFrame
from pandas import MultiIndex
from pandas import concat
from vaex import open_many

from oups.defines import DIR_SEP
from oups.defines import OUPS_METADATA_KEY
from oups.store.ordered_parquet_dataset.parquet_adapter import ParquetAdapter
from oups.store.ordered_parquet_dataset.parquet_adapter import check_cmidx
from oups.store.write import write


EMPTY_DATAFRAME = DataFrame()
KEY_ORDERED_ON = "ordered_on"
ORDERED_ON_MINS = "ordered_on_mins"
ORDERED_ON_MAXS = "ordered_on_maxs"
N_ROWS = "n_rows"
FILE_IDS = "file_ids"
# Do not change this order, it is expected by OrderedParquetDataset.write_row_group_files()
RGS_STATS_COLUMNS = [FILE_IDS, N_ROWS, ORDERED_ON_MINS, ORDERED_ON_MAXS]
RGS_STATS_BASE_DTYPES = {
    N_ROWS: uint32,
    FILE_IDS: uint16,
}


def get_parquet_filename(file_id: int, file_id_n_digits: int) -> str:
    """
    Get standardized parquet file name format.

    Parameters
    ----------
    file_id : int
        The file ID to use in the filename.
    file_id_n_digits : int, optional
        Number of digits to use for 'file_id' in filename.

    Returns
    -------
    str
        The formatted file name.

    """
    return f"file_{file_id:0{file_id_n_digits}}.parquet"


class OrderedParquetDataset(ParquetFile):
    """
    Handle to parquet dataset and statistics on disk.

    Attributes
    ----------
    dirpath : str
        Directory path from where to load data.
    pf : ParquetFile
        `ParquetFile` (fastparquet) instance.
    pdf : pDataframe
        Dataframe in pandas format.
    vdf : vDataFrame
        Dataframe in vaex format.

    Methods
    -------
    min_max(col)
        Retrieve min and max from statistics for a column.

    """

    def __init__(self, dirpath: str, ordered_on: str = None, df_like: DataFrame = EMPTY_DATAFRAME):
        """
        Instantiate parquet handle (ParquetFile instance).

        If not existing, create a new one from an empty DataFrame.

        Parameters
        ----------
        dirpath : str
            Directory path from where to load data.
        ordered_on : str
            Column name to order row groups by.
        df_like : Optional[DataFrame], default empty DataFrame
            DataFrame to use as template to create a new ParquetFile.

        """
        try:
            super().__init__(dirpath)
        except (ValueError, FileNotFoundError):
            # In case multi-index is used, check that it complies with fastparquet
            # limitations.
            if df_like is None:
                fp_write(dirpath, DataFrame(), file_scheme="hive")
            else:
                if isinstance(df_like.columns, MultiIndex):
                    check_cmidx(df_like.columns)
                fp_write(dirpath, df_like.iloc[:0], file_scheme="hive")
            super().__init__(dirpath)
        self._dirpath = dirpath
        self._ordered_on = ordered_on

    @property
    def dirpath(self):
        """
        Return dirpath.
        """
        return self._dirpath

    @property
    def ordered_on(self):
        """
        Return ordered_on.
        """
        return self._ordered_on

    def write(self, **kwargs):
        """
        Write data to disk.

        Parameters
        ----------
        **kwargs : dict
            Keywords in 'kwargs' are forwarded to `write.write_ordered`.

        """
        if KEY_ORDERED_ON in kwargs:
            if self._ordered_on is None:
                self._ordered_on = kwargs.pop(KEY_ORDERED_ON)
            elif self._ordered_on != kwargs[KEY_ORDERED_ON]:
                raise ValueError(
                    f"'ordered_on' attribute {self._ordered_on} is not the "
                    f"same as 'ordered_on' parameter {kwargs[KEY_ORDERED_ON]}",
                )
        write(self, ordered_on=self._ordered_on, **kwargs)

    @cached_property
    def pf(self):
        """
        Return handle to data through a parquet file.
        """
        return self

    @property
    def pdf(self):
        """
        Return data as a pandas dataframe.
        """
        return ParquetFile(self._dirpath).to_pandas()

    @property
    def vdf(self):
        """
        Return handle to data through a vaex dataframe.
        """
        # To circumvent vaex lexicographic filename sorting to order row
        # groups, ordering the list of files is required.
        files = [file.name for file in scandir(self._dirpath) if file.name[-7:] == "parquet"]
        files.sort(key=lambda x: int(x[5:-8]))
        prefix_dirpath = f"{str(self._dirpath)}{DIR_SEP}".__add__
        return open_many(map(prefix_dirpath, files))

    @property
    def _oups_metadata(self) -> dict:
        """
        Return specific oups metadata.
        """
        md = self.pf.key_value_metadata
        if OUPS_METADATA_KEY in md:
            return loads(md[OUPS_METADATA_KEY])

    def sort_rgs(self, ordered_on: str):
        """
        Sort row groups by 'ordered_on' column.

        Parameters
        ----------
        ordered_on : str
            Column name to sort row groups by.

        """
        ordered_on_idx = self.columns.index(ordered_on)
        self.fmd.row_groups = sorted(
            self.fmd.row_groups,
            key=lambda rg: statistics(rg.columns[ordered_on_idx])["max"],
        )

    def write_metadata(
        self,
        metadata: Dict[str, str] = None,
    ):
        """
        Write metadata to disk.

        Update oups-specific metadata and merge to user-defined metadata.
        "oups-specific" metadata is retrieved from OUPS_METADATA dict.

        Parameters
        ----------
        pf : ParquetFile
            ParquetFile which metadata are to be updated.
        metadata : Dict[str, str], optional
            User-defined key-value metadata to write, or update in dataset.

        Notes
        -----
        - These specific oups metadata are available in global variable
        ``OUPS_METADATA``.
        - Update strategy of oups specific metadata depends if key found in
        ``OUPS_METADATA``metadata` is also found in already existing metadata,
        as well as its value.

        - If not found in existing, it is added.
        - If found in existing, it is updated.
        - If its value is `None`, it is not added, and if found in existing, it
            is removed from existing.

        """
        if metadata:
            new_oups_spec_md = metadata
            if OUPS_METADATA_KEY in (existing_metadata := self.key_value_metadata):
                # Case 'append' to existing metadata.
                # oups-specific metadata is expected to be a dict itself.
                # To be noticed, 'md_key' is not written itself in metadata to
                # disk.
                existing_oups_spec_md = loads(existing_metadata[OUPS_METADATA_KEY])
                for key, value in new_oups_spec_md.items():
                    if key in existing_oups_spec_md:
                        if value is None:
                            # Case 'remove'.
                            del existing_oups_spec_md[key]
                        else:
                            # Case 'update'.
                            existing_oups_spec_md[key] = value
                    elif value:
                        # Case 'add'.
                        existing_oups_spec_md[key] = value
            else:
                existing_oups_spec_md = new_oups_spec_md
            update_custom_metadata(self, {OUPS_METADATA_KEY: dumps(existing_oups_spec_md)})
        self._write_common_metadata()


parquet_adapter = ParquetAdapter(use_arro3=False)


class OrderedParquetDataset2:
    """
    Ordered Parquet Dataset.

    Attributes
    ----------
    _file_ids_n_digits : int
        Number of digits to use for 'file_id' in filename. It is kept as an
        attribute to avoid recomputing it at each call to
        'get_parquet_file_name()'.
    _max_file_id : int
        Maximum allowed file id. Kept as hidden attribute to avoid
        recomputing it at each call in 'write_row_group_files()'.
    _max_n_rows : int
        Maximum allowed number of rows in a row group. Kept as hidden
        attribute to avoid recomputing it at each call in
        'write_row_group_files()'.
    dirpath : str
        Directory path from where to load data.
    forbidden_to_write_row_group_files : bool
        Flag warning if writing is blocked. Writing is blocked when the opd
        object is a subset (use of '_getitem__' method).
        This flag is a security to prevent adding then new row group files
        without knowing all file ids that may already exist in the dataset.
        To effectively remove row group files, use 'remove_row_group_files()'
        method.
    forbidden_to_remove_row_group_files : bool
        Flag warning if removing is blocked. Removing is blocked when
        'remove_row_group_files()' method is called, but reset when
        'write_metadata()' method is then called.
        This flag is a security to prevent iterating
        'remove_row_group_files()' method without the sense that the process
        involving the current opd object has been completed first.
        It is anticipated/expected that the completion of such a process
        involves a 'write_metadata()' step.
    kvm : dict
        Key-value metadata, from user and including 'ordered_on' column name.
    ordered_on : str
        Column name to order row groups by.
    row_group_stats : DataFrame
        Row groups statistics,
          - "ordered_on_min", min value in 'ordered_on' column for this group,
          - "ordered_on_max", max value in 'ordered_on' column for this group,
          - "n_rows": number of rows per row group,
          - "file_id": an int indicating the file id for this group.

    Methods
    -------
    __getitem__()
        Select among the row-groups using integer/slicing.
    __len__()
        Return number of row groups in the dataset.
    align_file_ids()
        Align file ids to row group position in the dataset.
    remove_row_group_files()
        Remove row group files from disk. Row group indexes are also removed
        from OrderedParquetDataset.row_group_stats.
    sort_row_groups()
        Sort row groups according their min value in 'ordered_on' column.
    to_pandas()
        Return data as a pandas dataframe.
    write()
        Write data to disk, merging with existing data.
    write_metadata()
        Write metadata to disk.
    write_row_group_files()
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
        Initialize OrderedParquetDataset.

        Parameters
        ----------
        dirpath : str
            Directory path from where to load data.
        ordered_on : Optional[str], default None
            Column name to order row groups by.

        """
        self.dirpath = dirpath
        try:
            self.row_group_stats, self.kvm = parquet_adapter.read_parquet(
                f"{self.dirpath}_opdmd",
                return_metadata=True,
            )
            if ordered_on and self.kvm[KEY_ORDERED_ON] != ordered_on:
                raise ValueError(
                    f"'ordered_on' parameter value '{ordered_on}' does not match "
                    f"'{self.kvm[KEY_ORDERED_ON]}' in record dataset.",
                )
        except FileNotFoundError:
            # Using an empty Dataframe so that it can be written in the case
            # user is only using 'write_metadata()' without adding row groups.
            if ordered_on is None:
                raise ValueError("'ordered_on' column name must be provided.")
            self.row_group_stats = DataFrame(columns=RGS_STATS_COLUMNS).astype(
                RGS_STATS_BASE_DTYPES,
            )
            self.kvm = {KEY_ORDERED_ON: ordered_on}
        self.ordered_on = self.kvm[KEY_ORDERED_ON]
        self.forbidden_to_write_row_group_files = False
        self.forbidden_to_remove_row_group_files = False

    def __getitem__(self, item):
        """
        Select among the row-groups using integer/slicing.

        Parameters
        ----------
        item : int or slice
            Integer or slice to select row groups.

        Returns
        -------
        OrderedParquetDataset

        """
        new_opd = object.__new__(OrderedParquetDataset2)
        # To preserve Dataframe format.
        new_row_group_stats = (
            self.row_group_stats.iloc[item : item + 1]
            if isinstance(item, int)
            else self.row_group_stats.iloc[item]
        )
        new_opd.__dict__ = self.__dict__ | {
            "forbidden_to_write_row_group_files": True,
            "kvm": deepcopy(self.kvm),
            "row_group_stats": new_row_group_stats,
        }
        return new_opd

    def __len__(self):
        """
        Return number of row groups in the dataset.
        """
        return len(self.row_group_stats)

    @cached_property
    def _max_file_id(self):
        """
        Return maximum allowed file id.
        """
        return iinfo(self.row_group_stats[FILE_IDS].dtype).max

    @cached_property
    def _file_id_n_digits(self):
        """
        Return number of digits imposed to format file ids in file names.
        """
        return len(str(self._max_file_id))

    @cached_property
    def _max_n_rows(self):
        """
        Return maximum allowed number of rows in a row group.
        """
        return iinfo(self.row_group_stats[N_ROWS].dtype).max

    def align_file_ids(self):
        """
        Align file ids to row group position in the dataset.

        This method also formats file ids in file names to have the same number
        of digits.

        Notes
        -----
        - Enforcing file ids number of digits is done so that the files of
          dataset can be read seamlessly in the correct order by other parquet
          readers.

        """
        pass

    def remove_row_group_files(self, file_ids: List[int]):
        """
        Remove row group files from disk.

        Row group indexes are also removed from 'self.row_group_stats'.

        Parameters
        ----------
        file_ids : Iterable[int]
            File ids to remove.

        Notes
        -----
        It is anticipated that 'file_ids' may be generated from row group
        indexes. If definition of 'file_ids' from row group indexes occurs in a
        loop where 'remove_row_group_files()' is called, and that row group
        indexes are defined before execution of the loop, then row group indexes
        may not be valid anylonger at a next iteration.
        To mitigate this issue, removing row group files is blocked after
        running 'remove_row_group_files()' once by using
        'forbidden_to_remove_row_group_files' flag.
        This flag is a security to prevent iterating 'remove_row_group_files()'
        method without the sense that the process involving the current opd
        object has been completed first. It is anticipated/expected that the
        completion of such a process involves a 'write_metadata()' step.
        For this reason, writing opdmd data right after removing row group files
        is not proposed either.
        The 'forbidden_to_remove_row_group_files' flag is then reset when
        'write_metadata()' method is called.

        """
        if self.forbidden_to_remove_row_group_files:
            raise ValueError("removing row group files is blocked.")
        # Remove files from disk.
        for file_id in file_ids:
            remove(
                os_path.join(self.dirpath, get_parquet_filename(file_id, self._file_id_n_digits)),
            )
        # Remove corresponding file ids from 'self.row_group_stats'.
        ids_to_keep = ones(len(self.row_group_stats), dtype=bool)
        ids_to_keep[file_ids] = False
        self.row_group_stats = (
            self.row_group_stats.set_index(FILE_IDS).iloc[ids_to_keep].reset_index()
        )
        self.forbidden_to_remove_row_group_files = True

    def to_pandas(self):
        """
        Return data as a pandas dataframe.

        Returns
        -------
        pandas.DataFrame
            Dataframe.

        """
        pass

    def write_metadata(self, metadata: Dict[str, str] = None):
        """
        Write metadata to disk.

        Metadata are 2 different types of data,
          - ``self.kvm``, a dict which values can be set by user, and which also
            contain ``self.ordered_on`` parameter.
          - ``self.row_group_stats``
        oups metadata is retrieved from ``OUPS_METADATA_KEY`` key.

        Parameters
        ----------
        metadata : Dict[str, str], optional
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

        """
        existing_md = self.kvm
        if metadata:
            for key, value in metadata.items():
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
        parquet_adapter.write_parquet(
            path=f"{self.dirpath}_opdmd",
            df=self.row_group_stats,
            metadata=existing_md,
        )
        # Reset 'forbidden_to_remove_row_group_files' flag, in case it was set.
        # This flag is a security to prevent iterating
        # 'remove_row_group_files()' method without the sense that the process
        # involving the current opd object has been completed first.
        # It is anticipated/expected that the completion of such a process
        # involves a 'write_metadata()' step.
        self.forbidden_to_remove_row_group_files = False

    def write_row_group_files(self, dfs: Iterable[DataFrame], write_opdmd: bool = True):
        """
        Write row group as files to disk. One row group per file.

        Parameters
        ----------
        dfs : Iterable[DataFrame]
            Dataframes to write.

        """
        if self.forbidden_to_write_row_group_files:
            raise ValueError("writing row group files is blocked.")
        iter_dfs = iter(dfs)
        first_df = next(iter_dfs)
        if self.ordered_on not in first_df.columns:
            raise ValueError(
                f"'ordered_on' column '{self.ordered_on}' is not in dataframe columns.",
            )
        dfs = chain([first_df], iter_dfs)
        buffer = []
        file_id = (
            0 if self.row_group_stats.empty else self.row_group_stats.loc[:, FILE_IDS].max() + 1
        )
        max_file_id_exceeded = False
        max_n_rows_exceeded = False
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        for df in dfs:
            if file_id > self._max_file_id:
                max_file_id_exceeded = True
                break
            if len(df) > self._max_n_rows:
                max_n_rows_exceeded = True
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
                path=os_path.join(
                    self.dirpath,
                    get_parquet_filename(file_id, self._file_id_n_digits),
                ),
                df=df,
            )
            file_id += 1
        self.row_group_stats = concat(
            [
                None if self.row_group_stats.empty else self.row_group_stats,
                DataFrame(data=buffer, columns=RGS_STATS_COLUMNS).astype(RGS_STATS_BASE_DTYPES),
            ],
            ignore_index=True,
            copy=False,
        )
        if write_opdmd or max_file_id_exceeded or max_n_rows_exceeded:
            self.write_metadata()
        if max_file_id_exceeded:
            raise ValueError(
                f"file id '{file_id}' exceeds max value {self._max_file_id}. "
                "Metadata has been written before the exception has been raised.",
            )
        if max_n_rows_exceeded:
            raise ValueError(
                f"number of rows {len(df)} exceeds max value {self._max_n_rows}. "
                "Metadata has been written before the exception has been raised.",
            )


# TODO:
# Create:
#  - __len__: number of row_groups to check if empty opd or not?
#  - __getitem__ / to_pandas(): test to_pandas on row group subset
#  - clean oups.store.write.write() and colllection.py
#  - rename collection.py
#  - remove vaex dependency
#  - set numpy above 2.0
#  - test case when reaching MAX_FILE_ID
#  - test case in write when removing 2 sequence of row groups, to check that
#    when not based on row group indexes, but on file_ids, it works.
