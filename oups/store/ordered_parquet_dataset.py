#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh

"""
from base64 import b64decode
from base64 import b64encode
from functools import cached_property
from os import path as os_path
from os import scandir
from pickle import dumps
from pickle import loads
from typing import Dict, Iterable

# from pandas import read_parquet
from arro3.io import read_parquet
from arro3.io import write_parquet
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from fastparquet.api import statistics
from fastparquet.util import update_custom_metadata
from numpy import uint16
from numpy import uint32
from pandas import DataFrame
from pandas import MultiIndex
from pandas import concat
from vaex import open_many

from oups.defines import DIR_SEP
from oups.defines import OUPS_METADATA_KEY
from oups.store.write import write


EMPTY_DATAFRAME = DataFrame()
KEY_ORDERED_ON = "ordered_on"
ORDERED_ON_MIN = "ordered_on_min"
ORDERED_ON_MAX = "ordered_on_max"
N_ROWS = "n_rows"
PART_ID = "part_id"
RGS_STATS_COLUMNS = [ORDERED_ON_MIN, ORDERED_ON_MAX, N_ROWS, PART_ID]
MIN_PART_ID_N_DIGITS = 4
RGS_STATS_BASE_DTYPES = {
    N_ROWS: uint32,
    PART_ID: uint16,
}
EMPTY_RGS_STATS = DataFrame(columns=RGS_STATS_COLUMNS).astype(RGS_STATS_BASE_DTYPES)


def check_cmidx(cmidx: MultiIndex):
    """
    Check if column multi-index complies with fastparquet requirements.

    Library fastparquet requires names for each level in a Multiindex.
    Also, column names have to be tuple of string.

    Parameters
    ----------
    cmidx : MultiIndex
        MultiIndex to check.

    """
    # Check level names.
    if None in cmidx.names:
        raise ValueError(
            "not possible to have level name set to None.",
        )  # If an item of the column name is not a string, turn it into a string.
    # Check column names.
    for level in cmidx.levels:
        for name in level:
            if not isinstance(name, str):
                raise TypeError(f"name {name} has to be of type 'string', not '{type(name)}'.")


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

    def min_max(self, col: str) -> tuple:
        """
        Return min and max of values of a column.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        tuple
            Min and max values of column.

        """
        pf_stats = ParquetFile(self._dirpath).statistics
        return (min(pf_stats["min"][col]), max(pf_stats["max"][col]))

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


class OrderedParquetDataset2:
    """
    Ordered Parquet Dataset.

    Attributes
    ----------
    dirpath : str
        Directory path from where to load data.
    kvm : dict
        Custom key-value metadata.
    ordered_on : str
        Column name to order row groups by.
    rgs_stats : DataFrame
        Row groups statistics.

    Methods
    -------
    __getitem__()
        Select among the row-groups using integer/slicing.
    align_part_names()
        Align part names to row group position in the dataset. Also format
        part names to have the same number of digits.
    sort_rgs()
        Sort row groups according their min value in 'ordered_on' column.
    to_pandas()
        Return data as a pandas dataframe.
    write()
        Write data to disk.
    write_metadata()
        Write metadata to disk.
    write_row_group_files()
        Write row group as files to disk. One row group per file.

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
        self.ordered_on = ordered_on
        try:
            record_batch = read_parquet(f"{self.dirpath}_opdmd").read_all()
            self.rgs_stats = (
                EMPTY_RGS_STATS
                if len(record_batch) == 0
                else DataFrame(record_batch.to_struct_array())
            )
            self.kvm = loads(
                b64decode((record_batch.schema.metadata_str[OUPS_METADATA_KEY]).encode()),
            )
            if ordered_on is not None and self.kvm[KEY_ORDERED_ON] != self.ordered_on:
                raise ValueError(
                    f"ordered_on {self.kvm[KEY_ORDERED_ON]} in record dataset does not match {self.ordered_on} in constructor.",
                )
        except FileNotFoundError:
            # Using an empty Dataframe so that it can be written in the case
            # user is only using 'write_metadata()' without adding row groups.
            self.rgs_stats = EMPTY_RGS_STATS
            self.kvm = {KEY_ORDERED_ON: self.ordered_on}

    def write_metadata(self, metadata: Dict[str, str] = None):
        """
        Write metadata to disk.
        """
        if metadata:
            self.kvm.update(metadata)
        write_parquet(
            self.rgs_stats,
            f"{self.dirpath}_opdmd",
            key_value_metadata={OUPS_METADATA_KEY: b64encode(dumps(self.kvm)).decode()},
        )

    def write_row_group_files(self, dfs: Iterable[DataFrame], write_opdmd: bool = True):
        """
        Write row group as files to disk. One row group per file.

        Parameters
        ----------
        dfs : Iterable[DataFrame]
            Dataframes to write.

        """
        ordered_on_mins = []
        ordered_on_maxs = []
        n_rows = []
        part_id = 0 if self.rgs_stats.empty else self.rgs_stats.loc[:, PART_ID].max()
        part_ids = []
        part_id_n_digits = max(MIN_PART_ID_N_DIGITS, len(str(part_id)))
        for df in dfs:
            ordered_on_mins.append(df.loc[:, self.ordered_on].iloc[0])
            ordered_on_maxs.append(df.loc[:, self.ordered_on].iloc[-1])
            n_rows.append(len(df))
            part_id += 1
            part_ids.append(part_id)
            write_parquet(
                df,
                os_path.join(
                    self.dirpath,
                    f"part_{part_id:0{part_id_n_digits}}.parquet",
                ),
            )
        self.rgs_stats = concat(
            [
                None if self.rgs_stats.empty else self.rgs_stats,
                DataFrame(
                    {
                        ORDERED_ON_MIN: ordered_on_mins,
                        ORDERED_ON_MAX: ordered_on_maxs,
                        N_ROWS: n_rows,
                        PART_ID: part_ids,
                    },
                ).astype(RGS_STATS_BASE_DTYPES),
            ],
            ignore_index=True,
            copy=False,
        )
        if write_opdmd:
            self.write_metadata()

    def to_pandas(self):
        """
        Return data as a pandas dataframe.
        """
        return DataFrame


# TODO:
# Create:
#  - __init__
#      - with sorting parquet file names if _opd_metadata is not existing
#  - clean oups.store.write.write() and colllection.py
#  - rename collection.py
#  - remove vaex dependency
#  - set numpy above 2.0
