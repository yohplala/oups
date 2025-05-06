#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh

"""
from functools import cached_property
from os import scandir
from pickle import loads

from fastparquet import ParquetFile
from fastparquet import write
from fastparquet.api import statistics
from pandas import DataFrame
from pandas import MultiIndex
from vaex import open_many

from oups.store.defines import DIR_SEP
from oups.store.defines import OUPS_METADATA_KEY


EMPTY_DATAFRAME = DataFrame()


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


class ParquetHandle(ParquetFile):
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

    def __init__(self, dirpath: str, df_like: DataFrame = EMPTY_DATAFRAME):
        """
        Instantiate parquet handle (ParquetFile instance).

        If not existing, create a new one from an empty DataFrame.

        Parameters
        ----------
        dirpath : str
            Directory path from where to load data.
        df_like : Optional[DataFrame], default empty DataFrame
            DataFrame to use as template to create a new ParquetFile.

        """
        try:
            super().__init__(dirpath)
        except (ValueError, FileNotFoundError):
            # In case multi-index is used, check that it complies with fastparquet
            # limitations.
            if isinstance(df_like.columns, MultiIndex):
                check_cmidx(df_like.columns)
            write(dirpath, df_like.iloc[:0], file_scheme="hive")
            super().__init__(dirpath)
        self._dirpath = dirpath

    @property
    def dirpath(self):
        """
        Return dirpath.
        """
        return self._dirpath

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
    def metadata(self) -> dict:
        """
        Return metadata stored when using `oups.writer.write()`.
        """
        return self.pf.key_value_metadata

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
