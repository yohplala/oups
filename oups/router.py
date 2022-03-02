#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh
"""
from os import scandir

from fastparquet import ParquetFile
from vaex import open_many

from oups.defines import DIR_SEP


class ParquetHandle:
    """Handle to parquet dataset and statistics on disk.

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

    def __init__(self, dirpath: str):
        """Instantiate parquet handle.

        Parameter
        ---------
        dirpath : str
            Directory path from where to load data.
        """
        self._dirpath = dirpath

    @property
    def dirpath(self):
        """Return dirpath."""
        return self._dirpath

    @property
    def pf(self):
        """Return handle to data through a parquet file."""
        return ParquetFile(self._dirpath)

    @property
    def pdf(self):
        """Return data as a pandas dataframe."""
        return ParquetFile(self._dirpath).to_pandas()

    @property
    def vdf(self):
        """Return handle to data through a vaex dataframe."""
        # To circumvent vaex lexicographic filename sorting to order row
        # groups, ordering the list of files is required.
        files = [file.name for file in scandir(self._dirpath) if file.name[-7:] == "parquet"]
        files.sort(key=lambda x: int(x[5:-8]))
        prefix_dirpath = f"{str(self._dirpath)}{DIR_SEP}".__add__
        return open_many(map(prefix_dirpath, files))

    def min_max(self, col: str) -> tuple:
        """Return min and max of values of a column.

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
