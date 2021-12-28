#!/usr/bin/env python3
"""
Created on Wed Dec 26 22:30:00 2021.

@author: yoh
"""
from fastparquet import ParquetFile
from vaex import open as vx_open


class ParquetHandle:
    """Handle to parquet dataset on disk.

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
        """Return parquet file."""
        return ParquetFile(self._dirpath)

    @property
    def pdf(self):
        """Return pandas dataframe."""
        return ParquetFile(self._dirpath).to_pandas()

    @property
    def vdf(self):
        """Return vaex dataframe."""
        return vx_open(self._dirpath)
