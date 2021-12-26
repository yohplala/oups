#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:30:00 2021
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
    def __init__(self, dirpath:str):
        """
        Parameter
        ---------
        dirpath : str
            Directory path from where to load data.
        """
        self._dirpath = dirpath

    @property
    def dirpath(self):
        return self._dirpath

    @property
    def pf(self):
        return ParquetFile(self._dirpath)

    @property
    def pdf(self):
        return ParquetFile(self._dirpath).to_pandas()

    @property
    def vdf(self):
        return vx_open(self._dirpath)
