#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:30:00 2021
@author: yoh
"""
from os import path as os_path, makedirs
from typing import List, Union

from fastparquet import ParquetFile, write as fp_write
from pandas import DataFrame as pDataFrame, Index, MultiIndex
from vaex.dataframe import DataFrame as vDataFrame


def to_idx(midx: MultiIndex, sep:str) -> Index:
    """Flatten a pandas multi-index to a single level one.

    Parameters
    ----------
    midx : MultiIndex
        Pandas multi-index.
    sep : str
        Separator around which joining index values.

    Returns
    -------
    MultiIndex
        List of single level column names expanded into a pandas multi-index.

    Notes
    -----
    If some column names have fewer occurences of `sep` (resulting in fewer
    index levels), these column names are appended with empty strings '' as
    required to be of equal levels number than the longest column names.
    """
    cmidx = [tuple(s.split(sep)) for s in cmidx]
    # Get max number of levels.
    max_levels = max(map(len,cmidx))
    # Equalize length of tuples.
    cmidx = [(*t, *['']*n) if (n:=(max_levels-len(t))) else t for t in cmidx]
    return MultiIndex.from_tuples(cmidx)

def write(path:str, data:Union[pDataFrame, vDataFrame],
          row_group_size:int=None, compression:str=COMPRESSION,
          expand_cmidx:bool=False, cmidx_sep:str=None):
    """Write data to disk at location specified by path.

    Parameters
    ----------
    path : str
        Directory where writing pandas dataframe.
    data : Union[pDataFrame, vDataFrame]
        Data to write.
    row_group_size : int, optional
        Max row group size. If not set, default to 50_000_000.
    compression : str, default SNAPPY
        Algorithm to use for compressing data. This parameter is fastparquet
        specific. Please see fastparquet documentation for more information.
    expand_cmidx : bool, default False
        If `True`, expand column index into a column multi-index.
        This requires `cmidx_sep` to be provided.
        This parameter is only used at creation of the dataset. Once column
        names are set, they cannot be modified by use of this parameter.
    cmidx_sep : str, optional
        Characters with which splitting column names to expand them into a
        column multi-index. Required if `cmidx=True`.
        This parameter is only used at creation of the dataset. Once column
        names are set, they cannot be modified by use of this parameter.

    Notes
    -----
    When writing a dataframe with this function,
     - index of dataframe is not written to disk.
     - file scheme used is 'hive'.
    """
    iter_data = iter_dataframe(data, row_group_size)
    try:
        pf = ParquetFile(path)
    except (FileNotFoundError, ValueError):
        # First time writing.
        chunk = next(iter_data)
        if expand_cmidx and cmidx_sep:
            chunk.columns = to_cmidx(chunk.columns, cmidx_sep)
        elif expand_cmidx and not cmidx_sep:
            raise ValueError('Setting `cmidx` but not `cmidx_sep` is not '
                             'possible.')
        fp_write(path, chunk, row_group_offsets=row_group_size,
                 compression=compression, file_scheme='hive',
                 write_index=False, append=False)
        # Re-open to write remaining chunks.
        pf = ParquetFile(path)
    # Appending
    pf.write_row_groups(data=iter_data, row_group_offsets=None,
                        compression=compression)
    # TODO: case 'update' by comparing value on 'ordered_on' column.
