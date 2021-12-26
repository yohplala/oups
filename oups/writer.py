#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:30:00 2021
@author: yoh
"""
from typing import List, Union

from fastparquet import ParquetFile, write as fp_write
from pandas import DataFrame as pDataFrame, Index, MultiIndex
from vaex.dataframe import DataFrame as vDataFrame


ROW_GROUP_SIZE = 5_000_000
COMPRESSION = 'SNAPPY'

def iter_dataframe(data: Union[pDataFrame, vDataFrame],
                   row_group_size:int=None):
    """Generator yielding dataframe chunks.

    Parameters
    ----------
    data : Union[pDataFrame, vDataFrame]
        Data to split in row groups.
    row_group_size : int, default 50_000_000
        Size of row groups.

    Yields
    ------
    pDataFrame
        Chunk of data.
    """
    if row_group_size is None:
        row_group_size = ROW_GROUP_SIZE
    if isinstance(data, vDataFrame):
        # Drop any possible the lazy indexing, to make the length of data
        # equals its filtered length
        data = data.extract()
    # TODO: implement 'replicate_groups' (use of 'ordered_on' column).
    # Define row group offsets.
    # Acknowledging this piece of code to be an extract from fastparquet.
    n_rows = len(data)
    n_parts = (n_rows-1)//row_group_size + 1
    row_group_size = min((n_rows-1)//n_parts+1, n_rows)
    starts = list(range(0, n_rows, row_group_size))
    ends = starts[1:] + [None]
    if isinstance(data, vDataFrame):
        for start, end in zip(starts,ends):
            yield data[start:end].to_pandas_df()
    else:
        for start, end in zip(starts,ends):
            yield data.iloc[start:end]

def to_midx(idx: Index, sep:str, levels:List[str]=None) -> MultiIndex:
    """Expand a pandas index into a multi-index.

    Parameters
    ----------
    idx : Index
        Pandas index, with values containing or not `cmidx_sep`.
    sep : str
        Separator to split names (index values).

    Returns
    -------
    MultiIndex
        Pandas multi-index.

    Notes
    -----
    If some column names have fewer occurences of `sep` (resulting in fewer
    index levels), these column names are appended with empty strings '' as
    required to be of equal levels number than the longest column names.
    """
    idx = [tuple(s.split(sep)) for s in idx]
    # Get max number of levels.
    max_levels = max(map(len,idx))
    # Generate names of levels if required.
    diff = 0
    if levels is None:
        levels = []
        len_lev = 0
        diff = max_levels
    elif (len_lev := len(levels)) < max_levels:
        diff = max_levels - len_lev
    if diff > 0:
        levels.extend([f'l{i}' for i in range(len_lev, max_levels)])
    # Equalize length of tuples.
    tuples = [(*t, *['']*n) if (n:=(max_levels-len(t))) else t for t in idx]
    return MultiIndex.from_tuples(tuples, names=levels)

def write(dirpath:str, data:Union[pDataFrame, vDataFrame],
          row_group_size:int=None, compression:str=COMPRESSION,
          cmidx_expand:bool=False, cmidx_sep:str=None,
          cmidx_levels:List[str]=None):
    """Write data to disk at location specified by path.

    Parameters
    ----------
    dirpath : str
        Directory where writing pandas dataframe.
    data : Union[pDataFrame, vDataFrame]
        Data to write.
    row_group_size : int, optional
        Max row group size. If not set, default to 50_000_000.
    compression : str, default SNAPPY
        Algorithm to use for compressing data. This parameter is fastparquet
        specific. Please see fastparquet documentation for more information.
    cmidx_expand : bool, default False
        If `True`, expand column index into a column multi-index.
        This requires `cmidx_sep` to be provided.
        This parameter is only used at creation of the dataset. Once column
        names are set, they cannot be modified by use of this parameter.
    cmidx_sep : str, optional
        Characters with which splitting column names to expand them into a
        column multi-index. Required if `cmidx=True`.
        This parameter is only used at creation of the dataset. Once column
        names are set, they cannot be modified by use of this parameter.
    cmidx_levels : List[str], optional
        Names of levels to be used when expanding column names into a
        multi-index. If not provided, levels are given names 'l1', 'l2', ...

    Notes
    -----
    When writing a dataframe with this function,
     - index of dataframe is not written to disk.
     - parquet file scheme is 'hive' (one row group per parquet file).
    """
    iter_data = iter_dataframe(data, row_group_size)
    try:
        pf = ParquetFile(dirpath)
    except (FileNotFoundError, ValueError):
        # First time writing.
        chunk = next(iter_data)
        if cmidx_expand and cmidx_sep:
            chunk.columns = to_midx(chunk.columns, cmidx_sep, cmidx_levels)
        elif cmidx_expand and not cmidx_sep:
            raise ValueError('Setting `cmidx` but not `cmidx_sep` is not '
                             'possible.')
        fp_write(dirpath, chunk, row_group_offsets=row_group_size,
                 compression=compression, file_scheme='hive',
                 write_index=False, append=False)
        # Re-open to write remaining chunks.
        pf = ParquetFile(dirpath)
    # Appending
    pf.write_row_groups(data=iter_data, row_group_offsets=None,
                        compression=compression)
    # TODO: implement 'update'.
    # (to be run depending value on 'ordered_on' column)
