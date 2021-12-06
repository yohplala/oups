#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:30:00 2021
@author: yoh
"""
from os import path as os_path
from typing import Any, Type, Union

import fastparquet as fp
from pandas import DataFrame as pDataFrame
from vaex.dataframe import DataFrame as vDataFrame


def write_pandas_dataframe(path:str, data:pDataFrame, row_group_size:int):
    """
    Write pandas dataframe to disk at location specified by path.
    
    Usage notes
     - Index of dataframe is not written to disk.
     - Compression is set to 'SNAPPY'.

    Parameters
    path : str
        Directory where writing pandas dataframe.
    data : pDataFrame
        Data to write.
    row_group_size : int
        Max row group size.
    """
    fp.write(
             filename=path,
             data=data,
             row_group_offsets=row_group_size,
             file_scheme='hive',
             write_index=False,
             compression='SNAPPY'
            )
    return

def write_vaex_dataframe(path:str, data:vDataFrame, row_group_size:int):
    """
    Write vaex dataframe to disk at location specified by path by calling
    'write_pandas_dataframe'.

    Parameters
    path : str
        Directory where writing pandas dataframe.
    data : pDataFrame
        Data to write.
    row_group_size : int
        Max row group size.
    """
    # Iterator delivering row numbers for start and end of the chunk,
    # then the chunk.
    for _, _, chunk in data.to_pandas_df(chunk_size=row_group_size):
        write_pandas_dataframe(path, chunk, row_group_size)
    return

