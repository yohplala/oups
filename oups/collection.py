#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:00:00 2021
@author: yoh
"""
from dataclasses import dataclass
from sortedcontainers import SortedList
from typing import Any, Iterator, List, Type, Tuple

from oups.defines import DIR_SEP
from oups.indexer import is_toplevel
from oups.utils import files_at_depth


def is_parquet_file(file:str) -> bool:
    """
    Return True if file name is identified as a parquet file, i.e. ending with
    '.parquet' or '.parq' or being named '_metadata' or '_common_metadata'.
    """
    return (file.endswith('.parquet') or file.endswith('.parquet')
            or file == '_metadata' or file == '_common_metadata')

# basepath = '/home/yoh/Documents/code/data/oups'
def get_keys(basepath:str, indexer:Type[dataclass]) -> SortedList[dataclass]:
    """
    Scan 'basepath' directory and create instances of 'indexer' class from
    compatible subpath.
    Only non-empty directories, with files either ending with '.parq' or
    '.parquet' extension, or named '_medatada' are kept.
    
    Parameters
    basepath : str
        Path to directory containing a dataset collection, in folders complying
        with the schema defined by 'indexer' dataclass.
    indexer: Type[dataclass]
        Class decorated with '@toplevel' decorator, and defining a path
        schema.

    Returns
    SortedList[dataclass]: 
        Sorted list of keys (i.e. instances of indexer class) that can be
        found in 'basepath' directory and with a 'valid' dataset (directory
        with files either ending with '.parq' or '.parquet' extension, or named
        '_medatada').
    """
    depth = indexer._depth
    paths_files = files_at_depth(basepath, depth)
    # Filter, keeping only folders having files with correct extension,
    # then materialize paths into keys, filtering out those that can't.
    keys = SortedList(
               [indexer.from_path(DIR_SEP.join(path.rsplit(DIR_SEP,depth)[1:]))
                for path, files in paths_files
                if any((is_parquet_file(file) for file in files))])
    return keys

class ParquetCollection:
    """
    Store keys of a type defined at 'ParquetStore' instanciation as a sorted
    list.
    """
    def __init__(self, basepath: str, indexer: object):
        # Check.
        if not is_toplevel(indexer):
            raise TypeError(f'Indexer not decorated with "@toplevel".')
        self.keys = get_keys(basepath, indexer)




#        self.update(dict(*args, **kwargs))  # use the free update to set keys





#    def __getitem__(self, key):
#        return self.store[self._keytransform(key)]

#    def __setitem__(self, key, value):
#        self.store[self._keytransform(key)] = value

#    def __delitem__(self, key):
#        del self.store[self._keytransform(key)]

#    def __iter__(self):
#        return iter(self.store)
    
#    def __len__(self):
#        return len(self.store)

#    def _keytransform(self, key):
#        return key
