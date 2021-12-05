#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:00:00 2021
@author: yoh
"""
from dataclasses import dataclass
from sortedcontainers import SortedSet
from typing import Type

from oups.defines import DIR_SEP
from oups.indexer import is_toplevel
from oups.utils import files_at_depth


def is_parquet_file(file:str) -> bool:
    """
    Return True if file name is identified as a parquet file, i.e. ending with
    '.parquet' or '.parq' or being named '_metadata' or '_common_metadata'.
    """
    return (file.endswith('.parquet') or file.endswith('.parq')
            or file == '_metadata' or file == '_common_metadata')

def get_keys(basepath:str, indexer:Type[dataclass]) -> SortedSet:
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
    SortedSet[dataclass]: 
        Sorted set of keys (i.e. instances of indexer dataclass) that can be
        found in 'basepath' directory and with a 'valid' dataset (directory
        with files either ending with '.parq' or '.parquet' extension, or named
        '_medatada').
    """
    depth = indexer.depth
    paths_files = files_at_depth(basepath, depth)
    # Filter, keeping only folders having files with correct extension,
    # then materialize paths into keys, filtering out those that can't.
    keys = SortedSet([key for path, files in paths_files
                      if (any((is_parquet_file(file) for file in files))
                          and (key := indexer.from_path(
                                  DIR_SEP.join(path.rsplit(DIR_SEP,depth)[1:]))
                              ))])
    return keys

class ParquetSet:
    """
    Sorted set of keys (indexes to parquet datasets).
    """
    def __init__(self, basepath:str, indexer:Type[dataclass]):
        """
        Parameters
        basepath : str
            Path of directory containing parquet datasets.
        indexer : Type[dataclass]
            Class (not class instance) of the indexer to be used for:
            - identifying existing parquet datasets in 'basepath' directory,
            - creating the folders where recording new parquet datasets.

        Attributes
        basepath : str
        indexer : Type[dataclass]
        """
        if not is_toplevel(indexer):
            raise TypeError('Indexer not decorated with "@toplevel".')
        self._keys = get_keys(basepath, indexer)
        self._basepath = basepath
        self._indexer = indexer

    @property
    def basepath(self):
        return self._basepath
    
    @property
    def indexer(self):
        return self._indexer

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return '\n'.join(map(str,self._keys))



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
