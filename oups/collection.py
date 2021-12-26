#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:00:00 2021
@author: yoh
"""
from dataclasses import dataclass
from os import path as os_path
from sortedcontainers import SortedSet
from typing import Type, Union

from pandas import DataFrame as pDataFrame
from vaex.dataframe import DataFrame as vDataFrame

from oups.defines import DIR_SEP
from oups.indexer import is_toplevel
from oups.utils import files_at_depth
from oups.writer import write


# Column multi-index separator, available as a proposal.
CMIDX_SEP = '__'

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
    Sorted list of keys (indexes to parquet datasets).
    """
    def __init__(self, basepath:str, indexer:Type[dataclass]):
        """
        Parameters
        ----------
        basepath : str
            Path of directory containing parquet datasets.
        indexer : Type[dataclass]
            Class (not class instance) of the indexer to be used for:
            - identifying existing parquet datasets in 'basepath' directory,
            - creating the folders where recording new parquet datasets.

        Attributes
        ----------
        basepath : str
            Directory path to the set of parquet datasets.
        indexer : Type[dataclass]
            Indexer schema (class) to be used to index parquet datasets.
        keys : SortedSet
            Set of indexes of existing parquet datasets.

        Notes
        -----
        `SortedSet` is the data structure retained for `keys` instead of
        `SortedList` as `__contains__` appears faster.
        """
        if not is_toplevel(indexer):
            raise TypeError(f'{indexer.__name__} has to be "@toplevel"\
 decorated.')
        self._basepath = basepath
        self._indexer = indexer
        self._keys = get_keys(basepath, indexer)

    @property
    def basepath(self):
        return self._basepath
    
    @property
    def indexer(self):
        return self._indexer

    @property
    def keys(self):
        return self._keys

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return '\n'.join(map(str,self._keys))

    def __contains__(self, key):
        return key in self._keys
    
    def set(self, key:dataclass, data:Union[pDataFrame, vDataFrame], **kwargs):
        """Write 'data' to disk, at location defined by 'key'.

        Parameters
        ----------
        key : dataclass
            Key specifying the location where to write the data. It has to be
            an instance of the dataclass provided at ParquetSet instanciation.
        data : Union[pandas.DataFrame, vaex.dataframe.DataFrame]
            A dataframe, either in pandas format, or in vaex format.

        Other Parameters
        ----------------
        **kwargs : dict
            Keywords in 'kwargs' are forwarded to `writer.write`.
        """
        if not isinstance(key, self._indexer):
            raise TypeError(f'{key} is not an instance of \
{self._indexer.__name__}.')
        if (not isinstance(data, pDataFrame)
            and not isinstance(data, vDataFrame)):
                raise TypeError('Data should be a pandas or vaex dataframe.')
        dirpath = os_path.join(self._basepath, key.to_path)
        write(dirpath=dirpath, data=data, **kwargs)
        # If no trouble from writing, add key.
        self._keys.add(key)
        return
        
    def __setitem__(self, key, data):
        """Alias for `set`.

        Parameters
        ----------
        key : dataclass
            Key specifying the location where to write the data. It has to be
            an instance of the dataclass provided at ParquetSet instanciation.
        data : Union[Tuple[dict, Union[pDataFrame, vDataFrame]],
                      Union[pDataFrame, vDataFrame]]
            If a ``tuple``, first element is a ``dict`` containing parameter
            setting for `writer.write`, and second element is a dataframe
            (pandas or vaex).
            Or can be directly  a dataframe (pandas or vaex).
        """
        if isinstance(data, tuple):
            kwargs, data = data
            if not isinstance(kwargs, dict):
                raise TypeError(f'First item {kwargs} should be a dict to '
                                 'define parameter setting for '
                                 '`writer.write`.')
        else:
            kwargs = {}
        self.set(key, data, **kwargs)


# inner namespace
# https://stackoverflow.com/questions/45663249/namespaces-inside-class-in-python3


#    def __getitem__(self, key):
#        return self.store[self._keytransform(key)]

#    def __delitem__(self, key):
#        del self.store[self._keytransform(key)]
# or __del__ ?

#    def __iter__(self):
#        return iter(self.store)
    
#    def __len__(self):
#        return len(self.store)

#    def _keytransform(self, key):
#        return key
