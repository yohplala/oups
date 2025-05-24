#!/usr/bin/env python3
"""
Created on Wed Dec  4 18:00:00 2021.

@author: yoh

"""
from dataclasses import dataclass
from os import listdir
from os import rmdir
from shutil import rmtree
from typing import Type

from sortedcontainers import SortedSet

from oups.defines import DIR_SEP
from oups.defines import OPDMD_EXTENSION
from oups.store.indexer import is_toplevel
from oups.store.ordered_parquet_dataset import OrderedParquetDataset
from oups.store.store.utils import files_at_depth
from oups.store.store.utils import strip_path_tail


LEN_OPDMD_EXTENSION = len(OPDMD_EXTENSION)


def get_opd_basepath(store_path: str, key: dataclass) -> str:
    """
    Return path to 'opd' base directory corresponding to given key.

    Parameters
    ----------
    store_path : str
        Path to store directory.
    key : dataclass
        Key specifying the location where to read the data from. It has to
        be an instance of the dataclass provided at Store instantiation.

    """
    return f"{store_path}{DIR_SEP}{key.to_path}"


def get_keys(basepath: str, indexer: Type[dataclass]) -> SortedSet:
    """
    Identify ordered parquet dataset in directory.

    Scan 'basepath' directory and create instances of 'indexer' class from
    compatible subpaths. Only file which name ends by '_opdmd' are retained
    to construct a key.

    Parameters
    ----------
    basepath : str
        Path to directory containing a dataset collection, in folders complying
        with the schema defined by 'indexer' dataclass.
    indexer : Type[dataclass]
        Class decorated with '@toplevel' decorator, and defining a path
        schema.

    Returns
    -------
    SortedSet[dataclass]
        Sorted set of keys (i.e. instances of indexer dataclass) that can be
        found in 'basepath' directory and with a 'valid' opd metadata file
        (ending by '_opdmd').

    """
    depth = indexer.depth - 1
    # Filter, keeping only folders having files with correct extension,
    # then materialize paths into keys, filtering out those that can't.
    return SortedSet(
        [
            key
            for path, files in files_at_depth(basepath, depth)
            for file in files
            if (
                file.endswith(OPDMD_EXTENSION)
                and (
                    key := indexer.from_path(
                        DIR_SEP.join(
                            path.rsplit(DIR_SEP, depth)[1:] + [file[:-LEN_OPDMD_EXTENSION]],
                        ),
                    )
                )
            )
        ],
    )


class Store:
    """
    Sorted list of keys (indexes to parquet datasets).

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
    ``SortedSet`` is the data structure retained for ``keys`` instead of
    ``SortedList`` as its ``__contains__`` appears faster.

    """

    def __init__(self, basepath: str, indexer: Type[dataclass]):
        """
        Instantiate parquet set.

        Parameters
        ----------
        basepath : str
            Path of directory containing parquet datasets.
        indexer : Type[dataclass]
            Class (not class instance) of the indexer to be used for:

              - identifying existing parquet datasets in 'basepath' directory,
              - creating the folders where recording new parquet datasets.

        """
        if not is_toplevel(indexer):
            raise TypeError(f"{indexer.__name__} has to be '@toplevel' decorated.")
        self._basepath = basepath
        self._indexer = indexer
        self._keys = get_keys(basepath, indexer)
        self._has_initialized_a_new_opd = False

    @property
    def basepath(self):
        """
        Return basepath.
        """
        return self._basepath

    @property
    def indexer(self):
        """
        Return indexer.
        """
        return self._indexer

    @property
    def keys(self):
        """
        Return keys.
        """
        if self._has_initialized_a_new_opd:
            # Refresh keys.
            self._keys = get_keys(self.basepath, self.indexer)
            self._has_initialized_a_new_opd = False
        return self._keys

    def __len__(self):
        """
        Return number of datasets.
        """
        return len(self.keys)

    def __repr__(self):
        """
        List of datasets.
        """
        return "\n".join(map(str, self.keys))

    def __contains__(self, key):
        """
        Assess presence of this dataset.

        Parameters
        ----------
        key : dataclass
            Key to assess presence of.

        """
        return key in self.keys

    def __iter__(self):
        """
        Iterate over keys.
        """
        yield from self.keys

    def __getitem__(self, key: dataclass):
        """
        Return the ``OrderedParquetDataset`` instance corresponding to ``key``.

        Parameters
        ----------
        key : dataclass
            Key specifying the location where to read the data from. It has to
            be an instance of the dataclass provided at Store instantiation.

        """
        opd = OrderedParquetDataset(get_opd_basepath(self._basepath, key))
        if opd.is_opdmd_file_missing:
            self._has_initialized_a_new_opd = True
        return opd

    def __delitem__(self, key: dataclass):
        """
        Remove dataset from parquet set.

        Parameter
        ---------
        key : dataclass
            Key specifying the location where to delete the data. It has to be
            an instance of the dataclass provided at Store instantiation.

        """
        # TODO: remove opdmd file.
        # TODO: check if rmtree raise error if directory does not exist (only metadata)
        if key in self.keys:
            # Keep track of intermediate partition folders, in case one get
            # empty.
            basepath = self.basepath
            dirpath = f"{basepath}{DIR_SEP}{key.to_path}"
            rmtree(dirpath)
            self._keys.remove(key)
            # Remove possibly empty directories.
            upper_dir = strip_path_tail(dirpath)
            while (upper_dir != basepath) and (not listdir(upper_dir)):
                rmdir(upper_dir)
                upper_dir = strip_path_tail(upper_dir)
