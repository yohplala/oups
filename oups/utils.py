#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from os import scandir
from typing import Iterator, List, Tuple

from pandas import Grouper
from pandas import Series
from pandas import cut
from pandas import date_range
from pandas.core.resample import _get_timestamp_range_edges as gtre

from oups.defines import DIR_SEP


def files_at_depth(basepath: str, depth: int = 2) -> Iterator[Tuple[str, List[str]]]:
    """Yield file list in dirs.

    Generator yielding a tuple which:
        - 1st value is the path of a non-empty directory at 'depth' sublevel,
          counting from 'basepath'.
        - 2nd value is the list of files in this directory.

    Parameters
    ----------
    basepath : str
        Path to directory from which scanning.
    depth : int, default 2
        Number of levels for directories to be retained (includes top level).
        By default, at least 2 levels.

    Yields
    ------
    Iterator[Dict[str,List[str]]]
        List of files within directory specified by the key. Empty directories
        are not returned.
    """
    if depth == 0:
        files = [
            entry.path.rsplit(DIR_SEP, 1)[1]
            for entry in scandir(basepath)
            if not entry.is_dir(follow_symlinks=False)
        ]
        if files:
            yield basepath, files
    if depth > 0:
        try:
            dirs = [
                entry.path for entry in scandir(basepath) if entry.is_dir(follow_symlinks=False)
            ]
        except FileNotFoundError:
            # If directory not existing, return `None`
            return
        depth -= 1
        for path in dirs:
            yield from files_at_depth(path, depth)


def strip_path_tail(dirpath: str) -> str:
    """Remove last level directory from provided path.

    Parameters
    ----------
    dirpath : str
        Directory path.

    Returns
    -------
    str
        Directory path stripped from last directory.
    """
    if DIR_SEP in dirpath:
        return dirpath.rsplit("/", 1)[0]


def tcut(data: Series, grouper: Grouper):
    """Perform binning with provided grouper.

    Achieve binning of data by dates as specified by provided pandas grouper.
    Rows falling in same period will get same label.
    On the opposite to labels that would be returned by a pandas `groupby`
    using provided grouper, labels returned by `dcut` systematically fall in
    the bin they belong to.

    Parameters
    ----------
    data : pandas.Series
        Series or array-like which values will be binned.
    grouper : pandas.Grouper
        Specifications according which defining the time bins.

    Returns
    -------
    pandas.Series
        Series with same length as initial array, with values as categories,
        which name fall into bins specified by the grouper.

    Notes
    -----
    - When performing a `groupby` with a grouper having attributes `closed` set
      to `right` and `labels` set to `left`, resulting labels do not fall in
      bins. Such a result cannot be achieved with `tcut`. Labels returned by
      `tcut` always fall in time bins. The motivation is that then, a `groupby`
      can be achieved in a 2nd step with same grouper to provide expected
      results.
    - To use results in a `groupby` operation while re-using provided grouper,
      labels (being categories), need to be materialized again as timestamps
      with ``astype('datetime64')``.
    """
    start, end = gtre(
        first=data.iloc[0],
        last=data.iloc[-1],
        freq=grouper.freq,
        closed=grouper.closed,
        origin=grouper.origin,
        offset=grouper.offset,
    )
    # Shifting end by one more period to generate all required bins.
    end += grouper.freq
    bins = date_range(start, end, freq=grouper.freq)
    as_binned = cut(data, bins, right=(grouper.closed == "right"))
    if grouper.closed == "right":
        # If closed on 'right', change label of bin so that the label fall in
        # expected bin.
        return as_binned.cat.rename_categories(as_binned.cat.categories.right)
    else:
        return as_binned.cat.rename_categories(as_binned.cat.categories.left)
