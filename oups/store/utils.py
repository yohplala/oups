#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh

"""
from os import scandir
from typing import Iterator, List, Tuple

from pandas import DataFrame

from oups.defines import DIR_SEP


def files_at_depth(basepath: str, depth: int = 2) -> Iterator[Tuple[str, List[str]]]:
    """
    Yield file list in dirs.

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
    """
    Remove last level directory from provided path.

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


def conform_cmidx(df: DataFrame):
    """
    Conform pandas column multi-index.

    Library fastparquet has several requirements to handle column MultiIndex.

      - It requires names for each level in a Multiindex. If these are not set,
        there are set to '', an empty string.
      - It requires column names to be tuple of string. If an object is
        different than a string (for instance float or int), it is turned into
        a string.

    DataFrame is modified in-place.

    Parameters
    ----------
    df : DataFrame
        DataFrame with a column multi-index to check and possibly adjust.

    Returns
    -------
    None

    """
    # If a name is 'None', set it to '' instead.
    cmidx = df.columns
    if None in cmidx.names:
        level_updated_idx = [i for i, name in enumerate(cmidx.names) if name is None]
        cmidx.set_names([""] * len(level_updated_idx), level=level_updated_idx, inplace=True)
    # If an item of the column name is not a string, turn it into a string.
    # Using 'set_levels()' instead of rconstructing a MultiIndex to preserve
    # index names directly.
    level_updated_idx = []
    level_updated = []
    for i, level in enumerate(cmidx.levels):
        str_level = [name if isinstance(name, str) else str(name) for name in level]
        if level.to_list() != str_level:
            level_updated_idx.append(i)
            level_updated.append(str_level)
    if level_updated:
        df.columns = df.columns.set_levels(level_updated, level=level_updated_idx)
