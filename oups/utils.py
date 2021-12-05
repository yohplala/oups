#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:30:00 2021
@author: yoh
"""
from os import scandir
from typing import Iterator, List, Tuple

from oups.defines import DIR_SEP


def files_at_depth(basepath:str, depth:int=1)\
    -> Iterator[Tuple[str,List[str]]]:
    """
    Generator yielding a tuple which:
        - 1st value is the path of a non-empty directory at 'depth' sublevel,
          counting from 'basepath'.
        - 2nd value is the list of files in this directory.

    Parameters
    basepath : str
        Path to directory from which scanning.
    depth : int, default 1
        Number of sublevels for directories to be retained.
        By default, at least 1 sublevel.

    Yields
    Iterator[Dict[str,List[str]]]
        List of files within directory specified by the key. Empty directories
        are not returned.
    """
    if depth == 0:
        files = [entry.path.rsplit(DIR_SEP,1)[1] for entry in scandir(basepath)
                 if not entry.is_dir(follow_symlinks=False)]
        if files:
            yield basepath, files
    if depth > 0:
        dirs = [entry.path for entry in scandir(basepath)
                if entry.is_dir(follow_symlinks=False)]
        depth -= 1
        for path in dirs:
            yield from files_at_depth(path, depth)
