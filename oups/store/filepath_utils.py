#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh

"""
from pathlib import Path
from typing import Iterator, List, Tuple


def files_at_depth(basepath: Path, depth: int = 2) -> Iterator[Tuple[Path, List[str]]]:
    """
    Yield file list in dirs.

    Generator yielding a tuple which:
        - 1st value is the path of a non-empty directory at 'depth' sublevel,
          counting from 'basepath'.
        - 2nd value is the list of files in this directory.

    Parameters
    ----------
    basepath : Path
        Path to directory from which scanning.
    depth : int, default 2
        Number of levels for directories to be retained (includes top level).
        By default, at least 2 levels.

    Yields
    ------
    Iterator[Tuple[Path,List[str]]]
        List of files within directory specified by the key. Empty directories
        are not returned.

    """
    if not basepath.exists():
        return
    if depth == 0:
        files = [entry.name for entry in basepath.iterdir() if not entry.is_dir()]
        if files:
            yield basepath, files
    if depth > 0:
        try:
            dirs = [entry for entry in basepath.iterdir() if entry.is_dir()]
        except FileNotFoundError:
            # If directory not existing, return `None`
            return
        depth -= 1
        for path in dirs:
            yield from files_at_depth(path, depth)


def remove_dir(path: Path):
    """
    Remove directory and all its contents.

    Parameters
    ----------
    path : Path
        Path to directory to be removed.

    """
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for p in path.iterdir():
        remove_dir(p)
    path.rmdir()
