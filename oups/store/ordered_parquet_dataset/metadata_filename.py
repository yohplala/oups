#!/usr/bin/env python3
"""
Created on Sat May 24 18:00:00 2025.

@author: yoh

"""
from pathlib import Path
from typing import Union


OPDMD_EXTENSION = "_opdmd"


def get_md_filepath(dirpath: Path) -> Path:
    """
    Get standardized opd metadata file path.

    Parameters
    ----------
    dirpath : Path
        The directory path to use in the file path.

    Returns
    -------
    Path
        The formatted file name.

    """
    return dirpath.parent / (dirpath.name + OPDMD_EXTENSION)


def get_md_basename(filepath: Union[str, Path]) -> str:
    """
    Get the basename of the opd metadata file.

    Parameters
    ----------
    filepath : Union[str, Path]
        The file path from which extract the basename.

    Returns
    -------
    str
        The formatted file basename if file extension is present, None
        otherwise.

    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    return (
        filepath.name[: -len(OPDMD_EXTENSION)] if filepath.name.endswith(OPDMD_EXTENSION) else None
    )
