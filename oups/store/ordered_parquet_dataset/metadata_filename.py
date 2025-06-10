#!/usr/bin/env python3
"""
Created on Sat May 24 18:00:00 2025.

@author: yoh

"""
from pathlib import Path
from typing import Union


OPDMD_EXTENSION = "_opdmd"


def get_md_filepath(dirpath: str) -> str:
    """
    Get standardized opd metadata file path.

    Parameters
    ----------
    dirpath : str
        The directory path to use in the file path.

    Returns
    -------
    str
        The formatted file name.

    """
    return f"{dirpath}{OPDMD_EXTENSION}"


def get_md_basename(filepath: Union[str, Path]) -> str:
    """
    Get the basename of the opd metadata file.

    Parameters
    ----------
    filepath : str
        The file path from which extract the basename.

    Returns
    -------
    str
        The formatted file basename if file extension is present, None
        otherwise.

    """
    filepath_name = filepath.name if isinstance(filepath, Path) else Path(filepath).name
    return (
        filepath_name[: -len(OPDMD_EXTENSION)] if filepath_name.endswith(OPDMD_EXTENSION) else None
    )
