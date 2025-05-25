#!/usr/bin/env python3
"""
Created on Sat May 24 18:00:00 2025.

@author: yoh

"""
from oups.defines import DIR_SEP


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


def get_md_basename(filepath: str) -> str:
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
    return (
        filepath.rsplit(DIR_SEP, 1)[-1][: -len(OPDMD_EXTENSION)]
        if filepath.endswith(OPDMD_EXTENSION)
        else None
    )
