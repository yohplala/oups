#!/usr/bin/env python3
"""
Created on Sat May 24 18:00:00 2025.

@author: yoh

"""


from oups.defines import OPDMD_EXTENSION


def get_opdmd_filepath(dirpath: str) -> str:
    """
    Get standardized opdmd file path.

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
