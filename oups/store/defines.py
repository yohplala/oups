#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
from os import path as os_path


# Directory separator.
DIR_SEP = os_path.sep

# In a fastparquet `ParquetFile`, oups-specific metadata is stored as value for
# key `OUPS_METADATA_KEY`.
OUPS_METADATA_KEY = "oups"
