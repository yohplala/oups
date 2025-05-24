#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
from os import path as os_path


# Directory separator.
DIR_SEP = os_path.sep
# Central key in oups world, used as ID of the column name according which
# dataframes are ordered.
KEY_ORDERED_ON = "ordered_on"
# Other shared keys.
KEY_FILE_IDS = "file_ids"
KEY_N_ROWS = "n_rows"
KEY_ORDERED_ON_MINS = "ordered_on_mins"
KEY_ORDERED_ON_MAXS = "ordered_on_maxs"
# A specific key for a function parameter with a three-fold type:
# - None, meaning no duplicates check,
# - an empty list, meaning identify duplicate on all columns of the dataframe,
# - a not empty list of a string, the columns to identify row duplicates.
KEY_DUPLICATES_ON = "duplicates_on"
# In a fastparquet `ParquetFile`, oups-specific metadata is stored as value for
# key `KEY_METADATA_KEY`.
KEY_OUPS_METADATA = "oups"
