#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
import zipfile
from os import path as os_path

import pytest
from fastparquet import write
from pandas import DataFrame
from pandas import MultiIndex

from oups.defines import DIR_SEP
from oups.store.utils import conform_cmidx
from oups.store.utils import files_at_depth

from .. import TEST_DATA


def test_files_at_depth(tmp_path):
    fn = os_path.join(TEST_DATA, "dummy_store.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    basepath = os_path.join(tmp_path, "store")
    # Test with 'depth=2'.
    depth = 2
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    paths_ref2 = [
        (f"stockholm.pressure{DIR_SEP}flemings.spring", ["innerplace.morning_opdmd"]),
    ]
    assert paths_files == paths_ref2
    # Test with 'depth=2'.
    depth = 1
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    paths_ref1 = [
        ("london.temperature", ["greenwich.summer_opdmd", "westminster.winter_dummy"]),
        ("paris.temperature", ["bastille.summer_opdmd"]),
        ("stockholm.pressure", ["skansen.fall_opdmd"]),
    ]
    assert paths_files == paths_ref1
    # Test with 'depth=3'.
    depth = 3
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    assert paths_files == []


def test_conform_cmidx(tmp_path):
    # Check first that fastparquet is unable to write a DataFrame with
    # identified shortcomings, then confirm it works with 'conform_cmidx()'.
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(TypeError, match="^Column name must be a string"):
        write(tmp_path, df, file_scheme="hive")
    # Check then one column name is not a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(ValueError, match="^\\('Column names must be multi-index,"):
        write(tmp_path, df, file_scheme="hive")
    df = DataFrame({("a", 1, "o"): [1]})
    df.columns.set_names(["oh"], level=[1], inplace=True)
    # Conform cmidx.
    conform_cmidx(df)
    assert df.columns == MultiIndex.from_tuples([("a", "1", "o")], names=["", "oh", ""])
