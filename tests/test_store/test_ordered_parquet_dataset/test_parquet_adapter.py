#!/usr/bin/env python3
"""
Created on Sun May 18 15:00:00 2025.

@author: yoh

"""
import pytest
from fastparquet import write
from pandas import DataFrame
from pandas import MultiIndex

from oups.store.ordered_parquet_dataset import OrderedParquetDataset
from oups.store.ordered_parquet_dataset import conform_cmidx


def test_exception_check_cmidx(tmp_path):
    tmp_path = str(tmp_path)
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(ValueError, match="^not possible to have level name"):
        OrderedParquetDataset(tmp_path, ordered_on="a")._write_row_group_files([df])
    # Check with one column name not being a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(TypeError, match="^name 1 has to be"):
        OrderedParquetDataset(tmp_path, ordered_on="a")._write_row_group_files([df])


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
