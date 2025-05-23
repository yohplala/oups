#!/usr/bin/env python3
"""
Created on Sun May 18 15:00:00 2025.

@author: yoh

"""
import pytest
from pandas import DataFrame

from oups.store.ordered_parquet_dataset import OrderedParquetDataset


def test_exception_check_cmidx(tmp_path):
    tmp_path = str(tmp_path)
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(ValueError, match="^not possible to have level name"):
        OrderedParquetDataset(tmp_path, ordered_on="a").write_row_group_files([df])
    # Check with one column name not being a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(TypeError, match="^name 1 has to be"):
        OrderedParquetDataset(tmp_path, ordered_on="a").write_row_group_files([df])
