#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh

Test utils.
TEST_DATA = 'test-data'
tmp_path = os_path.expanduser('~/Documents/code/data/oups')

"""
import zipfile
from os import path as os_path

import numpy as np
import pytest
from fastparquet import ParquetFile
from pandas import DataFrame
from pandas import date_range
from vaex.dataframe import DataFrame as vDataFrame

from oups.store.indexer import toplevel
from oups.store.ordered_parquet_dataset import OrderedParquetDataset

from .. import TEST_DATA


df_ref = DataFrame(
    {
        "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
        "temperature": [8.4, 5.3, 4.9, 2.3],
    },
)


@toplevel
class Indexer:
    loc: str


key_ref = Indexer("ah")


def test_parquet_file(tmp_path):
    # Read as parquet file.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp")
    pf = ph.pf
    assert len(pf.row_groups) == 2
    assert sorted(df_ref.columns) == sorted(pf.columns)


def test_pandas_dataframe(tmp_path):
    # Read as pandas dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp")
    pdf = ph.pdf
    assert isinstance(pdf, DataFrame)
    assert pdf.equals(df_ref)


def test_vaex_dataframe(tmp_path):
    # Read as vaex dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp")
    vdf = ph.vdf
    assert isinstance(vdf, vDataFrame)
    assert vdf.to_pandas_df().equals(df_ref)


def test_write(tmp_path):
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp", df_like=df_ref)
    ph.write(df=df_ref)
    assert ph.to_pandas().equals(df_ref)


def test_min_max(tmp_path):
    # Read as parquet file.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    col = "timestamp"
    min_max = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp").min_max(col)
    min_ref = np.datetime64(df_ref[col].min())
    max_ref = np.datetime64(df_ref[col].max())
    assert min_max == (min_ref, max_ref)
    col = "temperature"
    min_max = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp").min_max(col)
    min_ref = df_ref[col].min()
    max_ref = df_ref[col].max()
    assert min_max == (min_ref, max_ref)


def test_parquet_handle_not_existing(tmp_path):
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp")
    assert isinstance(ph, ParquetFile)
    assert ph.pdf.empty


def test_parquet_handle_folder_not_existing(tmp_path):
    tmp_path = os_path.join(tmp_path, "test")
    ph = OrderedParquetDataset(str(tmp_path), ordered_on="timestamp")
    assert isinstance(ph, ParquetFile)
    assert ph.pdf.empty


def test_exception_check_cmidx(tmp_path):
    tmp_path = str(tmp_path)
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(ValueError, match="^not possible to have level name"):
        OrderedParquetDataset(tmp_path, ordered_on="a", df_like=df)
    # Check with one column name not being a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(TypeError, match="^name 1 has to be"):
        OrderedParquetDataset(tmp_path, ordered_on="a", df_like=df)


def test_exception_ordered_on_write(tmp_path):
    tmp_path = str(tmp_path)
    df = DataFrame({"a": [1], "b": [2]})
    opd = OrderedParquetDataset(tmp_path, ordered_on="a", df_like=df)
    with pytest.raises(ValueError, match="^'ordered_on' attribute a is not "):
        opd.write(df=df, ordered_on="b")
