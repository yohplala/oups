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
from pandas import DataFrame as pDataFrame
from pandas import date_range
from vaex.dataframe import DataFrame as vDataFrame

from oups.store.indexer import toplevel
from oups.store.router import ParquetHandle

from .. import TEST_DATA


df_ref = pDataFrame(
    {
        "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2H"),
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
    ph = ParquetHandle(str(tmp_path))
    pf = ph.pf
    assert len(pf.row_groups) == 2
    assert sorted(df_ref.columns) == sorted(pf.columns)


def test_pandas_dataframe(tmp_path):
    # Read as pandas dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = ParquetHandle(str(tmp_path))
    pdf = ph.pdf
    assert isinstance(pdf, pDataFrame)
    assert pdf.equals(df_ref)


def test_vaex_dataframe(tmp_path):
    # Read as vaex dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = ParquetHandle(str(tmp_path))
    vdf = ph.vdf
    assert isinstance(vdf, vDataFrame)
    assert vdf.to_pandas_df().equals(df_ref)


def test_min_max(tmp_path):
    # Read as parquet file.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    col = "timestamp"
    min_max = ParquetHandle(str(tmp_path)).min_max(col)
    min_ref = np.datetime64(df_ref[col].min())
    max_ref = np.datetime64(df_ref[col].max())
    assert min_max == (min_ref, max_ref)
    col = "temperature"
    min_max = ParquetHandle(str(tmp_path)).min_max(col)
    min_ref = df_ref[col].min()
    max_ref = df_ref[col].max()
    assert min_max == (min_ref, max_ref)
