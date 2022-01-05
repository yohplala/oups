#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh
"""
import zipfile
from os import path as os_path

from pandas import DataFrame as pDataFrame
from pandas import date_range
from vaex.dataframe import DataFrame as vDataFrame

from oups.router import ParquetHandle

from . import TEST_DATA


df_ref = pDataFrame(
    {
        "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2H"),
        "temperature": [8.4, 5.3, 4.9, 2.3],
    }
)


def test_parquet_file(tmp_path):
    # Read as parquet file.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = ParquetHandle(tmp_path)
    pf = ph.pf
    assert len(pf.row_groups) == 2
    assert sorted(df_ref.columns) == sorted(pf.columns)


def test_pandas_dataframe(tmp_path):
    # Read as pandas dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = ParquetHandle(tmp_path)
    pdf = ph.pdf
    assert isinstance(pdf, pDataFrame)
    assert pdf.equals(df_ref)


def test_vaex_dataframe(tmp_path):
    # Read as vaex dataframe.
    fn = os_path.join(TEST_DATA, "df_ts_temp_4rows_2rgs.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    ph = ParquetHandle(tmp_path)
    vdf = ph.vdf
    assert isinstance(vdf, vDataFrame)
    assert vdf.to_pandas_df().equals(df_ref)