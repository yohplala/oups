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
from numpy import uint16
from numpy import uint32
from pandas import DataFrame
from pandas import date_range
from vaex.dataframe import DataFrame as vDataFrame

from oups.defines import KEY_ORDERED_ON
from oups.store.indexer import toplevel
from oups.store.ordered_parquet_dataset import N_ROWS
from oups.store.ordered_parquet_dataset import ORDERED_ON_MAX
from oups.store.ordered_parquet_dataset import ORDERED_ON_MIN
from oups.store.ordered_parquet_dataset import PART_ID
from oups.store.ordered_parquet_dataset import OrderedParquetDataset
from oups.store.ordered_parquet_dataset import OrderedParquetDataset2

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


def test_opd2_init_empty(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="a")
    assert opd.dirpath == tmp_path
    assert opd.ordered_on == "a"
    assert opd.rgs_stats.empty
    assert opd.kvm == {KEY_ORDERED_ON: "a"}


def test_opd2_write_metadata(tmp_path):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="a")
    opd1.write_metadata(metadata={"a": "b"})
    metadata_ref = {KEY_ORDERED_ON: "a", "a": "b"}
    assert opd1.rgs_stats.empty
    assert opd1.kvm == metadata_ref
    opd2 = OrderedParquetDataset2(tmp_path)
    assert opd2.rgs_stats.empty
    assert opd2.kvm == metadata_ref


def test_opd2_write_row_group_files(tmp_path):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    opd1.write_row_group_files([df_ref.iloc[:2], df_ref.iloc[2:]], write_opdmd=False)
    rgs_stats_ref = DataFrame(
        {
            ORDERED_ON_MIN: [
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[0]),
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[2]),
            ],
            ORDERED_ON_MAX: [
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[1]),
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[3]),
            ],
            N_ROWS: [2, 2],
            PART_ID: [1, 2],
        },
    ).astype({N_ROWS: uint32, PART_ID: uint16})
    assert opd1.rgs_stats.equals(rgs_stats_ref)
