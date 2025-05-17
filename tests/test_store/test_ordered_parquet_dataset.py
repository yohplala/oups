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
from numpy import iinfo
from numpy import int8
from pandas import DataFrame
from pandas import date_range
from vaex.dataframe import DataFrame as vDataFrame

from oups.defines import KEY_ORDERED_ON
from oups.store.indexer import toplevel
from oups.store.ordered_parquet_dataset import FILE_IDS
from oups.store.ordered_parquet_dataset import N_ROWS
from oups.store.ordered_parquet_dataset import ORDERED_ON_MAXS
from oups.store.ordered_parquet_dataset import ORDERED_ON_MINS
from oups.store.ordered_parquet_dataset import RGS_STATS_BASE_DTYPES
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


def test_opd_init_empty(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="a")
    assert opd.dirpath == tmp_path
    assert opd.ordered_on == "a"
    assert opd.row_group_stats.empty
    assert opd.kvm == {KEY_ORDERED_ON: "a"}


def test_opd_write_metadata(tmp_path):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="a")
    opd1.write_metadata(metadata={"a": "b"})
    metadata_ref = {KEY_ORDERED_ON: "a", "a": "b"}
    assert opd1.row_group_stats.empty
    assert opd1.kvm == metadata_ref
    opd2 = OrderedParquetDataset2(tmp_path)
    assert opd2.row_group_stats.empty
    assert opd2.kvm == metadata_ref

    # TODO: test with some binary data in metadata


def test_opd_write_row_group_files(tmp_path):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    opd1.write_row_group_files([df_ref.iloc[:2], df_ref.iloc[2:]], write_opdmd=False)
    rgs_stats_ref = DataFrame(
        {
            ORDERED_ON_MINS: [
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[0]),
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[2]),
            ],
            ORDERED_ON_MAXS: [
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[1]),
                np.datetime64(df_ref.loc[:, "timestamp"].iloc[3]),
            ],
            N_ROWS: [2, 2],
            FILE_IDS: [0, 1],
        },
    ).astype(RGS_STATS_BASE_DTYPES)
    assert opd1.row_group_stats.equals(rgs_stats_ref)

    # TODO: modify with writing and checking metadata


def test_exception_opd_write_row_group_files_max_file_id_reached(tmp_path, monkeypatch):
    """
    Test error when reaching maximum file ID.
    """
    # Modify RGS_STATS_BASE_DTYPES to use int8 for FILE_IDS which has a lower max value
    int8_type = int8
    exceeding_max_n_files = iinfo(int8_type).max + 2
    # Patch the RGS_STATS_BASE_DTYPES dictionary
    monkeypatch.setitem(RGS_STATS_BASE_DTYPES, FILE_IDS, int8_type)

    opd = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    # Create iterable of dataframes.
    large_df = DataFrame(
        {
            "timestamp": date_range("2021/01/01", periods=exceeding_max_n_files, freq="1min"),
            "value": range(exceeding_max_n_files),
        },
    )

    def dataframes():
        for _, new_row in large_df.iterrows():
            yield DataFrame([new_row.to_list()], columns=new_row.index)

    dataframes = list(dataframes())
    # Write max_file_id dataframes.
    opd.write_row_group_files(dataframes[: exceeding_max_n_files - 2], write_opdmd=True)

    opd_tmp = OrderedParquetDataset2(tmp_path)
    assert opd_tmp.row_group_stats.loc[:, FILE_IDS].iloc[-1] == exceeding_max_n_files - 3

    # Try to write one more.
    max_n_files = exceeding_max_n_files - 1
    with pytest.raises(
        ValueError,
        match=f"^file id {max_n_files} exceeds max value {max_n_files-1}",
    ):
        opd.write_row_group_files(dataframes[max_n_files:], write_opdmd=False)

    opd_tmp = OrderedParquetDataset2(tmp_path)
    # Check that the opmd file has been correctly rewritten.
    assert opd_tmp.row_group_stats.loc[:, FILE_IDS].iloc[-1] == exceeding_max_n_files - 2


def test_exception_opd_write_row_group_files_max_n_rows_reached(tmp_path, monkeypatch):
    """
    Test error when reaching maximum number of rows.
    """
    # Modify RGS_STATS_BASE_DTYPES to use int8 for N_ROWS which has a lower max value
    int8_type = int8
    exceeding_max_n_rows = iinfo(int8_type).max + 1
    # Patch the RGS_STATS_BASE_DTYPES dictionary
    monkeypatch.setitem(RGS_STATS_BASE_DTYPES, N_ROWS, int8_type)

    opd = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    # Create a dataframe with more rows than the max
    large_df = DataFrame(
        {
            "timestamp": date_range("2021/01/01", periods=exceeding_max_n_rows, freq="1min"),
            "temperature": [20.0] * (exceeding_max_n_rows),
        },
    )

    # Try to write the large dataframe (this should fail)
    with pytest.raises(
        ValueError,
        match=f"^number of rows {exceeding_max_n_rows} exceeds max value {exceeding_max_n_rows-1}",
    ):
        opd.write_row_group_files([large_df])
