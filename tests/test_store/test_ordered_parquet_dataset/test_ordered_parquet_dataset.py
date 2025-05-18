#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh

"""
import pytest
from numpy import iinfo
from numpy import int8
from pandas import DataFrame
from pandas import Timestamp
from pandas import date_range

from oups.defines import KEY_ORDERED_ON
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import FILE_IDS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import N_ROWS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import ORDERED_ON_MAXS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import ORDERED_ON_MINS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import RGS_STATS_BASE_DTYPES
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import OrderedParquetDataset2


df_ref = DataFrame(
    {
        "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
        "temperature": [8.4, 5.3, 4.9, 2.3],
    },
)


def test_opd_init_empty(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="a")
    assert opd.dirpath == tmp_path
    assert opd.ordered_on == "a"
    assert opd.row_group_stats.empty
    assert opd.kvm == {KEY_ORDERED_ON: "a"}


@pytest.mark.parametrize(
    "ordered_on, err_msg",
    [
        (None, "'ordered_on' column name must be provided."),
        ("b", "^'ordered_on' parameter value 'b' does not match"),
    ],
)
def test_exception_opd_init_ordered_on(tmp_path, ordered_on, err_msg):
    if ordered_on:
        # Write a 1st dataset with a different 'ordered_on' column name.
        opd = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
        opd.write_row_group_files([df_ref], write_opdmd=True)
    with pytest.raises(
        ValueError,
        match=err_msg,
    ):
        opd = OrderedParquetDataset2(tmp_path, ordered_on=ordered_on)


def test_opd_write_metadata(tmp_path):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="a")
    additional_metadata_in = {"a": "b", "ts": Timestamp("2021-01-01")}
    opd1.write_metadata(metadata=additional_metadata_in)
    metadata_ref = {KEY_ORDERED_ON: "a", **additional_metadata_in}
    assert opd1.row_group_stats.empty
    assert opd1.kvm == metadata_ref
    opd2 = OrderedParquetDataset2(tmp_path)
    assert opd2.row_group_stats.empty
    assert opd2.kvm == metadata_ref
    # Changing some metadata values, removing another one.
    additional_metadata_in = {"a": "c", "ts": None}
    opd1.write_metadata(metadata=additional_metadata_in)
    metadata_ref = {KEY_ORDERED_ON: "a", "a": "c"}
    assert opd1.kvm == metadata_ref
    opd2 = OrderedParquetDataset2(tmp_path)
    assert opd2.kvm == metadata_ref


@pytest.mark.parametrize("write_opdmd", [False, True])
def test_opd_write_row_group_files(tmp_path, write_opdmd):
    opd1 = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    opd1.write_row_group_files([df_ref.iloc[:2], df_ref.iloc[2:]], write_opdmd=write_opdmd)
    rgs_stats_ref = DataFrame(
        {
            FILE_IDS: [0, 1],
            N_ROWS: [2, 2],
            ORDERED_ON_MINS: [
                df_ref.loc[:, "timestamp"].iloc[0],
                df_ref.loc[:, "timestamp"].iloc[2],
            ],
            ORDERED_ON_MAXS: [
                df_ref.loc[:, "timestamp"].iloc[1],
                df_ref.loc[:, "timestamp"].iloc[3],
            ],
        },
    ).astype(RGS_STATS_BASE_DTYPES)
    assert opd1.row_group_stats.equals(rgs_stats_ref)
    if write_opdmd:
        opd2 = OrderedParquetDataset2(tmp_path)
        assert opd2.row_group_stats.equals(rgs_stats_ref)


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
    max_file_id = exceeding_max_n_files - 2
    # Write max_file_id dataframes.
    opd.write_row_group_files(dataframes[:max_file_id], write_opdmd=True)

    opd_tmp = OrderedParquetDataset2(tmp_path)
    assert opd_tmp.row_group_stats.loc[:, FILE_IDS].iloc[-1] == max_file_id - 1

    # Try to write one more.
    with pytest.raises(
        ValueError,
        match=f"^file id '{max_file_id+1}' exceeds max value {max_file_id}",
    ):
        opd.write_row_group_files(dataframes[max_file_id:], write_opdmd=False)

    opd_tmp = OrderedParquetDataset2(tmp_path)
    # Check that the opmd file has been correctly rewritten.
    assert opd_tmp.row_group_stats.loc[:, FILE_IDS].iloc[-1] == max_file_id


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


def test_exception_opd_write_row_group_files_ordered_on(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="a")
    with pytest.raises(
        ValueError,
        match="^'ordered_on' column 'a' is not in",
    ):
        opd.write_row_group_files([df_ref])


def test_opd_getitem_and_len(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    assert len(opd) == len(df_ref)
    assert not opd.forbidden_to_write_row_group_files
    opd_sub1 = opd[1]
    # Using slice notation to preserve DataFrame format.
    assert opd_sub1.row_group_stats.equals(opd.row_group_stats.iloc[1:2])
    assert opd_sub1.kvm == opd.kvm
    assert opd_sub1.ordered_on == opd.ordered_on
    assert opd_sub1.__dict__.keys() == opd.__dict__.keys()
    assert len(opd_sub1) == 1
    assert opd_sub1.forbidden_to_write_row_group_files
    opd_sub2 = opd[1:3]
    assert opd_sub2.row_group_stats.equals(opd.row_group_stats.iloc[1:3])
    assert opd_sub2.kvm == opd.kvm
    assert opd_sub2.ordered_on == opd.ordered_on
    assert opd_sub2.__dict__.keys() == opd.__dict__.keys()
    assert len(opd_sub2) == 2


def test_opd_remove_row_group_files(tmp_path):
    opd = OrderedParquetDataset2(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    # Keep ref before removing.
    rg_stats_ref = opd.row_group_stats.iloc[[1, 3]].reset_index(drop=True)
    assert not opd.forbidden_to_remove_row_group_files
    opd.remove_row_group_files([0, 2])
    assert len(opd) == 2
    assert opd.row_group_stats.equals(rg_stats_ref)
    assert opd.forbidden_to_remove_row_group_files
