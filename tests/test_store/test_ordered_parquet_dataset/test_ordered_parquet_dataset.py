#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh

"""
from os import rename

import pytest
from numpy import iinfo
from numpy import int8
from pandas import DataFrame
from pandas import Timestamp
from pandas import date_range

from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import KEY_FILE_IDS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import KEY_N_ROWS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import KEY_ORDERED_ON_MAXS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import KEY_ORDERED_ON_MINS
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import RGS_STATS_BASE_DTYPES
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import OrderedParquetDataset
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import get_parquet_filepaths


df_ref = DataFrame(
    {
        "timestamp": date_range(start="2021/01/01 08:00", periods=8, freq="2h"),
        "temperature": [8.4, 5.3, 4.9, 2.3, 1.0, 0.5, 0.2, 0.1],
    },
)


def switch_row_group_file_ids(opd, file_id_1, file_id_2):
    """
    Switch two row group file ids in row group filename and row group stats.

    This function does not change the row group order. It only switches the file
    ids.

    Parameters
    ----------
    opd : OrderedParquetDataset2
        The ordered parquet dataset.
    file_id_1 : int
        The first file id to switch.
    file_id_2 : int
        The second file id to switch with file_id_1.

    """
    tmp_id = len(df_ref)
    file_id_list = opd.row_group_stats.loc[:, KEY_FILE_IDS].to_list()
    file_ids_col_idx = opd.row_group_stats.columns.get_loc(KEY_FILE_IDS)
    file_id_1_row_idx = file_id_list.index(file_id_1)
    file_id_2_row_idx = file_id_list.index(file_id_2)
    rename(
        get_parquet_filepaths(opd.dirpath, file_id_1, opd._file_id_n_digits),
        get_parquet_filepaths(opd.dirpath, tmp_id, opd._file_id_n_digits),
    )
    rename(
        get_parquet_filepaths(opd.dirpath, file_id_2, opd._file_id_n_digits),
        get_parquet_filepaths(opd.dirpath, file_id_1, opd._file_id_n_digits),
    )
    rename(
        get_parquet_filepaths(opd.dirpath, tmp_id, opd._file_id_n_digits),
        get_parquet_filepaths(opd.dirpath, file_id_2, opd._file_id_n_digits),
    )
    tmp_file_id = opd.row_group_stats.iloc[file_id_1_row_idx, file_ids_col_idx]
    opd._row_group_stats.iloc[file_id_1_row_idx, file_ids_col_idx] = opd.row_group_stats.iloc[
        file_id_2_row_idx,
        file_ids_col_idx,
    ]
    opd._row_group_stats.iloc[file_id_2_row_idx, file_ids_col_idx] = tmp_file_id


def test_opd_init_empty(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="a")
    assert opd.dirpath == tmp_path
    assert opd.ordered_on == "a"
    assert opd.row_group_stats.empty
    assert opd.key_value_metadata == {}


def test_exception_opd_init_ordered_on(tmp_path):
    # Write a 1st dataset with a different 'ordered_on' column name.
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    opd.write_row_group_files([df_ref], write_opdmd=True)
    with pytest.raises(
        ValueError,
        match="^'ordered_on' parameter value 'b'",
    ):
        opd = OrderedParquetDataset(tmp_path, ordered_on="b")


def test_opd_getitem_and_len(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    assert len(opd) == len(df_ref)
    assert not opd._is_row_group_subset
    opd_sub1 = opd[1]
    # Using slice notation to preserve DataFrame format.
    assert opd_sub1.row_group_stats.equals(opd._row_group_stats.iloc[1:2])
    assert opd_sub1.key_value_metadata == opd.key_value_metadata
    assert opd_sub1.ordered_on == opd.ordered_on
    assert opd_sub1.__dict__.keys() == opd.__dict__.keys()
    assert len(opd_sub1) == 1
    assert opd_sub1._is_row_group_subset
    opd_sub2 = opd[1:3]
    assert opd_sub2.row_group_stats.equals(opd.row_group_stats.iloc[1:3])
    assert opd_sub2.key_value_metadata == opd.key_value_metadata
    assert opd_sub2.ordered_on == opd.ordered_on
    assert opd_sub2.__dict__.keys() == opd.__dict__.keys()
    assert len(opd_sub2) == 2


def test_opd_getitem_and_max_file_id(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    max_file_id_ref = len(df_ref) - 1
    assert opd.row_group_stats.loc[:, KEY_FILE_IDS].max() == max_file_id_ref
    assert opd.max_file_id == max_file_id_ref
    assert opd[2:4].row_group_stats.loc[:, KEY_FILE_IDS].max() == 3
    # Remove first row group.
    opd.remove_row_group_files(file_ids=[0])
    assert opd.max_file_id == max_file_id_ref
    # check 'max_file_id' is correctly computed on a subset.
    assert opd[2:4].max_file_id == max_file_id_ref


def test_opd_align_file_ids(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=False,
    )
    # Remove 2nd row group. It keeps one original id and requires to update the
    # 2 last ids.
    opd.remove_row_group_files(file_ids=[1])
    # Introduce two loops in file_ids.
    # 2 <-> 3.
    # 4 <-> 5.
    switch_row_group_file_ids(opd, 2, 4)
    switch_row_group_file_ids(opd, 4, 6)
    opd.align_file_ids()
    ordered_on_mins_maxs_ref = [
        df_ref.loc[:, "timestamp"].iloc[0],
        df_ref.loc[:, "timestamp"].iloc[2],
        df_ref.loc[:, "timestamp"].iloc[3],
        df_ref.loc[:, "timestamp"].iloc[4],
        df_ref.loc[:, "timestamp"].iloc[5],
        df_ref.loc[:, "timestamp"].iloc[6],
        df_ref.loc[:, "timestamp"].iloc[7],
    ]
    rg_stats_ref = DataFrame(
        {
            KEY_FILE_IDS: [0, 1, 2, 3, 4, 5, 6],
            KEY_N_ROWS: [1, 1, 1, 1, 1, 1, 1],
            KEY_ORDERED_ON_MINS: ordered_on_mins_maxs_ref,
            KEY_ORDERED_ON_MAXS: ordered_on_mins_maxs_ref,
        },
    ).astype(RGS_STATS_BASE_DTYPES)
    assert opd.row_group_stats.equals(rg_stats_ref)


def test_opd_remove_row_group_files(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    # Keep ref before removing.
    file_ids_to_remove = [0, 2]
    file_ids_to_keep = [i for i in opd.row_group_stats.index if i not in file_ids_to_remove]
    rg_stats_ref = opd.row_group_stats.iloc[file_ids_to_keep].reset_index(drop=True)
    assert not opd._has_row_groups_already_removed
    opd.remove_row_group_files(file_ids=file_ids_to_remove)
    assert len(opd) == len(df_ref) - len(file_ids_to_remove)
    assert opd.row_group_stats.equals(rg_stats_ref)
    assert opd._has_row_groups_already_removed


def test_opd_sort_row_groups(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=False,
    )
    # Unsorting file_ids.
    switch_row_group_file_ids(opd, 2, 4)
    switch_row_group_file_ids(opd, 4, 6)
    # Un-order row groups 'ordered_on_mins' and 'ordered_on_maxs'.
    opd.row_group_stats.sort_values(by=KEY_FILE_IDS, inplace=True)
    assert not opd.row_group_stats[KEY_ORDERED_ON_MINS].is_monotonic_increasing
    opd.sort_row_groups()
    assert opd.row_group_stats[KEY_ORDERED_ON_MINS].is_monotonic_increasing
    df_res = opd[2:].to_pandas()
    assert df_ref.iloc[2:].equals(df_res)


def test_opd_to_pandas(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    range_df = list(range(len(df_ref) + 1))
    opd.write_row_group_files(
        [df_ref.iloc[i:j] for i, j in zip(range_df[:-1], range_df[1:])],
        write_opdmd=True,
    )
    df_res = opd[2:4].to_pandas()
    assert df_ref.iloc[2:4].equals(df_res)


def test_opd_write_metadata(tmp_path):
    opd1 = OrderedParquetDataset(tmp_path, ordered_on="a")
    metadata_ref = {"a": "b", "ts": Timestamp("2021-01-01")}
    opd1.write_metadata(key_value_metadata=metadata_ref)
    assert opd1.row_group_stats.empty
    assert opd1.key_value_metadata == metadata_ref
    opd2 = OrderedParquetDataset(tmp_path)
    assert opd2.row_group_stats.empty
    assert opd2.key_value_metadata == metadata_ref
    # Changing some metadata values, removing another one.
    additional_metadata_in = {"a": "c", "ts": None}
    opd1.write_metadata(key_value_metadata=additional_metadata_in)
    metadata_ref = {"a": "c"}
    assert opd1.key_value_metadata == metadata_ref
    opd2 = OrderedParquetDataset(tmp_path)
    assert opd2.key_value_metadata == metadata_ref


@pytest.mark.parametrize("write_opdmd", [False, True])
def test_opd_write_row_group_files(tmp_path, write_opdmd):
    opd1 = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    opd1.write_row_group_files([df_ref.iloc[:2], df_ref.iloc[2:]], write_opdmd=write_opdmd)
    rgs_stats_ref = DataFrame(
        {
            KEY_FILE_IDS: [0, 1],
            KEY_N_ROWS: [2, len(df_ref) - 2],
            KEY_ORDERED_ON_MINS: [
                df_ref.loc[:, "timestamp"].iloc[0],
                df_ref.loc[:, "timestamp"].iloc[2],
            ],
            KEY_ORDERED_ON_MAXS: [
                df_ref.loc[:, "timestamp"].iloc[1],
                df_ref.loc[:, "timestamp"].iloc[-1],
            ],
        },
    ).astype(RGS_STATS_BASE_DTYPES)
    assert opd1.row_group_stats.equals(rgs_stats_ref)
    if write_opdmd:
        opd2 = OrderedParquetDataset(tmp_path)
        assert opd2.row_group_stats.equals(rgs_stats_ref)


def test_exception_opd_write_row_group_files_max_file_id_reached(tmp_path, monkeypatch):
    """
    Test error when reaching maximum file ID.
    """
    # Modify RGS_STATS_BASE_DTYPES to use int8 for KEY_FILE_IDS which has a lower max value
    int8_type = int8
    exceeding_max_n_files = iinfo(int8_type).max + 2
    # Patch the RGS_STATS_BASE_DTYPES dictionary
    monkeypatch.setitem(RGS_STATS_BASE_DTYPES, KEY_FILE_IDS, int8_type)

    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
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

    opd_tmp = OrderedParquetDataset(tmp_path)
    assert opd_tmp.row_group_stats.loc[:, KEY_FILE_IDS].iloc[-1] == max_file_id - 1

    # Try to write one more.
    with pytest.raises(
        ValueError,
        match=f"^file id '{max_file_id+1}' exceeds max value {max_file_id}",
    ):
        opd.write_row_group_files(dataframes[max_file_id:], write_opdmd=False)

    opd_tmp = OrderedParquetDataset(tmp_path)
    # Check that the opmd file has been correctly rewritten.
    assert opd_tmp.row_group_stats.loc[:, KEY_FILE_IDS].iloc[-1] == max_file_id


def test_exception_opd_write_row_group_files_max_n_rows_reached(tmp_path, monkeypatch):
    """
    Test error when reaching maximum number of rows.
    """
    # Modify RGS_STATS_BASE_DTYPES to use int8 for KEY_N_ROWS which has a lower max value
    int8_type = int8
    exceeding_max_n_rows = iinfo(int8_type).max + 1
    # Patch the RGS_STATS_BASE_DTYPES dictionary
    monkeypatch.setitem(RGS_STATS_BASE_DTYPES, KEY_N_ROWS, int8_type)

    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
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
    opd = OrderedParquetDataset(tmp_path, ordered_on="a")
    with pytest.raises(
        ValueError,
        match="^'ordered_on' column 'a' is not in",
    ):
        opd.write_row_group_files([df_ref])


def test_opd_write(tmp_path):
    opd = OrderedParquetDataset(tmp_path, ordered_on="timestamp")
    opd.write(
        df=df_ref,
        row_group_target_size="1h",
        max_n_off_target_rgs=2,
        duplicates_on="temperature",
    )
    written_df = OrderedParquetDataset(tmp_path).to_pandas()
    assert written_df.equals(df_ref)
