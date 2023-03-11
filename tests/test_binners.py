#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import arange
from numpy import array
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import date_range

from oups.binners import KEY_LAST_KEY
from oups.binners import KEY_ROWS_IN_LAST_BIN
from oups.binners import _next_chunk_starts
from oups.binners import bin_by_time
from oups.binners import bin_by_x_rows
from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_INT64


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "data, right_edges, right, ref, n_null_chunks_ref",
    [
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      2  4  4  5  6
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            True,
            array([2, 4, 4, 5, 6], dtype=DTYPE_INT64),
            1,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      1  4  4  4  5
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([1, 4, 4, 4, 5], dtype=DTYPE_INT64),
            2,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      7   7   7   7
            array([50, 60, 70, 88], dtype=DTYPE_INT64),
            False,
            array([7, 7, 7, 7], dtype=DTYPE_INT64),
            3,
        ),
        (
            #       0   1   2
            array([10, 22, 32], dtype=DTYPE_INT64),
            #      0  0  0  0
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 0, 0, 0], dtype=DTYPE_INT64),
            4,
        ),
        (
            #      0  1  2
            array([5, 5, 6], dtype=DTYPE_INT64),
            #      0  2  3  3
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 3, 3], dtype=DTYPE_INT64),
            2,
        ),
        (
            #      0  1  2  3  4
            array([5, 5, 7, 7, 7], dtype=DTYPE_INT64),
            #      0  2  2  5
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 2, 5], dtype=DTYPE_INT64),
            2,
        ),
    ],
)
def test_next_chunk_starts(data, right_edges, right, ref, n_null_chunks_ref):
    next_chunk_starts = zeros(len(right_edges), dtype=DTYPE_INT64)
    n_null_chunks = zeros(1, dtype=DTYPE_INT64)
    _next_chunk_starts(data, right_edges, right, next_chunk_starts, n_null_chunks)
    assert nall(ref == next_chunk_starts)
    assert n_null_chunks[0] == n_null_chunks_ref


@pytest.mark.parametrize(
    "bin_on, by, bin_labels_ref, next_chunk_starts_ref, n_null_bins_ref",
    [
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(["2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="left"),
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(["2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="right"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="right"),
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
    ],
)
def test_bin_by_time(bin_on, by, bin_labels_ref, next_chunk_starts_ref, n_null_bins_ref):
    next_chunk_starts, bins, n_null_bins = bin_by_time(bin_on, by)
    bin_labels = bins.left if by.label == "left" else bins.right
    assert nall(bin_labels == bin_labels_ref)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert n_null_bins == n_null_bins_ref


@pytest.mark.parametrize(
    "len_data, x_rows, buffer_in, buffer_out, chunk_starts_ref, next_chunk_starts_ref, set_first_key",
    [
        (3, 4, None, None, array([0]), array([3]), False),
        (7, 4, None, None, array([0, 4]), array([4, 7]), False),
        (8, 4, None, None, array([0, 4]), array([4, 8]), False),
        (
            3,
            4,
            {KEY_ROWS_IN_LAST_BIN: 1, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            {KEY_ROWS_IN_LAST_BIN: 4, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            array([0]),
            array([3]),
            True,
        ),
        (
            7,
            4,
            {KEY_ROWS_IN_LAST_BIN: 1, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            {KEY_ROWS_IN_LAST_BIN: 4, KEY_LAST_KEY: pTimestamp("2022/01/01 11:00")},
            array([0, 3]),
            array([3, 7]),
            True,
        ),
        (
            7,
            4,
            {KEY_ROWS_IN_LAST_BIN: 4, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            {KEY_ROWS_IN_LAST_BIN: 3, KEY_LAST_KEY: pTimestamp("2022/01/01 12:00")},
            array([0, 4]),
            array([4, 7]),
            False,
        ),
        (
            8,
            4,
            {KEY_ROWS_IN_LAST_BIN: 1, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            {KEY_ROWS_IN_LAST_BIN: 1, KEY_LAST_KEY: pTimestamp("2022/01/01 15:00")},
            array([0, 3, 7]),
            array([3, 7, 8]),
            True,
        ),
        (
            8,
            4,
            {KEY_ROWS_IN_LAST_BIN: 4, KEY_LAST_KEY: pTimestamp("2022/01/01 07:50")},
            {KEY_ROWS_IN_LAST_BIN: 4, KEY_LAST_KEY: pTimestamp("2022/01/01 12:00")},
            array([0, 4]),
            array([4, 8]),
            False,
        ),
    ],
)
def test_bin_by_x_rows(
    len_data, x_rows, buffer_in, buffer_out, chunk_starts_ref, next_chunk_starts_ref, set_first_key
):
    start = pTimestamp("2022/01/01 08:00")
    dummy_data = arange(len_data)
    data = pDataFrame(
        {"dummy_data": dummy_data, "dti": date_range(start, periods=len_data, freq="1H")}
    )
    bin_labels_ref = data.iloc[chunk_starts_ref, -1].reset_index(drop=True)
    if buffer_out is not None and set_first_key:
        bin_labels_ref.iloc[0] = buffer_in[KEY_LAST_KEY]
    next_chunk_starts, bin_labels, n_null_bins, bin_closed, bin_ends = bin_by_x_rows(
        data, buffer_in, x_rows
    )
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert nall(bin_labels == bin_labels_ref)
    assert not n_null_bins
    assert bin_closed == "left"
    # 'bin_ends' is expected to be the same than 'bin_labels'.
    assert nall(bin_ends == bin_labels_ref)
    if buffer_out is not None:
        assert buffer_in == buffer_out
