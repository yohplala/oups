#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from functools import partial

import pytest
from numpy import all as nall
from numpy import arange
from numpy import array
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import date_range

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_INT64
from oups.segmentby import BIN_BY
from oups.segmentby import LEFT
from oups.segmentby import NULL_INT64_1D_ARRAY
from oups.segmentby import ON_COLS
from oups.segmentby import ORDERED_ON
from oups.segmentby import RIGHT
from oups.segmentby import _next_chunk_starts
from oups.segmentby import by_scale
from oups.segmentby import by_x_rows
from oups.segmentby import mergesort
from oups.segmentby import segmentby
from oups.segmentby import setup_segmentby


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "data, right_edges, right, ref, n_null_chunks_ref, edges_traversed_ref",
    [
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      2  4  4  5  6
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            True,
            array([2, 4, 4, 5, 6], dtype=DTYPE_INT64),
            1,
            False,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      1  4  4  4  5
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([1, 4, 4, 4, 5], dtype=DTYPE_INT64),
            2,
            False,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      7   7   7   7
            array([50, 60, 70, 88], dtype=DTYPE_INT64),
            False,
            array([7], dtype=DTYPE_INT64),
            0,
            True,
        ),
        (
            #       0   1   2
            array([10, 22, 32], dtype=DTYPE_INT64),
            #      0  0  0  0
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 0, 0, 0], dtype=DTYPE_INT64),
            4,
            False,
        ),
        (
            #      0  1  2
            array([5, 5, 6], dtype=DTYPE_INT64),
            #      0  2  3  3
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 3], dtype=DTYPE_INT64),
            1,
            True,
        ),
        (
            #      0  1  2
            array([5, 5, 6], dtype=DTYPE_INT64),
            #      2  3  3  3
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            True,
            array([2, 3], dtype=DTYPE_INT64),
            0,
            True,
        ),
        (
            #      0  1  2  3  4
            array([5, 5, 7, 7, 7], dtype=DTYPE_INT64),
            #      0  2  2  5
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 2, 5], dtype=DTYPE_INT64),
            2,
            True,
        ),
    ],
)
def test_next_chunk_starts(data, right_edges, right, ref, n_null_chunks_ref, edges_traversed_ref):
    next_chunk_starts, n_null_chunks, edges_traversed = _next_chunk_starts(data, right_edges, right)
    assert nall(next_chunk_starts == ref)
    assert n_null_chunks == n_null_chunks_ref
    assert edges_traversed == edges_traversed_ref


@pytest.mark.parametrize(
    "on, by, chunk_labels_ref, next_chunk_starts_ref, n_null_chunks_ref, chunk_ends_ref",
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
            DatetimeIndex(
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
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
            DatetimeIndex(
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
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
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
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
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
        ),
    ],
)
def test_by_scale(
    on, by, chunk_labels_ref, next_chunk_starts_ref, n_null_chunks_ref, chunk_ends_ref
):
    # next_chunk_starts, chunk_labels, n_null_chunks, by_closed, chunk_ends, False
    (
        next_chunk_starts,
        chunk_labels,
        n_null_chunks,
        by_closed,
        chunk_ends,
        unknown_last_chunk_end,
    ) = by_scale(on, by)
    assert nall(chunk_labels == chunk_labels_ref)
    assert nall(chunk_ends == chunk_ends_ref)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert n_null_chunks == n_null_chunks_ref
    assert by_closed == by.closed
    assert not unknown_last_chunk_end


@pytest.mark.parametrize(
    "by, closed, exception_mess",
    [
        (DatetimeIndex([pTimestamp("2020/01/01 08:00")]), None, "^'closed' has to be set"),
        (
            # Labels
            (
                DatetimeIndex([pTimestamp("2020/01/01 08:00")]),
                # Ends
                date_range("2020/01/01 08:00", periods=2, freq="3T"),
            ),
            RIGHT,
            "^number of chunk labels",
        ),
    ],
)
def test_by_scale_exceptions_if_datetimeindex(by, closed, exception_mess):
    on = Series(date_range("2020/01/01 07:59", "2020/01/01 07:09", freq="2T"))
    with pytest.raises(ValueError, match=exception_mess):
        by_scale(on, by, closed)


@pytest.mark.parametrize(
    "len_data, x_rows, closed, chunk_starts_ref, next_chunk_starts_ref, unknown_last_bin_end_ref",
    [
        (3, 4, LEFT, array([0]), array([3]), True),
        (7, 4, LEFT, array([0, 4]), array([4, 7]), True),
        (7, 4, RIGHT, array([0, 4]), array([4, 7]), True),
        (8, 4, LEFT, array([0, 4]), array([4, 8]), True),
        (8, 4, RIGHT, array([0, 4]), array([4, 8]), False),
    ],
)
def test_by_x_rows(
    len_data,
    x_rows,
    closed,
    chunk_starts_ref,
    next_chunk_starts_ref,
    unknown_last_bin_end_ref,
):
    start = pTimestamp("2022/01/01 08:00")
    dummy_data = arange(len_data)
    data = pDataFrame(
        {"dummy_data": dummy_data, "dti": date_range(start, periods=len_data, freq="1H")}
    )
    chunk_labels_ref = data.iloc[chunk_starts_ref, -1].reset_index(drop=True)
    chunk_ends_idx = next_chunk_starts_ref.copy()
    if closed == LEFT:
        chunk_ends_idx[-1] = len_data - 1
        chunk_ends_ref = data.iloc[chunk_ends_idx, -1].reset_index(drop=True)
    else:
        chunk_ends_idx -= 1
        chunk_ends_ref = data.iloc[chunk_ends_idx, -1].reset_index(drop=True)
    (
        next_chunk_starts,
        chunk_labels,
        n_null_chunks,
        chunk_closed,
        chunk_ends,
        unknown_last_bin_end,
    ) = by_x_rows(data, x_rows, closed=closed)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert nall(chunk_labels == chunk_labels_ref)
    assert not n_null_chunks
    assert chunk_closed == closed
    # 'chunk_ends' is expected to be the same than 'chunk_labels'.
    assert nall(chunk_ends == chunk_ends_ref)
    assert unknown_last_bin_end == unknown_last_bin_end_ref


def test_mergesort_labels_and_keys():
    # Test data
    labels1 = array([10, 5, 98, 12, 32, 2, 2], dtype="int64")
    keys1 = array([1, 5, 9, 15, 19, 20, 20], dtype="int64")
    labels2 = array([20, 4, 16, 17, 18, 20], dtype="int64")
    keys2 = array([2, 4, 15, 15, 18, 20], dtype="int64")
    sorted_labels, sorted_idx_labels2 = mergesort((labels1, labels2), (keys1, keys2))
    # Insertion indices:         1  2              6   7   8            12
    ref_sorted_labels = array([10, 20, 4, 5, 98, 12, 16, 17, 18, 32, 2, 2, 20])
    ref_sorted_idx_labels2 = array([1, 2, 6, 7, 8, 12])
    # Test
    assert nall(sorted_labels == ref_sorted_labels)
    assert nall(sorted_idx_labels2 == ref_sorted_idx_labels2)


def test_mergesort_labels_and_keys_force_last():
    # Test data
    labels1 = array([10, 5, 98, 12, 32, 2, 2], dtype="int64")
    keys1 = array([1, 5, 9, 15, 19, 20, 20], dtype="int64")
    labels2 = array([20, 4, 16, 17, 18, 20], dtype="int64")
    keys2 = array([2, 4, 15, 15, 18, 19], dtype="int64")
    sorted_labels, sorted_idx_labels2 = mergesort((labels1, labels2), (keys1, keys2))
    # Insertion indices:         1  2              6   7   8      10
    ref_sorted_labels = array([10, 20, 4, 5, 98, 12, 16, 17, 18, 32, 20, 2, 2])
    ref_sorted_idx_labels2 = array([1, 2, 6, 7, 8, 10])
    # Test
    assert nall(sorted_labels == ref_sorted_labels)
    assert nall(sorted_idx_labels2 == ref_sorted_idx_labels2)
    # Force last to be from second array.
    sorted_labels, sorted_idx_labels2 = mergesort((labels1, labels2), (keys1, keys2), True)
    # Insertion indices:         1  2              6   7   8            12
    ref_sorted_labels = array([10, 20, 4, 5, 98, 12, 16, 17, 18, 32, 2, 2, 20])
    ref_sorted_idx_labels2 = array([1, 2, 6, 7, 8, 12])
    # Test
    assert nall(sorted_labels == ref_sorted_labels)
    assert nall(sorted_idx_labels2 == ref_sorted_idx_labels2)


@pytest.mark.parametrize(
    "labels1, labels2, exception_mess",
    [
        (
            array([10, 5], dtype="int64"),
            array([20], dtype="int64"),
            "^not possible to have arrays of different length for first",
        ),
        (
            array([10], dtype="int64"),
            array([20, 5], dtype="int64"),
            "^not possible to have arrays of different length for second",
        ),
    ],
)
def test_mergesort_exceptions_array_length(labels1, labels2, exception_mess):
    # Test data
    keys1 = array([1], dtype="int64")
    keys2 = array([2], dtype="int64")
    with pytest.raises(ValueError, match=exception_mess):
        mergesort((labels1, labels2), (keys1, keys2))


@pytest.mark.parametrize(
    "bin_by, bin_on, ordered_on, snap_by, on_cols_ref, ordered_on_ref, next_chunk_starts_ref",
    [
        (
            Grouper(key="dti", freq="5T", label="left", closed="left"),
            None,
            None,
            None,
            "dti",
            "dti",
            array([2, 3]),
        ),
        (
            Grouper(key="dti", freq="5T", label="left", closed="left"),
            "dti",
            None,
            None,
            "dti",
            "dti",
            array([2, 3]),
        ),
        (
            Grouper(key="dti", freq="5T", label="left", closed="left"),
            None,
            "dti",
            None,
            "dti",
            "dti",
            array([2, 3]),
        ),
        (
            by_x_rows,
            "dti",
            "ordered_on",
            None,
            ["dti", "ordered_on"],
            "ordered_on",
            array([3]),
        ),
        (
            by_x_rows,
            "dti",
            None,
            None,
            "dti",
            None,
            array([3]),
        ),
        (
            by_x_rows,
            "dti",
            None,
            Grouper(key="dti2", freq="5T", label="left", closed="left"),
            ["dti", "dti2"],
            "dti2",
            array([3]),
        ),
    ],
)
def test_setup_segmentby(
    bin_by, bin_on, ordered_on, snap_by, on_cols_ref, ordered_on_ref, next_chunk_starts_ref
):
    # Input data for testing 'bin_by' callable.
    res = setup_segmentby(bin_by, bin_on, ordered_on, snap_by)
    if isinstance(bin_by, Grouper):
        on = Series(date_range("2020/01/01 08:01", periods=3, freq="3T"))
        (next_chunk_starts, _, _, _, _, _) = res[BIN_BY](on=on)
    else:
        on = pDataFrame(
            {"dti": date_range("2020/01/01 08:01", periods=3, freq="3T"), "ordered_on": [1, 2, 3]}
        )
        (next_chunk_starts, _, _, _, _, _) = res[BIN_BY](on=on)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert res[ON_COLS] == on_cols_ref
    assert res[ORDERED_ON] == ordered_on_ref


@pytest.mark.parametrize(
    "bin_by, bin_on, ordered_on, snap_by, regex_ref",
    [
        (
            Grouper(key=None, freq="5T", label="left", closed="left"),
            None,
            None,
            None,
            "^not possible to set both 'bin_by.key' and 'bin_on'",
        ),
        (
            Grouper(key="dti1", freq="5T", label="left", closed="left"),
            "dti2",
            None,
            None,
            "^not possible to set 'bin_by.key' and 'bin_on'",
        ),
        (
            Grouper(key="dti1", freq="5T", label="left", closed="left"),
            None,
            "dti2",
            None,
            "^not possible to set 'bin_on' and 'ordered_on'",
        ),
        (
            by_x_rows,
            None,
            None,
            None,
            "not possible to set 'bin_on' to `None`.",
        ),
        (
            Grouper(key="dti1", freq="5T", label="left", closed="left"),
            None,
            None,
            Grouper(key="dti2", freq="5T", label="left", closed="left"),
            "^not possible to set 'ordered_on' and 'snap_by.key'",
        ),
        (
            Grouper(key="dti", freq="5T", label="left", closed="left"),
            None,
            None,
            Grouper(key="dti", freq="5T", label="left", closed="right"),
            "^not possible to set 'bin_by.closed' and 'snap_by.closed'",
        ),
        (
            by_x_rows,
            "dti",
            None,
            DatetimeIndex(["2020/01/01 08:00"]),
            "^not possible to leave 'ordered_on' to `None` in",
        ),
    ],
)
def test_setup_segmentby_exception(bin_by, bin_on, ordered_on, snap_by, regex_ref):
    with pytest.raises(ValueError, match=regex_ref):
        setup_segmentby(bin_by, bin_on, ordered_on, snap_by)


@pytest.mark.parametrize(
    "bin_by, bin_on, ordered_on, snap_by, next_chunk_starts_ref, bin_indices_ref, "
    "bin_labels_ref, n_null_bins_ref, snap_labels_ref, n_max_null_snaps_ref",
    [
        (
            # 0/ 'bin_by' only, as a Grouper.
            Grouper(key="dti", freq="5T", label="left", closed="left"),
            None,
            None,
            None,
            array([1, 2, 4]),
            NULL_INT64_1D_ARRAY,
            DatetimeIndex(["2020/01/01 08:00", "2020/01/01 08:05", "2020/01/01 08:10"]),
            0,
            None,
            0,
        ),
        (
            # 1/ 'bin_by' and 'snap_by' both as a Grouper, left-closed
            # 'snap_by' points excluded (left-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:06    b1-8:05
            #      8:07     s2-8:08    b2-8:10
            #               s3-8:10    b2
            #      8:10     s4-8:12    b3-8:15
            #      8:13     s5-8:14    b3
            Grouper(key="dti", freq="5T", label="right", closed="left"),
            None,
            None,
            Grouper(key="dti", freq="2T"),
            #     b1 s1 s2 s3 b2 s4 s5 b3
            #      0           4        7
            array([1, 1, 2, 2, 2, 3, 4, 4]),
            array([0, 4, 7]),
            DatetimeIndex(["2020/01/01 08:05", "2020/01/01 08:10", "2020/01/01 08:15"]),
            0,
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            2,
        ),
        (
            # 2/ 'bin_by' and 'snap_by' both as a Grouper, right-closed
            # 'snap_by' points included (right-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:04    b1-8:05
            #               s2-8:06    b2-8:10
            #      8:07     s3-8:08    b2
            #      8:10     s4-8:10    b2
            #               s5-8:12    b3-8:15
            #      8:13     s6-8:14    b3
            Grouper(key="dti", freq="5T", label="right", closed="right"),
            None,
            None,
            Grouper(key="dti", freq="2T", closed="right"),
            #     s1 b1 s2 s3 s4 b2 s5 s6 b3
            #         1           5        8
            array([1, 1, 1, 2, 3, 3, 3, 4, 4]),
            array([1, 5, 8]),
            DatetimeIndex(["2020/01/01 08:05", "2020/01/01 08:10", "2020/01/01 08:15"]),
            0,
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            # s2 and s5 are detected twice:
            # - for having same 'next_snap_start' as s1 and s4.
            # - for being after a bin end, sharing the same 'next_chunk_start'.
            4,
        ),
        (
            # 3/ 'bin_by' as a Grouper, left-closed, and 'snap_by' as a
            # DatetimeIndex.
            # 'snap_by' points excluded (left-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:06    b1-8:05
            #      8:07     s2-8:08    b2-8:10
            #               s3-8:10    b2
            #      8:10     s4-8:12    b3-8:15
            #      8:13     s5-8:14    b3
            Grouper(key="dti", freq="5T", label="right", closed="left"),
            None,
            None,
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            #     b1 s1 s2 s3 b2 s4 s5 b3
            #      0           4        7
            array([1, 1, 2, 2, 2, 3, 4, 4]),
            array([0, 4, 7]),
            DatetimeIndex(["2020/01/01 08:05", "2020/01/01 08:10", "2020/01/01 08:15"]),
            0,
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            2,
        ),
        (
            # 4/ 'bin_by' as a Grouper, right-closed, and 'snap_by' as a
            # DatetimeIndex.
            # 'snap_by' points included (right-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:04    b1-8:05
            #               s2-8:06    b2-8:10
            #      8:07     s3-8:08    b2
            #      8:10     s4-8:10    b2
            #               s5-8:12    b3-8:15
            #      8:13     s6-8:14    b3
            Grouper(key="dti", freq="5T", label="right", closed="right"),
            None,
            None,
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            #     s1 b1 s2 s3 s4 b2 s5 s6 b3
            #         1           5        8
            array([1, 1, 1, 2, 3, 3, 3, 4, 4]),
            array([1, 5, 8]),
            DatetimeIndex(["2020/01/01 08:05", "2020/01/01 08:10", "2020/01/01 08:15"]),
            0,
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            # s2 and s5 are detected twice:
            # - for having same 'next_snap_start' as s1 and s4.
            # - for being after a bin end, sharing the same 'next_chunk_start'.
            4,
        ),
        (
            # 5/ 'bin_by' as a Callable, left-closed.
            # 'snap_by' as a Grouper, points excluded (left-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:06    b1-8:06
            #      8:07     s2-8:08    b1
            #               s3-8:10    b1
            #      8:10                b1
            #               s4-8:12    b1
            #      8:13     s5-8:14    b2-8:03
            partial(by_x_rows, by=3),
            "dti",
            None,
            Grouper(key="dti", freq="2T"),
            #     s1 s2 s3 s4 b1 s5 b2
            #                  4     6
            array([1, 2, 2, 3, 3, 4, 4]),
            array([4, 6]),
            DatetimeIndex(["2020/01/01 08:04", "2020/01/01 08:13"])
            .to_series()
            .reset_index(drop=True),
            0,
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            1,
        ),
        (
            # 6/ 'bin_by' as a Callable, right-closed.
            # 'snap_by' as a Grouper, points included (right-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:04    b1-8:04
            #               s2-8:06    b1
            #      8:07     s3-8:08    b1
            #      8:10     s4-8:10    b1
            #               s5-8:12    b2-8:10 excl.
            #      8:13     s6-8:14    b2-8:13
            partial(by_x_rows, by=3, closed=RIGHT),
            "dti",
            None,
            Grouper(key="dti", freq="2T"),
            #     s1 s2 s3 s4 b1 s5 s6 b2
            #                  4        7
            array([1, 1, 2, 3, 3, 3, 4, 4]),
            array([4, 7]),
            DatetimeIndex(["2020/01/01 08:04", "2020/01/01 08:13"])
            .to_series()
            .reset_index(drop=True),
            0,
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            3,
        ),
        (
            # 7/ 'bin_by' as a Callable, left-closed.
            # 'snap_by' as a DatetimeIndex, points excluded (left-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:06    b1-8:06
            #      8:07     s2-8:08    b1
            #               s3-8:10    b1
            #      8:10                b1
            #               s4-8:12    b1
            #      8:13     s5-8:14    b2-8:03
            partial(by_x_rows, by=3),
            "dti",
            "dti",
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            #     s1 s2 s3 s4 b1 s5 b2
            #                  4     6
            array([1, 2, 2, 3, 3, 4, 4]),
            array([4, 6]),
            DatetimeIndex(["2020/01/01 08:04", "2020/01/01 08:13"])
            .to_series()
            .reset_index(drop=True),
            0,
            date_range("2020/01/01 08:06", periods=5, freq="2T"),
            1,
        ),
        (
            # 8/ 'bin_by' as a Callable, right-closed.
            # 'snap_by' as a DatetimeIndex, points included (right-closed)
            # 'data'
            #  datetime       snaps       bins
            #      8:04     s1-8:04    b1-8:04
            #               s2-8:06    b1
            #      8:07     s3-8:08    b1
            #      8:10     s4-8:10    b1
            #               s5-8:12    b2-8:10 excl
            #      8:13     s6-8:14    b2
            partial(by_x_rows, by=3, closed=RIGHT),
            "dti",
            "dti",
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            #     s1 s2 s3 s4 b1 s5 s6 b2
            #                  4        7
            array([1, 1, 2, 3, 3, 3, 4, 4]),
            array([4, 7]),
            DatetimeIndex(["2020/01/01 08:04", "2020/01/01 08:13"])
            .to_series()
            .reset_index(drop=True),
            0,
            date_range("2020/01/01 08:04", periods=6, freq="2T"),
            3,
        ),
    ],
)
def test_segmentby(
    bin_by,
    bin_on,
    ordered_on,
    snap_by,
    next_chunk_starts_ref,
    bin_indices_ref,
    bin_labels_ref,
    n_null_bins_ref,
    snap_labels_ref,
    n_max_null_snaps_ref,
):
    dti = date_range("2020/01/01 08:04", periods=4, freq="3T")
    data = pDataFrame({"dti": dti, "ordered_on": range(len(dti))})
    (
        next_chunk_starts,
        bin_indices,
        bin_labels,
        n_null_bins,
        snap_labels,
        n_max_null_snaps,
    ) = segmentby(data, bin_by, bin_on, ordered_on, snap_by)
    assert nall(next_chunk_starts_ref == next_chunk_starts)
    assert nall(bin_indices_ref == bin_indices)
    assert bin_labels_ref.equals(bin_labels)
    assert n_null_bins_ref == n_null_bins
    if snap_labels_ref is None:
        assert snap_labels is None
    else:
        assert snap_labels_ref.equals(snap_labels)
    assert n_max_null_snaps_ref == n_max_null_snaps


def test_segmentby_with_outer_setup():
    # 'bin_by' and 'snap_by' both as a Grouper, left-closed
    # 'snap_by' points excluded (left-closed)
    # 'data'
    #  datetime       snaps       bins
    #      8:04     s1-8:06    b1-8:05
    #      8:07     s2-8:08    b2-8:10
    #               s3-8:10    b2
    #      8:10     s4-8:12    b3-8:15
    #      8:13     s5-8:14    b3
    bin_by = Grouper(key="dti", freq="5T", label="right", closed="left")
    snap_by = Grouper(key="dti", freq="2T")
    bin_by = setup_segmentby(bin_by=bin_by, snap_by=snap_by)
    dti = date_range("2020/01/01 08:04", periods=4, freq="3T")
    data = pDataFrame({"dti": dti, "ordered_on": range(len(dti))})
    (
        next_chunk_starts,
        bin_indices,
        bin_labels,
        n_null_bins,
        snap_labels,
        n_max_null_snaps,
    ) = segmentby(data=data, bin_by=bin_by)
    # Reference results.
    #                             b1 s1 s2 s3 b2 s4 s5 b3
    #                              0           4        7
    next_chunk_starts_ref = array([1, 1, 2, 2, 2, 3, 4, 4])
    bin_indices_ref = array([0, 4, 7])
    bin_labels_ref = DatetimeIndex(["2020/01/01 08:05", "2020/01/01 08:10", "2020/01/01 08:15"])
    snap_labels_ref = date_range("2020/01/01 08:06", periods=5, freq="2T")
    assert nall(next_chunk_starts_ref == next_chunk_starts)
    assert nall(bin_indices_ref == bin_indices)
    assert bin_labels_ref.equals(bin_labels)
    assert not n_null_bins
    assert snap_labels_ref.equals(snap_labels)
    assert n_max_null_snaps == 2


def test_segmentby_binby_exceptions():
    # Check when 'next_chunk_starts', 'chunk_labels' and 'chunk_ends' from
    # 'bin_by' as a Callable are not all of the same length.
    bin_on = "dti"
    dti = date_range("2020/01/01 08:04", periods=4, freq="3T")
    data = pDataFrame({bin_on: dti, "ordered_on": range(len(dti))})

    def by_wrong_starts_labels(on, buffer=None):
        # 'next_chunk_starts' and 'chunk_labels' not the same size.
        return arange(1), Series(["a", "o"]), 0, LEFT, arange(2), True

    with pytest.raises(ValueError, match="^'next_chunk_starts' and 'chunk_labels'"):
        segmentby(data=data, bin_by=by_wrong_starts_labels, bin_on=bin_on)

    def by_wrong_starts_ends(on, buffer=None):
        # 'next_chunk_starts' and 'chunk_ends' not the same size.
        return arange(3), Series(["a", "o", "u"]), 0, LEFT, arange(1), True

    with pytest.raises(ValueError, match="^'next_chunk_starts' and 'chunk_ends'"):
        segmentby(data=data, bin_by=by_wrong_starts_ends, bin_on=bin_on)

    def by_wrong_closed(on, buffer=None):
        # 'closed' not 'left' or 'right'.
        return arange(2), Series(["a", "o"]), 0, "invented", arange(2), True

    with pytest.raises(ValueError, match="^'bin_closed' has to be set either"):
        segmentby(data=data, bin_by=by_wrong_closed, bin_on=bin_on)


def test_segmentby_orderedon_exceptions():
    # Check when 'ordered_on' is not ordered.
    bin_on = "ordered_on"
    bin_by = Grouper(key=bin_on, freq="2T")
    ordered_on = bin_on
    # 'ordered_on' is a DatetimeIndex.
    unordered = DatetimeIndex(["2020/01/01 08:00", "2020/01/01 09:00", "2020/01/01 07:00"])
    data = pDataFrame({ordered_on: unordered})

    with pytest.raises(ValueError, match="^column 'ordered_on' is not ordered"):
        segmentby(data=data, bin_by=bin_by, bin_on=bin_on, ordered_on=ordered_on)
    # 'ordered_on' are int values.
    unordered = [0, 4, 3]
    data = pDataFrame({ordered_on: unordered})

    with pytest.raises(ValueError, match="^column 'ordered_on' is not ordered"):
        segmentby(data=data, bin_by=bin_by, bin_on=bin_on, ordered_on=ordered_on)
