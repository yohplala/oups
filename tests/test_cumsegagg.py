#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import array
from numpy import ndarray
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_FLOAT64
from oups.cumsegagg import DTYPE_INT64
from oups.cumsegagg import _next_chunk_starts
from oups.cumsegagg import bin_by_time
from oups.cumsegagg import cumsegagg
from oups.cumsegagg import setup_cgb_agg
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import LAST
from oups.jcumsegagg import MAX
from oups.jcumsegagg import MIN
from oups.jcumsegagg import SUM


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


def test_setup_cgb_agg():
    # Test config generation for aggregation step in cumsegagg.
    # ``{dtype: List[str], 'cols_name_in_data'
    #                      column name in input data, with this dtype,
    #           List[str], 'cols_name_in_agg_res'
    #                      expected column names in aggregation result,
    #           ndarray[int64], 'cols_idx'
    #                           2d-array,
    #                           Per aggregation function (row index is agg func
    #                           id), column indices in input data, and results.
    #           ndarray[int64], 'n_cols'
    #                           1d-array, number of input columns in data,
    #                           to which apply this aggregation function,
    # Setup.
    df = pDataFrame(
        {
            "val1_float": [1.1, 2.1, 3.1],
            "val2_float": [4.1, 5.1, 6.1],
            "val3_int": [1, 2, 3],
            "val4_datetime": [
                pTimestamp("2022/01/01 08:00"),
                pTimestamp("2022/01/01 09:00"),
                pTimestamp("2022/01/01 08:00"),
            ],
        }
    )
    agg_cfg = {
        "val1_first": ("val1_float", FIRST),
        "val2_first": ("val2_float", FIRST),
        "val2_sum": ("val2_float", SUM),
        "val4_first": ("val4_datetime", FIRST),
        "val3_last": ("val3_int", LAST),
        "val3_min": ("val3_int", MIN),
        "val3_max": ("val3_int", MAX),
    }
    cgb_agg_cfg_res = setup_cgb_agg(agg_cfg, df.dtypes.to_dict())
    cgb_agg_cfg_ref = {
        DTYPE_FLOAT64: [
            ["val1_float", "val2_float"],
            ["val1_first", "val2_first", "val2_sum"],
            array(
                [
                    [[0, 0], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[1, 2], [0, 0]],
                ],
                dtype=DTYPE_INT64,
            ),
            array([2, 0, 0, 0, 1], dtype=DTYPE_INT64),
        ],
        DTYPE_DATETIME64: [
            ["val4_datetime"],
            ["val4_first"],
            array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]], dtype=DTYPE_INT64),
            array([1, 0, 0, 0, 0], dtype=DTYPE_INT64),
        ],
        DTYPE_INT64: [
            ["val3_int"],
            ["val3_last", "val3_min", "val3_max"],
            array([[[0, 0]], [[0, 0]], [[0, 1]], [[0, 2]], [[0, 0]]], dtype=DTYPE_INT64),
            array([0, 1, 1, 1, 0], dtype=DTYPE_INT64),
        ],
    }
    for val_res, val_ref in zip(cgb_agg_cfg_res.values(), cgb_agg_cfg_ref.values()):
        for it_res, it_ref in zip(val_res, val_ref):
            if isinstance(it_res, ndarray):
                assert nall(it_res == it_ref)
            else:
                assert it_res == it_ref


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
    bins, next_chunk_starts, n_null_bins = bin_by_time(bin_on, by)
    bin_labels = bins.left if by.label == "left" else bins.right
    assert nall(bin_labels == bin_labels_ref)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert n_null_bins == n_null_bins_ref


@pytest.mark.parametrize(
    "ndata, cols",
    [
        (
            array(
                [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]],
                dtype=DTYPE_FLOAT64,
            ),
            True,
        ),
        (array([[2, 20], [4, 40], [5, 50], [8, 80], [9, 90]], dtype=DTYPE_INT64), True),
        (
            array(
                [
                    ["2020-01-01T09:00", "2020-01-02T10:00"],
                    ["2020-01-01T09:05", "2020-01-02T10:05"],
                    ["2020-01-01T09:08", "2020-01-02T10:08"],
                    ["2020-01-01T09:12", "2020-01-02T10:12"],
                    ["2020-01-01T09:14", "2020-01-02T14:00"],
                ],
                dtype=DTYPE_DATETIME64,
            ),
            True,
        ),
        (array([2.0, 4.0, 5.0, 8.0, 9.0], dtype=DTYPE_FLOAT64), False),
        (array([20, 40, 50, 80, 90], dtype=DTYPE_INT64), False),
        (
            array(
                [
                    "2020-01-01T09:00",
                    "2020-01-01T09:05",
                    "2020-01-01T09:08",
                    "2020-01-01T09:12",
                    "2020-01-01T09:14",
                ],
                dtype=DTYPE_DATETIME64,
            ),
            False,
        ),
    ],
)
def test_cumsegagg_bin_single_dtype(ndata, cols):
    # Test binning aggregation for a single dtype.
    ndata_dti = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:05",
            "2020-01-01T08:08",
            "2020-01-01T08:12",
            "2020-01-01T08:14",
        ],
        dtype=DTYPE_DATETIME64,
    )
    time_idx = "datetime_idx"
    if cols:
        # 2-column data
        data = pDataFrame(
            {
                "col1": ndata[:, 0],
                "col2": ndata[:, 1],
                time_idx: ndata_dti,
            }
        )
        agg = {
            "res_first": ("col1", "first"),
            "res_last_col1": ("col1", "last"),
            "res_last_col2": ("col2", "last"),
        }
    else:
        # 1-column data
        data = pDataFrame(
            {
                "col1_f": ndata,
                time_idx: ndata_dti,
            }
        )
        agg = {"res_first": ("col1_f", "first"), "res_last": ("col1_f", "last")}
    by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    bin_res = cumsegagg(data=data, agg=agg, bin_by=by)
    bin_res_ref = data.groupby(by).agg(**agg)
    assert bin_res.equals(bin_res_ref)


def test_cumsegagg_bin_mixed_dtype():
    # Test binning aggregation for a mixed dtype.
    ar_float = array(
        [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]], dtype=DTYPE_FLOAT64
    )
    ar_int = array([[1, 10], [3, 30], [6, 60], [7, 70], [9, 90]], dtype=DTYPE_INT64)
    ar_dte = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:05",
            "2020-01-01T08:08",
            "2020-01-01T08:12",
            "2020-01-01T08:14",
        ],
        dtype=DTYPE_DATETIME64,
    )
    time_idx = "datetime_idx"
    data = pDataFrame(
        {
            "col1_f": ar_float[:, 0],
            "col2_f": ar_float[:, 1],
            "col3_i": ar_int[:, 0],
            "col4_i": ar_int[:, 1],
            time_idx: ar_dte,
        }
    )
    agg = {
        "res_first_f": ("col1_f", "first"),
        "res_sum_f": ("col1_f", "sum"),
        "res_last_f": ("col2_f", "last"),
        "res_min_f": ("col3_i", "min"),
        "res_max_f": ("col4_i", "max"),
        "res_first_d": ("datetime_idx", "first"),
    }
    by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    agg_res = cumsegagg(data=data, agg=agg, bin_by=by)
    agg_res_ref = data.groupby(by).agg(**agg)
    assert agg_res.equals(agg_res_ref)


# WiP
# test snapshot: as Grouper, as IntervalIndex
# segment(): test error message: 'by.closed' should either be 'right' or 'left', nothing else.
# Test with null values in agg_res (or modify test case above)
# Test error message if 'bin_on' is None in 'cumsegagg'.
# test error message input column in 'agg' not in input dataframe.
# test snapshot:
# with left-closed bin and snapshot included: snapshot come after bin
# with left-closed bin and snapsht excluded: snapshot come after bin
# Test error snap_by.key or snap_by.name not set
# WiP test case in case snap_by is IntevalIndex: its name has to be set after a column in data.
# WiP: test cases for 'bin_by_time'
# Wip test error: if bin_by and snap_by both a pd Grouper, check that both key parameter
# point on same column, otherwise, not possible possibly to compare
# Test case snapshot with a snapshot ending exactly on a bin end,
# and another not ending on a bin end.
# Snapshot test case with null rows (empty snapshots)
# Have a snapshot ending exactly on a bin end, then a next snapshot bigger than next bin

# Test exception when bin_on and bin_by.key not the same value
# Test sxeception
#   ordered_on not set but snap_by set
#   ordered_on different than snap_by.key
#   ordered_on not ordered
#   ordered_on different than bin_by.key when bin_by is a Grouper
# Test name of 'bin_res' index: ordered_on if ordered_on is set, else bin_on


# Test boh snapshot + bin, but in data there are
#  - only snapshots
#  - only bins
#  - a mix of them
# Make a test with empty bins
# Make a test with empty snapshots (to check all -1 in n_max_null_snap_indices are correctly removed)

# Todo Cumsegagg: remove "if" from the loop
