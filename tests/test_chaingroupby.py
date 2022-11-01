#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import array
from numpy import count_nonzero
from numpy import max as nmax
from numpy import ndarray
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp

from oups.chaingroupby import AGG_FUNC_IDS
from oups.chaingroupby import DTYPE_DATETIME64
from oups.chaingroupby import DTYPE_FLOAT64
from oups.chaingroupby import DTYPE_INT64
from oups.chaingroupby import FIRST
from oups.chaingroupby import ID_FIRST
from oups.chaingroupby import ID_LAST
from oups.chaingroupby import ID_MAX
from oups.chaingroupby import ID_MIN
from oups.chaingroupby import ID_SUM
from oups.chaingroupby import LAST
from oups.chaingroupby import MAX
from oups.chaingroupby import MIN
from oups.chaingroupby import SUM
from oups.chaingroupby import _histo_on_ordered
from oups.chaingroupby import _jitted_cgb
from oups.chaingroupby import by_time
from oups.chaingroupby import chaingroupby
from oups.chaingroupby import setup_cgb_agg


INT64 = "int64"
FLOAT64 = "float64"
DATETIME64 = "datetime64"

# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


def test_setup_cgb_agg():
    # Test config generation for aggregation step in chaingroupby.
    # ``{dtype: ndarray[int64], 'agg_func_idx'
    #                           1d-array, aggregation function indices,
    #           ndarray[int64], 'n_cols'
    #                           1d-array, number of input columns in data,
    #                           to which apply this aggregation function,
    #           List[str], 'cols_name_in_data'
    #                      column name in input data, with this dtype,
    #           ndarray[int], 'cols_idx_in_data'
    #                         2d-array, column indices in input data,
    #                         per aggregation function,
    #           List[str], 'cols_name_in_agg_res'
    #                      expected column names in aggregation result,
    #           ndarray[int], 'cols_idx_in_agg_res'
    #                         2d-array, column indices in aggregation
    #                         results, per aggregation function.
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
            array([ID_FIRST, ID_SUM], dtype=INT64),
            array([2, 1], dtype=INT64),
            ["val1_float", "val2_float"],
            array([[0, 1], [1, 0]], dtype=INT64),
            ["val1_first", "val2_first", "val2_sum"],
            array([[0, 1], [2, 0]], dtype=INT64),
        ],
        DTYPE_DATETIME64: [
            array([ID_FIRST], dtype=INT64),
            array([1], dtype=INT64),
            ["val4_datetime"],
            array([[0]], dtype=INT64),
            ["val4_first"],
            array([[0]], dtype=INT64),
        ],
        DTYPE_INT64: [
            array([ID_LAST, ID_MIN, ID_MAX], dtype=INT64),
            array([1, 1, 1], dtype=INT64),
            ["val3_int"],
            array([[0], [0], [0]], dtype=INT64),
            ["val3_last", "val3_min", "val3_max"],
            array([[0], [1], [2]], dtype=INT64),
        ],
    }
    for val_res, val_ref in zip(cgb_agg_cfg_res.values(), cgb_agg_cfg_ref.values()):
        for it_res, it_ref in zip(val_res, val_ref):
            if isinstance(it_res, ndarray):
                assert nall(it_res == it_ref)
            else:
                assert it_res == it_ref


def test_jitted_cgb_1d():
    # Data is 1d, and aggregation result is 1d.
    # Setup.
    group_sizes = array([3, 0, 2, 0, 1], dtype=INT64)
    agg_res_n_rows = len(group_sizes)
    n_nan_groups = agg_res_n_rows - count_nonzero(group_sizes)
    null_group_indices_res = zeros(n_nan_groups, dtype=INT64)
    # Define arrays for one type.
    data = array(
        [
            1.0,
            4.0,
            3.0,
            7.0,
            2.0,
            6.0,
        ],
        dtype=FLOAT64,
    ).reshape(-1, 1)
    # 3 aggregation functions defined.
    agg_func = array([AGG_FUNC_IDS[FIRST]], dtype=INT64)
    # 1st aggregation function: 1 column
    agg_func_n_cols = array([1], dtype=INT64)
    # Column indices for input data, per aggregation function.
    agg_cols_in_data = array([[0]], dtype=INT64)
    # Column indices for output data, per aggregation function.
    agg_cols_in_res = array([[0]], dtype=INT64)
    agg_res_n_cols = nmax(agg_cols_in_res) + 1
    agg_res = zeros((agg_res_n_rows, agg_res_n_cols), dtype=FLOAT64)
    # Define arrays for the "other" types.
    other_data_int = zeros(0, dtype=INT64)
    other_agg_func = zeros(0, dtype=INT64)
    other_agg_func_n_cols = zeros(0, dtype=INT64)
    other_agg_cols_in_data = zeros(0, dtype=INT64)
    other_agg_cols_in_res = zeros(0, dtype=INT64)
    other_agg_res_int = zeros(0, dtype=INT64)
    config = {
        "data_float": data,
        "agg_func_float": agg_func,
        "n_cols_float": agg_func_n_cols,
        "cols_in_data_float": agg_cols_in_data,
        "cols_in_agg_res_float": agg_cols_in_res,
        "agg_res_float": agg_res,
        "data_int": other_data_int,
        "agg_func_int": other_agg_func,
        "n_cols_int": other_agg_func_n_cols,
        "cols_in_data_int": other_agg_cols_in_data,
        "cols_in_agg_res_int": other_agg_cols_in_res,
        "agg_res_int": other_agg_res_int,
        "data_dte": other_data_int,
        "agg_func_dte": other_agg_func,
        "n_cols_dte": other_agg_func_n_cols,
        "cols_in_data_dte": other_agg_cols_in_data,
        "cols_in_agg_res_dte": other_agg_cols_in_res,
        "agg_res_dte": other_agg_res_int,
    }
    # Test.
    _jitted_cgb(group_sizes=group_sizes, **config, null_group_indices=null_group_indices_res)
    # Ref. results.
    # first in data col 0.
    ref_res = (
        array(
            [
                [1.0],
                [0.0],
                [7.0],
                [0.0],
                [6.0],
            ],
            dtype=FLOAT64,
        ),
    )
    assert nall(ref_res == agg_res)
    assert nall(null_group_indices_res == array([1, 3], dtype=INT64))


@pytest.mark.parametrize(
    "type_, agg_func1, agg_func2, agg_func3",
    [
        (FLOAT64, FIRST, MIN, LAST),
        (FLOAT64, LAST, MAX, SUM),
        (INT64, FIRST, MIN, LAST),
        (INT64, LAST, MAX, SUM),
        (DATETIME64, FIRST, MIN, LAST),
    ],
)
def test_jitted_cgb_2d(type_, agg_func1, agg_func2, agg_func3):
    # Data is 2d, and aggregation results are 2d.
    # Setup.
    group_sizes = array([3, 0, 2, 0, 1], dtype=INT64)
    agg_res_n_rows = len(group_sizes)
    n_nan_groups = agg_res_n_rows - count_nonzero(group_sizes)
    nan_group_indices_res = zeros(n_nan_groups, dtype=INT64)
    # Define arrays for one type.
    data = array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [3.0, 2.0, 1.0],
            [7.0, 8.0, 9.0],
            [2.0, 8.0, 1.0],
            [6.0, 4.0, 5.0],
        ],
        dtype=FLOAT64 if type_ == FLOAT64 else INT64,
    )
    # 3 aggregation functions defined.
    agg_func = array(
        [AGG_FUNC_IDS[agg_func1], AGG_FUNC_IDS[agg_func2], AGG_FUNC_IDS[agg_func3]], dtype=INT64
    )
    # 1st aggregation function: 2 columns
    # 2nd aggregation function: 2 columns
    # 3rd aggregation function: 1 column
    agg_func_n_cols = array([2, 2, 1], dtype=INT64)
    # Column indices for input data, per aggregation function.
    # Irrelevant value are set to -1. It could be anything, they won't be read.
    agg_cols_in_data = array([[0, 2, -1], [1, 2, -1], [1, -1, -1]], dtype=INT64)
    # Column indices for output data, per aggregation function.
    # Irrelevant value are set to -1. It could be anything, they won't be read.
    agg_cols_in_res = array([[0, 1, -1], [2, 3, -1], [4, -1, -1]], dtype=INT64)
    agg_res_n_cols = nmax(agg_cols_in_res) + 1
    agg_res = zeros((agg_res_n_rows, agg_res_n_cols), dtype=FLOAT64 if type_ == FLOAT64 else INT64)
    # Define arrays for the "other" types.
    other_data_float = zeros(0, dtype=FLOAT64)
    other_data_int = zeros(0, dtype=INT64)
    other_agg_func = zeros(0, dtype=INT64)
    other_agg_func_n_cols = zeros(0, dtype=INT64)
    other_agg_cols_in_data = zeros(0, dtype=INT64)
    other_agg_cols_in_res = zeros(0, dtype=INT64)
    other_agg_res_float = zeros(0, dtype=FLOAT64)
    other_agg_res_int = zeros(0, dtype=INT64)
    if type_ == FLOAT64:
        config = {
            "data_float": data,
            "agg_func_float": agg_func,
            "n_cols_float": agg_func_n_cols,
            "cols_in_data_float": agg_cols_in_data,
            "cols_in_agg_res_float": agg_cols_in_res,
            "agg_res_float": agg_res,
            "data_int": other_data_int,
            "agg_func_int": other_agg_func,
            "n_cols_int": other_agg_func_n_cols,
            "cols_in_data_int": other_agg_cols_in_data,
            "cols_in_agg_res_int": other_agg_cols_in_res,
            "agg_res_int": other_agg_res_int,
            "data_dte": other_data_int,
            "agg_func_dte": other_agg_func,
            "n_cols_dte": other_agg_func_n_cols,
            "cols_in_data_dte": other_agg_cols_in_data,
            "cols_in_agg_res_dte": other_agg_cols_in_res,
            "agg_res_dte": other_agg_res_int,
        }
    elif type_ == INT64:
        config = {
            "data_float": other_data_float,
            "agg_func_float": other_agg_func,
            "n_cols_float": other_agg_func_n_cols,
            "cols_in_data_float": other_agg_cols_in_data,
            "cols_in_agg_res_float": other_agg_cols_in_res,
            "agg_res_float": other_agg_res_float,
            "data_int": data,
            "agg_func_int": agg_func,
            "n_cols_int": agg_func_n_cols,
            "cols_in_data_int": agg_cols_in_data,
            "cols_in_agg_res_int": agg_cols_in_res,
            "agg_res_int": agg_res,
            "data_dte": other_data_int,
            "agg_func_dte": other_agg_func,
            "n_cols_dte": other_agg_func_n_cols,
            "cols_in_data_dte": other_agg_cols_in_data,
            "cols_in_agg_res_dte": other_agg_cols_in_res,
            "agg_res_dte": other_agg_res_int,
        }
    else:
        config = {
            "data_float": other_data_float,
            "agg_func_float": other_agg_func,
            "n_cols_float": other_agg_func_n_cols,
            "cols_in_data_float": other_agg_cols_in_data,
            "cols_in_agg_res_float": other_agg_cols_in_res,
            "agg_res_float": other_agg_res_float,
            "data_int": other_data_int,
            "agg_func_int": other_agg_func,
            "n_cols_int": other_agg_func_n_cols,
            "cols_in_data_int": other_agg_cols_in_data,
            "cols_in_agg_res_int": other_agg_cols_in_res,
            "agg_res_int": other_agg_res_int,
            "data_dte": data,
            "agg_func_dte": agg_func,
            "n_cols_dte": agg_func_n_cols,
            "cols_in_data_dte": agg_cols_in_data,
            "cols_in_agg_res_dte": agg_cols_in_res,
            "agg_res_dte": agg_res,
        }
    # Test.
    _jitted_cgb(group_sizes=group_sizes, **config, null_group_indices=nan_group_indices_res)
    # Ref. results.
    ref_res = {
        (FLOAT64, FIRST, MIN, LAST):
        # first in data cols 0 & 2.
        # min in data cols 1 & 2.
        # last in col 1.
        array(
            [
                [1.0, 3.0, 2.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [7.0, 9.0, 8.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [6.0, 5.0, 4.0, 5.0, 4.0],
            ],
            dtype=FLOAT64,
        ),
        (FLOAT64, LAST, MAX, SUM):
        # last in data cols 0 & 2.
        # max in data cols 1 & 2.
        # sum in col 1.
        array(
            [
                [3.0, 1.0, 5.0, 6.0, 9.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 8.0, 9.0, 16.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [6.0, 5.0, 4.0, 5.0, 4.0],
            ],
            dtype=FLOAT64,
        ),
        (INT64, FIRST, MIN, LAST):
        # first in data cols 0 & 2.
        # min in data cols 1 & 2.
        # last in col 1.
        array(
            [[1, 3, 2, 1, 2], [0, 0, 0, 0, 0], [7, 9, 8, 1, 8], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
            dtype=INT64,
        ),
        (INT64, LAST, MAX, SUM):
        # last in data cols 0 & 2.
        # max in data cols 1 & 2.
        # sum in col 1.
        array(
            [[3, 1, 5, 6, 9], [0, 0, 0, 0, 0], [2, 1, 8, 9, 16], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
            dtype=INT64,
        ),
        (DATETIME64, FIRST, MIN, LAST):
        # first in data cols 0 & 2.
        # min in data cols 1 & 2.
        # last in col 1.
        array(
            [[1, 3, 2, 1, 2], [0, 0, 0, 0, 0], [7, 9, 8, 1, 8], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
            dtype=INT64,
        ),
    }
    assert nall(ref_res[(type_, agg_func1, agg_func2, agg_func3)] == agg_res)
    assert nall(nan_group_indices_res == array([1, 3], dtype=INT64))


@pytest.mark.parametrize(
    "data, bins, right, hist_ref",
    [
        (
            array([1, 2, 3, 4, 7, 8, 9], dtype=INT64),
            array([2, 5, 6, 7, 8], dtype=INT64),
            True,
            array([2, 0, 1, 1], dtype=INT64),
        ),
        (
            array([1, 2, 3, 4, 7, 8, 9], dtype=INT64),
            array([2, 5, 6, 7, 8], dtype=INT64),
            False,
            array([3, 0, 0, 1], dtype=INT64),
        ),
        (
            array([1, 2, 3, 4, 7, 8, 9], dtype=INT64),
            array([50, 60, 70, 88], dtype=INT64),
            False,
            array([0, 0, 0], dtype=INT64),
        ),
        (
            array([10, 22, 32], dtype=INT64),
            array([5, 6, 7, 8], dtype=INT64),
            False,
            array([0, 0, 0], dtype=INT64),
        ),
        (
            array([5, 5, 6], dtype=INT64),
            array([5, 6, 7, 8], dtype=INT64),
            False,
            array([2, 1, 0], dtype=INT64),
        ),
        (
            array([5, 5, 7, 7, 7], dtype=INT64),
            array([5, 6, 7, 8], dtype=INT64),
            False,
            array([2, 0, 3], dtype=INT64),
        ),
    ],
)
def test_histo_on_ordered(data, bins, right, hist_ref):
    hist_res = zeros(len(bins) - 1, dtype=INT64)
    _histo_on_ordered(data, bins, right, hist_res)
    assert nall(hist_ref == hist_res)


@pytest.mark.parametrize(
    "bin_on, by, group_keys_ref, group_sizes_ref",
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
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([2, 1], dtype=INT64),
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
            DatetimeIndex(["2020-01-01 08:00:00"], dtype="datetime64[ns]", freq="5T"),
            array([3], dtype=INT64),
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
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([2, 1], dtype=INT64),
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
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([2, 1], dtype=INT64),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype="datetime64[ns]", freq="5T"),
            array([3], dtype=INT64),
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
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([1, 2], dtype=INT64),
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
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([1, 2], dtype=INT64),
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
            DatetimeIndex(["2020-01-01 08:00:00"], dtype="datetime64[ns]", freq="5T"),
            array([3], dtype=INT64),
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
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype="datetime64[ns]", freq="5T"
            ),
            array([1, 2], dtype=INT64),
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
            DatetimeIndex(["2020-01-01 08:05:00"], dtype="datetime64[ns]", freq="5T"),
            array([3], dtype=INT64),
        ),
    ],
)
def test_by_time(bin_on, by, group_keys_ref, group_sizes_ref):
    group_keys_res, group_sizes_res = by_time(bin_on, by)
    assert nall(group_keys_res == group_keys_ref)
    assert nall(group_sizes_res == group_sizes_ref)


@pytest.mark.parametrize(
    "ar, cols",
    [
        (
            array([[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]], dtype=FLOAT64),
            True,
        ),
        (array([[2, 20], [4, 40], [5, 50], [8, 80], [9, 90]], dtype=INT64), True),
        (
            array(
                [
                    ["2020-01-01T09:00", "2020-01-02T10:00"],
                    ["2020-01-01T09:05", "2020-01-02T10:05"],
                    ["2020-01-01T09:08", "2020-01-02T10:08"],
                    ["2020-01-01T09:12", "2020-01-02T10:12"],
                    ["2020-01-01T09:14", "2020-01-02T14:00"],
                ],
                dtype=DATETIME64,
            ),
            True,
        ),
        (array([2.0, 4.0, 5.0, 8.0, 9.0], dtype=FLOAT64), False),
        (array([20, 40, 50, 80, 90], dtype=INT64), False),
        (
            array(
                [
                    "2020-01-01T09:00",
                    "2020-01-01T09:05",
                    "2020-01-01T09:08",
                    "2020-01-01T09:12",
                    "2020-01-01T09:14",
                ],
                dtype=DATETIME64,
            ),
            False,
        ),
    ],
)
def test_chaingroupby_single_dtype(ar, cols):
    # Test aggregation for a single dtype.
    ar_dte = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:05",
            "2020-01-01T08:08",
            "2020-01-01T08:12",
            "2020-01-01T08:14",
        ],
        dtype=DATETIME64,
    )
    time_idx = "datetime_idx"
    if cols:
        # 2-column data
        data = pDataFrame(
            {
                "col1": ar[:, 0],
                "col2": ar[:, 1],
                time_idx: ar_dte,
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
                "col1_f": ar,
                time_idx: ar_dte,
            }
        )
        agg = {"res_first": ("col1_f", "first"), "res_last": ("col1_f", "last")}
    by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    agg_res = chaingroupby(by=by, agg=agg, data=data)
    agg_res_ref = data.groupby(by).agg(**agg)
    assert agg_res.equals(agg_res_ref)


# WiP
# Test case test_jitted_cgb on a single column array (1d) instead of 3 columns currently.
# Test case with single column array to check transpose things.
# Make a simple test with a single dtype to check all actions for other dtype
# does not perturb the overall work.
