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
from oups.chaingroupby import jitted_cgb
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
def test_jitted_cgb(type_, agg_func1, agg_func2, agg_func3):
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
    # Define arrays for the "other" type.
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
    jitted_cgb(group_sizes=group_sizes, **config, nan_group_indices=nan_group_indices_res)
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


# WiP
# Make a simple test with a single dtype to check all actions for other dtype
# does not perturb the overall work.
