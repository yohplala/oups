#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import array
from numpy import count_nonzero
from numpy import diff as ndiff
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
from oups.chaingroupby import _jitted_cgb2
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


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jitted_cgb_1d(dtype_):
    # Data is 1d, and aggregation result is 1d.
    # Setup.
    group_sizes = array([3, 0, 2, 0, 1], dtype=INT64)
    agg_res_n_rows = len(group_sizes)
    n_nan_groups = agg_res_n_rows - count_nonzero(group_sizes)
    null_group_indices_res = zeros(n_nan_groups, dtype=INT64)
    # Define arrays for one type.
    data = array(
        [
            1,
            4,
            3,
            7,
            2,
            6,
        ],
        dtype=dtype_,
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
    agg_res = zeros((agg_res_n_rows, agg_res_n_cols), dtype=dtype_)
    # Test.
    _jitted_cgb(
        group_sizes,
        data,
        agg_func,
        agg_func_n_cols,
        agg_cols_in_data,
        agg_cols_in_res,
        True,
        agg_res,
        null_group_indices_res,
    )
    # Ref. results.
    # first in data col 0.
    ref_res = (
        array(
            [
                [1],
                [0],
                [7],
                [0],
                [6],
            ],
            dtype=dtype_,
        ),
    )
    assert nall(ref_res == agg_res)
    assert nall(null_group_indices_res == array([1, 3], dtype=INT64))


@pytest.mark.parametrize(
    "dtype_, agg_func_id, ref_res",
    [
        (FLOAT64, ID_FIRST, [0, 0, 1, 0, 7, 0, 6]),
        (FLOAT64, ID_LAST, [0, 0, 3, 0, 2, 0, 6]),
        (FLOAT64, ID_MIN, [0, 0, 1, 0, 2, 0, 6]),
        (FLOAT64, ID_MAX, [0, 0, 4, 0, 7, 0, 6]),
        (FLOAT64, ID_SUM, [0, 0, 8, 0, 9, 0, 6]),
        (INT64, ID_FIRST, [0, 0, 1, 0, 7, 0, 6]),
        (INT64, ID_LAST, [0, 0, 3, 0, 2, 0, 6]),
        (INT64, ID_MIN, [0, 0, 1, 0, 2, 0, 6]),
        (INT64, ID_MAX, [0, 0, 4, 0, 7, 0, 6]),
        (INT64, ID_SUM, [0, 0, 8, 0, 9, 0, 6]),
    ],
)
def test_jitted_cgb2_bin_1d(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. No snapshot.
    # This test case essentially validates aggregation functions in a sequence
    # of bins.
    # IDX  VAL  BIN  EMPTY_BIN   FIRST  LAST  MIN  MAX  SUM
    #   0    1    2   0, 1           1    3     1    4    8
    #   1    4    2
    #   2    3    2   3              0    0     0    0    0
    #   3    7    4                  7    2     2    7    9
    #   4    2    4   5              0    0     0    0    0
    #   5    6    6                  6    6     6    6    6
    # Setup.
    # Number of columns onto which applying aggregation function
    N_AGG_COLS = 1
    # Row indices of starts for each 'next chunk'.
    next_chunk_starts = array([0, 0, 3, 3, 5, 5, 6], dtype=INT64)
    # Last row provides number of rows in data.
    # 'len()' provides number of chunks, because in this case, there are only
    # bins, no snapshot.
    agg_res_n_rows = len(next_chunk_starts)
    agg_res = zeros((agg_res_n_rows, N_AGG_COLS), dtype=dtype_)
    n_nan_bins = agg_res_n_rows - count_nonzero(ndiff(next_chunk_starts, prepend=0))
    null_bin_indices = zeros(n_nan_bins, dtype=INT64)
    # All chunks are bins.
    bin_indices = array(range(agg_res_n_rows), dtype=INT64)
    # Define arrays for one type.
    data = array([1, 4, 3, 7, 2, 6], dtype=dtype_).reshape(-1, 1)
    # 'n_cols' is always of length of number of aggregation functions.
    n_cols = zeros(len(AGG_FUNC_IDS), dtype=INT64)
    # 1 column in data over which applying 'first' function.
    n_cols[agg_func_id] = N_AGG_COLS
    # Column indices for input data, and results, per aggregation function.
    # Indices are 0.
    cols = zeros((len(AGG_FUNC_IDS), N_AGG_COLS, 2), dtype=INT64)
    # No snapshot.
    snap_res = zeros(0, dtype=FLOAT64)
    null_snap_indices = zeros(0, dtype=INT64)
    # Test.
    _jitted_cgb2(
        data,
        n_cols,
        cols,
        next_chunk_starts,
        bin_indices,
        agg_res,
        snap_res,
        null_bin_indices,
        null_snap_indices,
    )
    # Reference results.
    ref_res_ar = array(
        ref_res,
        dtype=dtype_,
    ).reshape(-1, 1)
    assert nall(ref_res_ar == agg_res)
    assert nall(null_bin_indices == array([0, 1, 3, 5], dtype=INT64))


@pytest.mark.parametrize(
    "dtype_, agg_func_id, ref_res",
    [
        (FLOAT64, ID_FIRST, [0, 0, 1, 1, 1, 1, 1]),
        (FLOAT64, ID_LAST, [0, 0, 3, 3, 2, 2, 6]),
        (FLOAT64, ID_MIN, [0, 0, 1, 1, 1, 1, 1]),
        (FLOAT64, ID_MAX, [0, 0, 4, 4, 7, 7, 7]),
        (FLOAT64, ID_SUM, [0, 0, 8, 8, 17, 17, 23]),
        (INT64, ID_FIRST, [0, 0, 1, 1, 1, 1, 1]),
        (INT64, ID_LAST, [0, 0, 3, 3, 2, 2, 6]),
        (INT64, ID_MIN, [0, 0, 1, 1, 1, 1, 1]),
        (INT64, ID_MAX, [0, 0, 4, 4, 7, 7, 7]),
        (INT64, ID_SUM, [0, 0, 8, 8, 17, 17, 23]),
    ],
)
def test_jitted_cgb2_snap_1d(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. No bin.
    # This test case essentially validates aggregation functions in a sequence
    # of snapshots.
    # IDX  VAL  SNAP  EMPTY_SNAP   FIRST  LAST  MIN  MAX  SUM
    #   0    1    2     0, 1           1    3     1    4    8
    #   1    4    2
    #   2    3    2     3              1    3     1    4    8
    #   3    7    4                    1    2     1    7   17
    #   4    2    4     5              1    2     1    7   17
    #   5    6    6                    1    6     1    7   23
    # Setup.
    # Number of columns onto which applying aggregation function
    N_AGG_COLS = 1
    # Row indices of starts for each 'next chunk'.
    next_chunk_starts = array([0, 0, 3, 3, 5, 5, 6], dtype=INT64)
    # Last row provides number of rows in data.
    # 'len()' provides number of chunks, because in this case, there are only
    # snapshots, no bin.
    snap_res_n_rows = len(next_chunk_starts)
    snap_res = zeros((snap_res_n_rows, N_AGG_COLS), dtype=dtype_)
    n_nan_snaps = snap_res_n_rows - count_nonzero(ndiff(next_chunk_starts, prepend=0))
    # Array to store null snapshot indices is oversized. We cannot really know
    # in advance how many there will be. It is good idea to initialize it with
    # a negative value, as row index cannot be negative.
    potential_null_snap_indices = zeros(n_nan_snaps, dtype=INT64) - 1
    # All chunks are snapshots. 'bin_indices' is an empty array.
    bin_indices = zeros(0, dtype=INT64)
    # Define arrays for one type.
    data = array([1, 4, 3, 7, 2, 6], dtype=dtype_).reshape(-1, 1)
    # 'n_cols' is always of length of number of aggregation functions.
    n_cols = zeros(len(AGG_FUNC_IDS), dtype=INT64)
    # 1 column in data over which applying 'first' function.
    n_cols[agg_func_id] = N_AGG_COLS
    # Column indices for input data, and results, per aggregation function.
    # Indices are 0.
    cols = zeros((len(AGG_FUNC_IDS), N_AGG_COLS, 2), dtype=INT64)
    # No bin.
    agg_res = zeros(0, dtype=FLOAT64)
    null_bin_indices = zeros(0, dtype=INT64)
    # Test.
    _jitted_cgb2(
        data,
        n_cols,
        cols,
        next_chunk_starts,
        bin_indices,
        agg_res,
        snap_res,
        null_bin_indices,
        potential_null_snap_indices,
    )
    # Reference results.
    ref_res_ar = array(
        ref_res,
        dtype=dtype_,
    ).reshape(-1, 1)
    assert nall(ref_res_ar == snap_res)
    assert nall(potential_null_snap_indices == array([0, 1, -1, -1], dtype=INT64))


@pytest.mark.parametrize(
    "dtype_, agg_func_id, bin_indices_, ref_bin_res_, ref_snap_res_, ref_null_snap_indices_",
    [
        (
            FLOAT64,
            ID_FIRST,
            [1, 3, 5, 8, 11, 12, 16],
            [0, 1, 7, 0, 2, 0, 5],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 5, 0, 0],
            [0, 3, 4, 10, 11],
        ),
        (
            INT64,
            ID_FIRST,
            [1, 3, 5, 6, 11, 12, 17],
            [0, 1, 7, 0, 2, 0, 5],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 5, 5, 0],
            [0, 3, 4, 11, -1],
        ),
        (
            FLOAT64,
            ID_LAST,
            [1, 3, 5, 8, 11, 12, 16],
            [0, 3, 7, 0, 11, 0, 9],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 9, 0, 0],
            [0, 3, 4, 10, 11],
        ),
        (
            INT64,
            ID_LAST,
            [1, 3, 5, 8, 11, 12, 17],
            [0, 3, 7, 0, 11, 0, 9],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 9, 9, 0],
            [0, 3, 4, 11, -1],
        ),
        (
            FLOAT64,
            ID_MIN,
            [1, 3, 5, 8, 11, 12, 16],
            [0, 1, 7, 0, 2, 0, 5],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 5, 0, 0],
            [0, 3, 4, 10, 11],
        ),
        (
            INT64,
            ID_MIN,
            [1, 3, 5, 8, 11, 12, 17],
            [0, 1, 7, 0, 2, 0, 5],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 5, 5, 0],
            [0, 3, 4, 11, -1],
        ),
        (
            FLOAT64,
            ID_MAX,
            [1, 3, 5, 8, 11, 12, 16],
            [0, 4, 7, 0, 11, 0, 13],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 13, 0, 0],
            [0, 3, 4, 10, 11],
        ),
        (
            INT64,
            ID_MAX,
            [1, 3, 5, 8, 11, 12, 17],
            [0, 4, 7, 0, 11, 0, 13],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 13, 13, 0],
            [0, 3, 4, 11, -1],
        ),
        (
            FLOAT64,
            ID_SUM,
            [1, 3, 5, 8, 11, 12, 16],
            [0, 8, 7, 0, 19, 0, 27],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 27, 0, 0],
            [0, 3, 4, 10, 11],
        ),
        (
            INT64,
            ID_SUM,
            [1, 3, 5, 8, 11, 12, 17],
            [0, 8, 7, 0, 19, 0, 27],
            [0, 1, 7, 0, 0, 2, 2, 5, 5, 27, 27, 0],
            [0, 3, 4, 11, -1],
        ),
    ],
)
def test_jitted_cgb2_bin_snap_1d(
    dtype_, agg_func_id, bin_indices_, ref_bin_res_, ref_snap_res_, ref_null_snap_indices_
):
    # 1d data, with a single agg function. Mixing bins & snapshots.
    # This test case validates correct reset of snapshot each time a new bin
    # starts.
    # IDX  VAL  BIN    SNAP           EMPTY | FIRST  LAST  MIN    MAX   SUM    res for
    #                                       |  s| b  s| b  s| b  s| b  s| b    snaps
    #                                s0, b0 |
    #   0    1   b1      s1                 |  1     1     1     1     1       s1
    #   1    4   b1      s2                 |
    #   2    3   b1      s2                 |     1     3     1     4     8
    #   3    7   b2      s2      s3, s4, b3 |  7  7  7  7  7  7  7  7  7  7    s2
    #   4    2   b4      s5              s6 |  2     2     2     2     2       s5/6
    #   5    6   b4      s7                 |
    #   6   11   b4      s7              b5 |     2    11     2    11    19
    #   7    5   b6      s7              s8 |  5     5     5     5     5       s7/8
    #   8   13   b6      s9                 |
    #   9    9   b6      s9        s10, s11 |  5  5  9  9  5  5 13 13 27 27    s9
    #
    # Additionally,
    # - order of s3, s4, b3 and b4 is mixed and results have to be
    #   the same: empty chunk, because after b2.
    # - order of b6, s10, s11 is mixed, and results are not the same:
    #   - if s9, b6, s10, s11: s10 & s11 are empty snapshots
    #   - if s9, s10, s11, b6: s10 & s11 forward results from s9
    # Setup.
    # Number of columns onto which applying aggregation function
    N_AGG_COLS = 1
    # Define data array.
    data = array([1, 4, 3, 7, 2, 6, 11, 5, 13, 9], dtype=dtype_).reshape(-1, 1)
    # 'n_cols' is always of length of number of aggregation functions.
    n_cols = zeros(len(AGG_FUNC_IDS), dtype=INT64)
    # 1 column in data over which applying 'first' function.
    n_cols[agg_func_id] = N_AGG_COLS
    # Column indices for input data, and results, per aggregation function.
    # Indices are 0.
    cols = zeros((len(AGG_FUNC_IDS), N_AGG_COLS, 2), dtype=INT64)
    # Row indices of starts for each 'next chunk'.
    #                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14  15  16  17  18
    #                      s0 b0 s1 b1 s2 b2 s3 s4 b3 s5 s6 b4 b5 s7 s8  s9  b6 s10 s11
    # next_chunk_starts = [ 0, 0, 1, 3, 4, 4, 4, 4, 4, 5, 5, 7, 7, 8, 8, 10, 10, 10, 10]
    next_chunk_starts = array(
        [0, 0, 1, 3, 4, 4, 4, 4, 4, 5, 5, 7, 7, 8, 8, 10, 10, 10, 10], dtype=INT64
    )
    bin_indices = array(bin_indices_, dtype=INT64)
    # Initializing result arrays.
    # Snapshots.
    snap_res_n_rows = len(ref_snap_res_)
    snap_res = zeros((snap_res_n_rows, N_AGG_COLS), dtype=dtype_)
    # Array to store null snapshot indices is oversized. We cannot really know
    # in advance how many there will be. It is good idea to initialize it with
    # a negative value, as row index cannot be negative.
    n_nan_snaps = len(ref_null_snap_indices_)
    null_snap_indices = zeros(n_nan_snaps, dtype=INT64) - 1
    # Bins.
    bin_res_n_rows = len(ref_bin_res_)
    bin_res = zeros((bin_res_n_rows, N_AGG_COLS), dtype=dtype_)
    ref_null_bin_indices = array([0, 3, 5], dtype=INT64)
    null_bin_indices = zeros(len(ref_null_bin_indices), dtype=INT64)
    # Test.
    _jitted_cgb2(
        data,
        n_cols,
        cols,
        next_chunk_starts,
        bin_indices,
        bin_res,
        snap_res,
        null_bin_indices,
        null_snap_indices,
    )
    # Reference results.
    ref_bin_res = array(
        ref_bin_res_,
        dtype=dtype_,
    ).reshape(-1, 1)
    ref_snap_res = array(
        ref_snap_res_,
        dtype=dtype_,
    ).reshape(-1, 1)
    ref_null_snap_indices = array(
        ref_null_snap_indices_,
        dtype=INT64,
    )
    assert nall(ref_bin_res == bin_res)
    assert nall(ref_snap_res == snap_res)
    assert nall(null_bin_indices == ref_null_bin_indices)
    assert nall(null_snap_indices == ref_null_snap_indices)


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jitted_cgb2_bin_snap_2d(dtype_):
    # 2d data, with all agg functions, mixing bins & snapshots.
    # This test case is to validate the iteration on column indices,
    # i.e. use of 'cols' parameter.
    # IDX  C0 C1 C2 BIN SNAP    EMPTY | F_C0  F_C2  L_C1  L_C2  MIC0  MIC2  MAC0  MAC1 SUMC1 SUMC2  res for
    #                                 | s| b  s| b  s| b  s| b  s| b  s| b  s| b  s| b  s| b  s| b  snaps
    #                          s0, b0 |
    #   0   1  9  5  b1   s1          | 1     5     9     5     1     5     1     9     9     5     s1
    #   1   4  7 11  b1   s2          |
    #   2   3  3 17  b1   s2          |
    #   3   7 12 19  b1   s2   b2, b3 | 1  1  5  5 12 12 19 19  1  1  5  5  7  7 12 12 31 31 52 52  s2
    #   4   8  5  6  b4   s3       s4 | 8     6     5     6     8     6     8     5     5     6     s3/4
    #   5   2 17  7  b4   s5          |
    #   6  11  8 13  b4   s5          | 8     6     8    13     2     6    11    17    30    26
    #   7   5 20  4  b4   s6       s7 | 8  6  6  6 20 20  4 4   2  2  4  4 11 11 20 20 50 50 30 30  s6/7
    #   8  13  4 10  b5   s8          |
    #   9   9  2  9  b5   s8  s9, s10 |13 13 10 10  2  2  9 9   9  9  9  9 13 13  4  4  6  6 19 19  s8/9
    #
    # Setup.
    # Max number of columns onto which applying agg functions (per agg func).
    N_AGG_COLS = 2
    # Define data array.
    data = array(
        [
            [1, 9, 5],
            [4, 7, 11],
            [3, 3, 17],
            [7, 12, 19],
            [8, 5, 6],
            [2, 17, 7],
            [11, 8, 13],
            [5, 20, 4],
            [13, 4, 10],
            [9, 2, 9],
        ],
        dtype=dtype_,
    )
    # Number of columns in data over which applying each agg functions.
    # 'n_cols' is always of length of number of aggregation functions.
    n_cols = zeros(len(AGG_FUNC_IDS), dtype=INT64) + N_AGG_COLS
    # Column indices for input data, and results, per aggregation function.
    # [[data, res], [data, res]]
    cols = array(
        [
            [[0, 0], [2, 1]],  # FIRST
            [[1, 2], [2, 3]],  # LAST
            [[0, 4], [2, 5]],  # MIN
            [[0, 6], [1, 7]],  # MAX
            [[1, 8], [2, 9]],  # SUM
        ],
        dtype=INT64,
    )
    # Row indices of starts for each 'next chunk'.
    #                       0  1  2  3  4  5  6  7  8  9 10 11 12  13  14  15  16
    #                      s0 b0 s1 s2 b1 b2 b3 s3 s4 s5 s6 s7 b4  s8  s9  b5 s10
    # next_chunk_starts = [ 0, 0, 1, 4, 4, 4, 4, 5, 5, 7, 8, 8, 8, 10, 10, 10, 10]
    next_chunk_starts = array([0, 0, 1, 4, 4, 4, 4, 5, 5, 7, 8, 8, 8, 10, 10, 10, 10], dtype=INT64)
    bin_indices = array([1, 4, 5, 6, 12, 15], dtype=INT64)
    # Initializing reference result arrays.
    # Snapshots.
    snap_ref_res = array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s0
            [1, 5, 9, 5, 1, 5, 1, 9, 9, 5],  # s1
            [1, 5, 12, 19, 1, 5, 7, 12, 31, 52],  # s2
            [8, 6, 5, 6, 8, 6, 8, 5, 5, 6],  # s3
            [8, 6, 5, 6, 8, 6, 8, 5, 5, 6],  # s4
            [8, 6, 8, 13, 2, 6, 11, 17, 30, 26],  # s5
            [8, 6, 20, 4, 2, 4, 11, 20, 50, 30],  # s6
            [8, 6, 20, 4, 2, 4, 11, 20, 50, 30],  # s7
            [13, 10, 2, 9, 9, 9, 13, 4, 6, 19],  # s8
            [13, 10, 2, 9, 9, 9, 13, 4, 6, 19],  # s9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # s10
        ],
        dtype=dtype_,
    )
    snap_res = zeros(snap_ref_res.shape, dtype=dtype_)
    # Array to store null snapshot indices is oversized. We cannot really know
    # in advance how many there will be. It is good idea to initialize it with
    # a negative value, as row index cannot be negative.
    ref_null_snap_indices = array([0, 10, -1], dtype=INT64)
    null_snap_indices = zeros(len(ref_null_snap_indices), dtype=INT64) - 1
    # Bins.
    bin_ref_res = array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b0
            [1, 5, 12, 19, 1, 5, 7, 12, 31, 52],  # b1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b2
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # b3
            [8, 6, 20, 4, 2, 4, 11, 20, 50, 30],  # b4
            [13, 10, 2, 9, 9, 9, 13, 4, 6, 19],  # b5
        ],
        dtype=dtype_,
    )
    bin_res = zeros(bin_ref_res.shape, dtype=dtype_)
    ref_null_bin_indices = array([0, 2, 3], dtype=INT64)
    null_bin_indices = zeros(len(ref_null_bin_indices), dtype=INT64)
    # Test.
    _jitted_cgb2(
        data,
        n_cols,
        cols,
        next_chunk_starts,
        bin_indices,
        bin_res,
        snap_res,
        null_bin_indices,
        null_snap_indices,
    )
    assert nall(bin_ref_res == bin_res)
    assert nall(snap_ref_res == snap_res)
    assert nall(null_bin_indices == ref_null_bin_indices)
    assert nall(null_snap_indices == ref_null_snap_indices)


@pytest.mark.parametrize(
    "dtype_, agg_func1, agg_func2, agg_func3, res_ref",
    [
        (
            FLOAT64,
            FIRST,
            MIN,
            LAST,
            [[1, 3, 2, 1, 2], [0, 0, 0, 0, 0], [7, 9, 8, 1, 8], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
        ),
        (
            FLOAT64,
            LAST,
            MAX,
            SUM,
            [[3, 1, 5, 6, 9], [0, 0, 0, 0, 0], [2, 1, 8, 9, 16], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
        ),
        (
            INT64,
            FIRST,
            MIN,
            LAST,
            [[1, 3, 2, 1, 2], [0, 0, 0, 0, 0], [7, 9, 8, 1, 8], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
        ),
        (
            INT64,
            LAST,
            MAX,
            SUM,
            [[3, 1, 5, 6, 9], [0, 0, 0, 0, 0], [2, 1, 8, 9, 16], [0, 0, 0, 0, 0], [6, 5, 4, 5, 4]],
        ),
    ],
)
def test_jitted_cgb_2d(dtype_, agg_func1, agg_func2, agg_func3, res_ref):
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
        dtype=dtype_,
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
    agg_res = zeros((agg_res_n_rows, agg_res_n_cols), dtype=dtype_)
    # Test.
    _jitted_cgb(
        group_sizes,
        data,
        agg_func,
        agg_func_n_cols,
        agg_cols_in_data,
        agg_cols_in_res,
        True,
        agg_res,
        nan_group_indices_res,
    )
    # Ref. results.
    ref_res = array(
        res_ref,
        dtype=dtype_,
    )
    assert nall(ref_res == agg_res)
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


def test_chaingroupby_mixed_dtype():
    # Test aggregation for a mixed dtype.
    ar_float = array(
        [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]], dtype=FLOAT64
    )
    ar_int = array([[1, 10], [3, 30], [6, 60], [7, 70], [9, 90]], dtype=INT64)
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
    agg_res = chaingroupby(by=by, agg=agg, data=data)
    agg_res_ref = data.groupby(by).agg(**agg)
    assert agg_res.equals(agg_res_ref)


# WiP
# Test with null values in agg_res (or modify test case above)
# Test error message if 'bin_on' is None in 'chaingroupby'.
# test error message input column in 'agg' not in input dataframe.

# Test case snapshot with a snapshot ending exactly on a bin end,
# and another not ending on a bin end.
# Snapshot test case with null rows (empty snapshots)
# Have a snapshot ending exactly on a bin end, then a next snapshot bigger than next bin
