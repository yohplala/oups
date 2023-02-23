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
from numpy import zeros

from oups.jcumsegagg import AGG_FUNC_IDS
from oups.jcumsegagg import ID_FIRST
from oups.jcumsegagg import ID_LAST
from oups.jcumsegagg import ID_MAX
from oups.jcumsegagg import ID_MIN
from oups.jcumsegagg import ID_SUM
from oups.jcumsegagg import jcsagg
from oups.jcumsegagg import jmax
from oups.jcumsegagg import jmin
from oups.jcumsegagg import jsum


INT64 = "int64"
FLOAT64 = "float64"


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jmax(dtype_):
    # Test 'jmax()'.
    ar = array([], dtype=dtype_)
    assert jmax(ar, None) is None
    assert jmax(ar, 3) == 3
    ar = array([2], dtype=dtype_)
    assert jmax(ar, None) == 2
    assert jmax(ar, 5) == 5
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jmax(ar, None) == 3
    assert jmax(ar, 4) == 4


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jmin(dtype_):
    # Test 'jmin()'.
    ar = array([], dtype=dtype_)
    assert jmin(ar, None) is None
    assert jmin(ar, 3) == 3
    ar = array([8], dtype=dtype_)
    assert jmin(ar, None) == 8
    assert jmin(ar, 5) == 5
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jmin(ar, None) == -1
    assert jmin(ar, -3) == -3


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jsum(dtype_):
    # Test 'jsum()'.
    ar = array([], dtype=dtype_)
    assert jsum(ar, None) is None
    assert jsum(ar, 3) == 3
    ar = array([8], dtype=dtype_)
    assert jsum(ar, None) == 8
    assert jsum(ar, 5) == 13
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jsum(ar, None) == 5
    assert jsum(ar, -3) == 2


@pytest.mark.parametrize(
    "dtype_, agg_func_id, ref_res",
    [
        (FLOAT64, ID_FIRST, [1, 7, 6]),
        (FLOAT64, ID_LAST, [3, 2, 6]),
        (FLOAT64, ID_MIN, [1, 2, 6]),
        (FLOAT64, ID_MAX, [4, 7, 6]),
        (FLOAT64, ID_SUM, [8, 9, 6]),
        (INT64, ID_FIRST, [1, 7, 6]),
        (INT64, ID_LAST, [3, 2, 6]),
        (INT64, ID_MIN, [1, 2, 6]),
        (INT64, ID_MAX, [4, 7, 6]),
        (INT64, ID_SUM, [8, 9, 6]),
    ],
)
def test_jcsagg_bin_1d_no_void(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. Only bins. Only full bins
    # This test case essentially validates aggregation functions in a sequence
    # of bins.
    # IDX  VAL  BIN  EMPTY_BIN   FIRST  LAST  MIN  MAX  SUM
    #   0    1    1                  1    3     1    4    8
    #   1    4    1
    #   2    3    1
    #   3    7    2                  7    2     2    7    9
    #   4    2    2
    #   5    6    3                  6    6     6    6    6
    # Setup.
    # Number of columns onto which applying aggregation function
    N_AGG_COLS = 1
    # Row indices of starts for each 'next chunk'.
    next_chunk_starts = array([3, 5, 6], dtype=INT64)
    # Last row provides number of rows in data.
    # 'len()' provides number of chunks, because in this case, there are only
    # bins, no snapshot.
    bin_res_n_rows = len(next_chunk_starts)
    bin_res = zeros((bin_res_n_rows, N_AGG_COLS), dtype=dtype_)
    null_bin_indices = zeros(0, dtype=INT64)
    # All chunks are bins.
    bin_indices = array(range(bin_res_n_rows), dtype=INT64)
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
    snap_res = zeros(0, dtype=FLOAT64).reshape(0, 0)
    null_snap_indices = zeros(0, dtype=INT64)
    # Test.
    jcsagg(
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
    ref_res_ar = array(
        ref_res,
        dtype=dtype_,
    ).reshape(-1, 1)
    assert nall(ref_res_ar == bin_res)
    assert nall(null_bin_indices == zeros(0, dtype=INT64))
    assert nall(snap_res == zeros(0, dtype=FLOAT64).reshape(0, 0))
    assert nall(null_snap_indices == zeros(0, dtype=INT64))


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
def test_jcsagg_bin_1d_with_void(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. Only bins.
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
    bin_res_n_rows = len(next_chunk_starts)
    bin_res = zeros((bin_res_n_rows, N_AGG_COLS), dtype=dtype_)
    n_nan_bins = bin_res_n_rows - count_nonzero(ndiff(next_chunk_starts, prepend=0))
    null_bin_indices = zeros(n_nan_bins, dtype=INT64)
    # All chunks are bins.
    bin_indices = array(range(bin_res_n_rows), dtype=INT64)
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
    snap_res = zeros(0, dtype=FLOAT64).reshape(0, 0)
    null_snap_indices = zeros(0, dtype=INT64)
    # Test.
    jcsagg(
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
    ref_res_ar = array(
        ref_res,
        dtype=dtype_,
    ).reshape(-1, 1)
    assert nall(ref_res_ar == bin_res)
    assert nall(null_bin_indices == array([0, 1, 3, 5], dtype=INT64))
    assert nall(snap_res == zeros(0, dtype=FLOAT64).reshape(0, 0))
    assert nall(null_snap_indices == zeros(0, dtype=INT64))


@pytest.mark.parametrize(
    "dtype_, agg_func_id, ref_res",
    [
        (FLOAT64, ID_FIRST, [1, 1, 1]),
        (FLOAT64, ID_LAST, [3, 2, 6]),
        (FLOAT64, ID_MIN, [1, 1, 1]),
        (FLOAT64, ID_MAX, [4, 7, 7]),
        (FLOAT64, ID_SUM, [8, 17, 23]),
        (INT64, ID_FIRST, [1, 1, 1]),
        (INT64, ID_LAST, [3, 2, 6]),
        (INT64, ID_MIN, [1, 1, 1]),
        (INT64, ID_MAX, [4, 7, 7]),
        (INT64, ID_SUM, [8, 17, 23]),
    ],
)
def test_jcsagg_snap_1d_no_void(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. Only snapshots.
    # This test case essentially validates aggregation functions in a sequence
    # of snapshots.
    # IDX  VAL  SNAP  EMPTY_SNAP   FIRST  LAST  MIN  MAX  SUM
    #   0    1    1                    1    3     1    4    8
    #   1    4    1
    #   2    3    1
    #   3    7    2                    1    2     1    7   17
    #   4    2    2
    #   5    6    3                    1    6     1    7   23
    # Setup.
    # Number of columns onto which applying aggregation function
    N_AGG_COLS = 1
    # Row indices of starts for each 'next chunk'.
    next_chunk_starts = array([3, 5, 6], dtype=INT64)
    # Last row provides number of rows in data.
    # 'len()' provides number of chunks, because in this case, there are only
    # snapshots, no bin.
    snap_res_n_rows = len(next_chunk_starts)
    snap_res = zeros((snap_res_n_rows, N_AGG_COLS), dtype=dtype_)
    null_snap_indices = zeros(0, dtype=INT64)
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
    bin_res = zeros(0, dtype=FLOAT64).reshape(0, 0)
    null_bin_indices = zeros(0, dtype=INT64)
    # Test.
    jcsagg(
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
    ref_res_ar = array(
        ref_res,
        dtype=dtype_,
    ).reshape(-1, 1)
    assert nall(ref_res_ar == snap_res)
    assert nall(null_snap_indices == zeros(0, dtype=INT64))
    assert nall(bin_res == zeros(0, dtype=FLOAT64).reshape(0, 0))
    assert nall(null_bin_indices == zeros(0, dtype=INT64))


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
def test_jcsagg_snap_1d_with_void(dtype_, agg_func_id, ref_res):
    # 1d data, with a single agg function. Only snapshots.
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
    bin_res = zeros(0, dtype=FLOAT64).reshape(0, 0)
    null_bin_indices = zeros(0, dtype=INT64)
    # Test.
    jcsagg(
        data,
        n_cols,
        cols,
        next_chunk_starts,
        bin_indices,
        bin_res,
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
    assert nall(bin_res == zeros(0, dtype=FLOAT64).reshape(0, 0))
    assert nall(null_bin_indices == zeros(0, dtype=INT64))


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
def test_jcsagg_bin_snap_1d(
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
    jcsagg(
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
def test_jcsagg_bin_snap_2d(dtype_):
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
    jcsagg(
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
