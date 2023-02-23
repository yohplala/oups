#!/usr/bin/env python3
"""
Created on Fri Dec 22 19:00:00 2022.

@author: yoh
"""

from numba import float64
from numba import int64
from numba import jit
from numba import optional
from numpy import dtype
from numpy import ndarray
from numpy import ndenumerate
from numpy import zeros


# Aggregation function ids.
FIRST = "first"
LAST = "last"
MIN = "min"
MAX = "max"
SUM = "sum"
ID_FIRST = 0
ID_LAST = 1
ID_MIN = 2
ID_MAX = 3
ID_SUM = 4
AGG_FUNC_IDS = {FIRST: ID_FIRST, LAST: ID_LAST, MIN: ID_MIN, MAX: ID_MAX, SUM: ID_SUM}


@jit(
    [optional(int64)(int64[:], optional(int64)), optional(float64)(float64[:], optional(float64))],
    nopython=True,
)
def jmax(ar: ndarray, initial=None):
    """Jitted max.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which retrieving the max value.
    initial : Union[float, int, None]
        Include this value in the max evaluation.
        Because the function is jitted, `None` should be explicitly
        mentioned.

    Returns
    -------
    Union[float, int, None]
        Max value within 'ar'.
    """
    len_ar = len(ar)
    if len_ar > 0:
        max_ = ar[0]
        if len_ar > 1:
            for val in ar[1:]:
                if val > max_:
                    max_ = val
        if initial is None:
            return max_
        else:
            if initial > max_:
                return initial
            else:
                return max_
    return initial


@jit(
    [optional(int64)(int64[:], optional(int64)), optional(float64)(float64[:], optional(float64))],
    nopython=True,
)
def jmin(ar: ndarray, initial=None):
    """Jitted min.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which retrieving the min value.
    initial : Union[float, int, None]
        Include this value in the min evaluation.
        Because the function is jitted, `None` should be explicitly
        mentioned.

    Returns
    -------
    Union[float, int, None]
        Min value within 'ar'.
    """
    len_ar = len(ar)
    if len_ar > 0:
        min_ = ar[0]
        if len_ar > 1:
            for val in ar[1:]:
                if val < min_:
                    min_ = val
        if initial is None:
            return min_
        else:
            if initial < min_:
                return initial
            else:
                return min_
    return initial


@jit(
    [optional(int64)(int64[:], optional(int64)), optional(float64)(float64[:], optional(float64))],
    nopython=True,
)
def jsum(ar: ndarray, initial=None):
    """Jitted sum.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which assessing the sum.
    initial : Union[float, int, None]
        Include this value in the sum.
        Because the function is jitted, `None` should be explicitly
        mentioned.

    Returns
    -------
    Union[float, int, None]
        Sum of values in 'ar'.
    """
    len_ar = len(ar)
    if len_ar > 0:
        sum_ = ar[0]
        if len_ar > 1:
            for val in ar[1:]:
                sum_ += val
        if initial is None:
            return sum_
        else:
            return sum_ + initial
    return initial


@jit(
    [optional(int64)(int64[:], optional(int64)), optional(float64)(float64[:], optional(float64))],
    nopython=True,
)
def jfirst(ar: ndarray, initial=None):
    """Jitted first.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array from which taking first value.
    initial : Union[float, int, None]
        Consider this value as previous first.
        Because the function is jitted, `None` should be explicitly
        mentioned.

    Returns
    -------
    Union[float, int, None]
        First in 'ar' if 'initial' is not provided, else 'initial'.
    """
    len_ar = len(ar)
    if initial is None and len_ar > 0:
        return ar[0]
    else:
        return initial


@jit(
    [optional(int64)(int64[:], optional(int64)), optional(float64)(float64[:], optional(float64))],
    nopython=True,
)
def jlast(ar: ndarray, initial=None):
    """Jitted last.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array from which taking last value.
    initial : Union[float, int, None]
        Consider this value as previous last.
        This value is used if 'ar' is a null array.
        Because the function is jitted, `None` should be explicitly
        mentioned.

    Returns
    -------
    Union[float, int, None]
        Last in 'ar' if 'ar' is not a null array.
    """
    len_ar = len(ar)
    if len_ar > 0:
        return ar[-1]
    else:
        return initial


AGG_FUNCS = {ID_FIRST: jfirst, ID_LAST: jlast, ID_MIN: jmin, ID_MAX: jmax, ID_SUM: jsum}


def _jcsa_setup(n_cols: ndarray, cols: ndarray, dtype_: dtype):
    """
    Return setup for the set of cumulative segmented aggregation functions.

    Parameters
    ----------
    n_cols : np.ndarray
        1d array, specifying per aggregation function how many columns in 'data'
        have to be aggregated with this function.
    cols: np.ndarray
        3d array, with one row per aggregation function.
        Per aggregation function, a 2d array containing the list of twin column
        indices onto which applying the aggregation function. The 1st index
        is the index of column in 'data', the 2nd index is the index of column
        in 'res'.
    dtype_ : dtype
        data array ``dtype``.

    Returns
    -------
    bool, Union[np.ndarray, None], Union[array, None]

        - bool: if the aggregation function has to be assessed.
        - Union[array, None]: array of column indices to retrieve data and store
          aggregation results.
        - Union[array, None]: array to use as buffer between loop iteration.
        - Union[array, None]: array to use as buffer for the calculations.

    """
    agg_func_config = []
    for agg_id, n_cols_ in enumerate(n_cols):
        if n_cols_ > 0:
            func = AGG_FUNCS[agg_id]
            cols_ = cols[agg_id, :n_cols_, :]
            agg_func_config.append((agg_id, func, cols_))
    buffer = zeros(cols.shape[:2], dtype=dtype_)
    res_loc = zeros(cols.shape[:2], dtype=dtype_)
    return tuple(agg_func_config), buffer, res_loc


def jcsagg(
    data: ndarray,  # 2d
    n_cols: ndarray,  # 1d
    cols: ndarray,  # 3d
    next_chunk_starts: ndarray,  # 1d
    bin_indices: ndarray,  # 1d
    bin_res: ndarray,  # 2d
    snap_res: ndarray,  # 2d
    null_bin_indices: ndarray,  # 1d
    null_snap_indices: ndarray,  # 1d
):
    """Group assuming contiguity.

    Parameters
    ----------
    data : np.ndarray
        Array over which performing aggregation functions.
    n_cols : np.ndarray
        One dimensional array of ``int``, specifying per aggregation function
        the number of columns to which applying related aggregation function
        (and consequently the number of columns in 'bin_res' to which recording
        the aggregation results).
    cols : np.ndarray
        Three dimensional array of ``int``, one row per aggregation function.
        Per row (2nd dimension), column indices in 'data' to which apply
        corresponding aggregation function.
        Any value in column past the number of relevant columns is not used.
        In last dimension, index 0 gives indices of columns in 'data'. Index 1
        gives indices of columns in 'xxx_res'.
    next_chunk_starts : np.ndarray
        Ordered one dimensional array of ``int``, indicating the index of the
        1st row of next chunk (or last row index of current chunk, excluded).
        May contain duplicates, indicating, depending the chunk type, possibly
        an empty bin or an empty snapshot.
    bin_indices : np.ndarray
        Sorted, one dimensional array of ``int``, of same size than the number
        of bins, and indicating that a chunk at this index in
        'next_chunk_starts' is a bin (and not a snapshot). Beware that it has
        to contain no duplicate values.
        In case of no snapshotting ('snap_res' is a null array), then
        'bin_indices' can be a null array.

    Returns
    -------
    bin_res : np.ndarray
        Results from aggregation, with same `dtype` than 'data' array, for
        bins.
    snap_res : np.ndarray
        Results from aggregation, with same `dtype` than 'data' array
        considering intermediate snapshots.
    null_bin_indices : np.ndarray
        One dimensional array containing row indices in 'bin_res' that
        correspond to "empty" bins, i.e. for which bin size has been set to 0.
    null_snap_indices : np.ndarray
        One dimensional array containing row indices in 'snap_res' that
        correspond to "empty" snapshots, i.e. for which snapshot size has been
        set to 0. Input array should be set to null values, so that unused
        rows can be identified clearly.
    """
    # Setup agg func constants.
    agg_func_config, buffer, res_loc = _jcsa_setup(n_cols, cols, data.dtype)
    # 'last_rows' is an array of `int`, providing the index of last row for
    # each chunk.
    # If a 'snapshot' chunk shares same last row than a 'bin' chunk, the
    # 'snapshot' is expected to be listed prior to the 'bin' chunk.
    # A 'snapshot' is an 'update'. A 'bin' is a 'reset'.
    bin_start = chunk_start = 0
    bin_res_idx = snap_res_idx = 0
    null_bin_idx = null_snap_idx = 0
    # 'pinnu' was 'prev_is_non_null_update'.
    # With 'pinnu' True, then cumulate (pass-through) previous results.
    pinnu = False
    if len(snap_res) != 0:
        # Case 'snapshots expected'.
        # Setup identification of bins (vs snapshots).
        some_snaps = True
        if len(bin_indices) > 0:
            # Case 'there are bins'.
            iter_bin_indices = iter(bin_indices)
            next_bin_idx = next(iter_bin_indices)
            last_bin_idx = bin_indices[-1]
        else:
            # Case 'only snapshots'.
            next_bin_idx = -1
    else:
        # Case 'no snapshot expected'.
        some_snaps = False
        is_update = False
    for (idx,), next_chunk_start in ndenumerate(next_chunk_starts):
        if some_snaps:
            # Is the current 'next_chunk_start' idx that of a 'bin' or that of
            # a 'snapshot'?
            if idx == next_bin_idx:
                # Case 'bin'.
                is_update = False
                if next_bin_idx != last_bin_idx:
                    next_bin_idx = next(iter_bin_indices)
            else:
                # Case 'snapshot'.
                is_update = True
        # Null chunk is identified if no new data since start of 'bin' whatever
        # 'bin' or 'snapshot' (update).
        # An update without any row is not necessarily a null update.
        # Values from past update may need to be forwarded.
        if bin_start == next_chunk_start:
            # Null chunk since the start of the bin.
            if is_update:
                null_snap_indices[null_snap_idx] = snap_res_idx
                null_snap_idx += 1
                snap_res_idx += 1
            else:
                null_bin_indices[null_bin_idx] = bin_res_idx
                null_bin_idx += 1
                bin_res_idx += 1
                pinnu = False
        else:
            # Chunk with some rows from the start of the bin.
            data_loc = data[chunk_start:next_chunk_start]
            # for conf in literal_unroll(red_func_config):
            for conf in agg_func_config:
                agg_id, func, cols = conf
                if pinnu:
                    if len(data_loc) == 0:
                        # Forward past results.
                        res_loc[agg_id, :] = buffer[agg_id, :]
                    else:
                        for col_buffer, (col_data, _) in enumerate(cols):
                            res_loc[agg_id, col_buffer] = func(
                                data_loc[:, col_data], initial=buffer[agg_id, col_buffer]
                            )
                else:
                    for col_buffer, (col_data, _) in enumerate(cols):
                        res_loc[agg_id, col_buffer] = func(data_loc[:, col_data], initial=None)
            if is_update:
                # Record result in 'snap_res'.
                for conf in agg_func_config:
                    agg_id, func, cols = conf
                    for col_buffer, (_, col_res) in enumerate(cols):
                        # Case of 'snapshot': update buffer
                        snap_res[snap_res_idx, col_res] = buffer[agg_id, col_buffer] = res_loc[
                            agg_id, col_buffer
                        ]
                snap_res_idx += 1
                pinnu = True
            else:
                # Record results in 'bin_res'.
                # No need to update related buffer, as it is end of bin.
                for conf in agg_func_config:
                    agg_id, func, cols = conf
                    for col_buffer, (_, col_res) in enumerate(cols):
                        # Case of 'bin'
                        bin_res[bin_res_idx, col_res] = res_loc[agg_id, col_buffer]
                bin_res_idx += 1
                bin_start = next_chunk_start
                pinnu = False
        chunk_start = next_chunk_start
