#!/usr/bin/env python3
"""
Created on Fri Dec 22 19:00:00 2022.

@author: yoh
"""
from typing import Callable

from numba import jit_module
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


def jmax(ar: ndarray, initial=None):
    """
    Jitted max.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which retrieving the max value.
    initial : Union[float, int], optional
        Include this value in the max evaluation. The default is None.

    Returns
    -------
    Union[float, int]
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
    elif initial:
        return initial


def jmin(ar: ndarray, initial=None):
    """
    Jitted min.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which retrieving the min value.
    initial : Union[float, int], optional
        Include this value in the min evaluation. The default is None.

    Returns
    -------
    Union[float, int]
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
    elif initial:
        return initial


def jsum(ar: ndarray, initial=None):
    """
    Jitted sum.

    Parameters
    ----------
    ar : np.ndarray
        Numpy array over which assessing the sum.
    initial : Union[float, int], optional
        Include this value in the sum. The default is None.

    Returns
    -------
    Union[float, int]
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
    elif initial:
        return initial


def jcumagg(
    agg: Callable,
    cumulate: bool,
    chunk_start: int,
    next_chunk_start: int,
    cols: ndarray,
    data: ndarray,
    res_idx: int,
    res: ndarray,
    buffer: ndarray,
    update_buffer: bool,
):
    """
    Jitted cumulative aggregation (was 'snapshot').

    Parameters
    ----------
    agg : Callable
        Jitted reduction function.
    cumulate : bool
        If aggregation has to cumulate with previous aggregation results
        (stored in buffer).
    chunk_start : int
        Row index for start of chunk in data.
    next_chunk_start : int
        Row index for start of next chunk in data.
    cols : np.ndarray
        2d array of ``int``, with as many rows as columns onto which
        aggregating. Each row contains 2 values, the index of the column in
        'data', and the index of the column in 'res'.
    data : np.ndarray
        2d array. Contiguous segments of rows are 'chunks'.
    res_idx : int
        Row index in 'res' to use for storing aggregation results.
    res : np.ndarray (out)
        2d array of results from the cumulative aggregation.
    buffer : np.ndarray (in/out)
        Buffer array, to re-use results of previous cumulative aggregations for
        current chunk.
    update_buffer : bool
        In case it is an "on-going" cumulative aggregation, buffer needs to be
        updated. If this is the last segment of a cumulative aggregation,
        update of buffer is useless. This can be de-activated, setting this
        parameter to ``False``.
    """
    if cumulate:
        if chunk_start == next_chunk_start:
            # Forward past results.
            for col_buffer, (_, col_res) in enumerate(cols):
                res[res_idx, col_res] = buffer[col_buffer]
        else:
            # Cumulate.
            if update_buffer:
                # Case of 'snapshot'.
                for col_buffer, (col_data, col_res) in enumerate(cols):
                    res[res_idx, col_res] = buffer[col_buffer] = agg(
                        data[chunk_start:next_chunk_start, col_data],
                        initial=buffer[col_buffer],
                    )
            else:
                # Case of 'bin'.
                for col_buffer, (col_data, col_res) in enumerate(cols):
                    res[res_idx, col_res] = agg(
                        data[chunk_start:next_chunk_start, col_data],
                        initial=buffer[col_buffer],
                    )
    else:
        if update_buffer:
            # Case of 'snapshot'.
            # 1st aggregation: need to initialize 'buffer'.
            for col_buffer, (col_data, col_res) in enumerate(cols):
                res[res_idx, col_res] = buffer[col_buffer] = agg(
                    data[chunk_start:next_chunk_start, col_data]
                )
        else:
            # Case of 'bin'.
            for col_data, col_res in cols:
                res[res_idx, col_res] = agg(data[chunk_start:next_chunk_start, col_data])


def jrowat(
    from_buffer: bool,
    data_idx: int,
    cols: ndarray,
    data: ndarray,
    res_idx: int,
    res: ndarray,
    buffer: ndarray,
    update_buffer: bool,
):
    """
    Jitted row picking.

    Parameters
    ----------
    from_buffer : bool
        If value stored in buffer has to be used instead.
    data_idx : int
        row index in 'data' from which retrieving the value.
    cols : np.ndarray
        2d array of ``int``, with as many rows as columns onto which
        aggregating. Each row contains 2 values, the index of the column in
        'data', and the index of the column in 'res'.
    data : np.ndarray
        2d array. Contiguous segments of rows are 'chunks'.
    res_idx : int
        Row index in 'res' to use for storing aggregation results.
    res : np.ndarray (out)
        2d array of results.
    buffer : np.ndarray (in/out)
        Buffer array, to re-use results of previous cumulative aggregations for
        current chunk.
    update_buffer : bool
        In case it is an "on-going" cumulative aggregation, buffer needs to be
        updated. If this is the last segment of a cumulative aggregation,
        update of buffer is useless. This can be de-activated, setting this
        parameter to ``False``.
    """
    if from_buffer:
        for col_buffer, (_, col_res) in enumerate(cols):
            res[res_idx, col_res] = buffer[col_buffer]
    else:
        if update_buffer:
            # Case of 'snapshot'.
            for col_buffer, (col_data, col_res) in enumerate(cols):
                res[res_idx, col_res] = buffer[col_buffer] = data[data_idx, col_data]
        else:
            # Case of 'bin'.
            for col_data, col_res in cols:
                res[res_idx, col_res] = data[data_idx, col_data]


def _jcsa_setup(agg_id: int, n_cols: ndarray, cols: ndarray, dtype_: dtype):
    """
    Return setup for one cumulative segmented aggregation function.

    Parameters
    ----------
    agg_id : int
        Id of aggregation function.
    n_cols : np.ndarray
        1d array, specifying per aggrgation function how many columns in 'data'
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
        - Union[array, None]: array to use as buffer.

    """
    n_cols_ = n_cols[agg_id]
    if n_cols_ != 0:
        assess = True
        cols = cols[agg_id, :n_cols_, :]
        buffer = zeros(n_cols_, dtype=dtype_)
        return assess, cols, buffer
    else:
        return False, None, None


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
    assess_FIRST, cols_FIRST, buffer_FIRST = _jcsa_setup(ID_FIRST, n_cols, cols, data.dtype)
    assess_LAST, cols_LAST, buffer_LAST = _jcsa_setup(ID_LAST, n_cols, cols, data.dtype)
    assess_MIN, cols_MIN, buffer_MIN = _jcsa_setup(ID_MIN, n_cols, cols, data.dtype)
    assess_MAX, cols_MAX, buffer_MAX = _jcsa_setup(ID_MAX, n_cols, cols, data.dtype)
    assess_SUM, cols_SUM, buffer_SUM = _jcsa_setup(ID_SUM, n_cols, cols, data.dtype)
    # 'last_rows' is an array of `int`, providing the index of last row for
    # each chunk.
    # If a 'snapshot' chunk shares same last row than a 'bin' chunk, the
    # 'snapshot' is expected to be listed prior to the 'bin' chunk.
    # A 'snapshot' is an 'update'. A 'bin' is a 'reset'.
    bin_start = chunk_start = 0
    bin_res_idx = snap_res_idx = 0
    null_bin_idx = null_snap_idx = 0
    # 'pinnu' was 'prev_is_non_null_update'.
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
            if is_update:
                # Make an update and record result in 'snap_res'.
                if assess_FIRST:
                    jrowat(
                        pinnu,
                        chunk_start,
                        cols_FIRST,
                        data,
                        snap_res_idx,
                        snap_res,
                        buffer_FIRST,
                        True,
                    )
                if assess_LAST:
                    # If current chunk is empty, values from buffer are used.
                    jrowat(
                        chunk_start == next_chunk_start,
                        next_chunk_start - 1,
                        cols_LAST,
                        data,
                        snap_res_idx,
                        snap_res,
                        buffer_LAST,
                        True,
                    )
                if assess_MIN:
                    jcumagg(
                        jmin,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_MIN,
                        data,
                        snap_res_idx,
                        snap_res,
                        buffer_MIN,
                        True,
                    )
                if assess_MAX:
                    jcumagg(
                        jmax,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_MAX,
                        data,
                        snap_res_idx,
                        snap_res,
                        buffer_MAX,
                        True,
                    )
                if assess_SUM:
                    jcumagg(
                        jsum,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_SUM,
                        data,
                        snap_res_idx,
                        snap_res,
                        buffer_SUM,
                        True,
                    )
                snap_res_idx += 1
                pinnu = True
            else:
                # Record result in 'bin_res'.
                # For these 'standard' aggregations', re-using results from previous updates,
                # no need to update related buffer, as it is end of bin.
                if assess_FIRST:
                    jrowat(
                        pinnu,
                        chunk_start,
                        cols_FIRST,
                        data,
                        bin_res_idx,
                        bin_res,
                        buffer_FIRST,
                        False,
                    )
                if assess_LAST:
                    # If current chunk is empty, values from buffer are used.
                    jrowat(
                        chunk_start == next_chunk_start,
                        next_chunk_start - 1,
                        cols_LAST,
                        data,
                        bin_res_idx,
                        bin_res,
                        buffer_LAST,
                        False,
                    )
                if assess_MIN:
                    jcumagg(
                        jmin,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_MIN,
                        data,
                        bin_res_idx,
                        bin_res,
                        buffer_MIN,
                        False,
                    )
                if assess_MAX:
                    jcumagg(
                        jmax,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_MAX,
                        data,
                        bin_res_idx,
                        bin_res,
                        buffer_MAX,
                        False,
                    )
                if assess_SUM:
                    jcumagg(
                        jsum,
                        pinnu,
                        chunk_start,
                        next_chunk_start,
                        cols_SUM,
                        data,
                        bin_res_idx,
                        bin_res,
                        buffer_SUM,
                        False,
                    )
                bin_res_idx += 1
                bin_start = next_chunk_start
                pinnu = False
        chunk_start = next_chunk_start


jit_module(nopython=True, error_model="numpy")
