#!/usr/bin/env python3
"""
Created on Fri Dec 22 19:00:00 2022.

@author: yoh
"""
from typing import Callable

from numba import njit
from numpy import array
from numpy import dtype
from numpy import zeros


@njit
def jmax(ar: array, initial=None):
    """
    Jitted max.

    Parameters
    ----------
    ar : np.array
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


@njit
def jmin(ar: array, initial=None):
    """
    Jitted min.

    Parameters
    ----------
    ar : np.array
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


@njit
def jsum(ar: array, initial=None):
    """
    Jitted sum.

    Parameters
    ----------
    ar : np.array
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


@njit
def jcumagg(
    agg: Callable,
    cumulate: bool,
    chunk_start: int,
    next_chunk_start: int,
    cols: array,
    data: array,
    res_idx: int,
    res: array,
    buffer: array,
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
    cols : array
        2d array of ``int``, with as many rows as columns onto which
        aggregating. Each row contains 2 values, the index of the column in
        'data', and the index of the column in 'res'.
    data : array
        2d array. Contiguous segments of rows are 'chunks'.
    res_idx : int
        Row index in 'res' to use for storing aggregation results.
    res : array (out)
        2d array of results from the cumulative aggregation.
    buffer : array (in/out)
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


@njit
def jrowat(
    from_buffer: bool,
    data_idx: int,
    cols: array,
    data: array,
    res_idx: int,
    res: array,
    buffer: array,
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
    cols : array
        2d array of ``int``, with as many rows as columns onto which
        aggregating. Each row contains 2 values, the index of the column in
        'data', and the index of the column in 'res'.
    data : array
        2d array. Contiguous segments of rows are 'chunks'.
    res_idx : int
        Row index in 'res' to use for storing aggregation results.
    res : array (out)
        2d array of results.
    buffer : array (in/out)
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


@njit
def jcsa_setup(agg_id: int, n_cols: array, cols: array, dtype_: dtype):
    """
    Return setup for one cumulative segmented aggregation function.

    Parameters
    ----------
    agg_id : int
        Id of aggregation function.
    n_cols : array
        1d array, specifying per aggrgation function how many columns in 'data'
        have to be aggregated with this function.
    cols: array
        3d array, is one row per aggregation function.
        Per aggregation function, a 2d array containing the list of twin column
        indices onto which applying the aggregation function. The 1st index
        is the index of column in 'data', the 2nd index is the index of column
        in 'res'.
    dtype_ : dtype
        data array ``dtype``.

    Returns
    -------
    bool, Union[array, None], Union[array, None]

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
