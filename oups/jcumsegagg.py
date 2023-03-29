#!/usr/bin/env python3
"""
Created on Fri Dec 22 19:00:00 2022.

@author: yoh
"""
from typing import Callable, Tuple

from numba import boolean
from numba import float64
from numba import int64
from numba import literal_unroll
from numba import njit
from numpy import ndarray
from numpy import ndenumerate
from numpy import zeros


@njit(
    [int64[:](int64[:, :], int64[:], boolean), float64[:](float64[:, :], float64[:], boolean)],
    cache=True,
)
def jfirst(ar: ndarray, initial: ndarray, use_init: bool):
    """Jitted first.

    Parameters
    ----------
    ar : np.ndarray
        2d numpy array from which taking first value.
    initial : np.ndarray
        1d numpy array containing 'initial' values to be considered as previous
        first.
    use_init : bool
        If 'initial' should be used.

    Returns
    -------
    np.ndarray
        First in 'ar' if 'use_init' is false, else 'initial'.
    """
    if use_init > 0:
        return initial
    elif len(ar) > 0:
        return ar[0]
    else:
        return zeros(ar.shape[1], dtype=ar.dtype)


@njit(
    [int64[:](int64[:, :], int64[:], boolean), float64[:](float64[:, :], float64[:], boolean)],
    cache=True,
)
def jlast(ar: ndarray, initial: ndarray, use_init: bool):
    """Jitted last.

    Parameters
    ----------
    ar : np.ndarray
        2d numpy array from which taking last value.
    initial : np.ndarray
        1d numpy array containing 'initial' values to be considered as previous
        last.
        These values are used if 'ar' is a null array.
    use_init : bool
        If 'initial' should be used.

    Returns
    -------
    np.ndarray
        Last in 'ar' if 'ar' is not a null array, else 'initial'.
    """
    if len(ar) > 0:
        return ar[-1]
    elif use_init:
        return initial
    else:
        return zeros(ar.shape[1], dtype=ar.dtype)


@njit(
    [int64[:](int64[:, :], int64[:], boolean), float64[:](float64[:, :], float64[:], boolean)],
    cache=True,
    parallel=True,
)
def jmax(ar: ndarray, initial: ndarray, use_init: bool):
    """Jitted max.

    Parameters
    ----------
    ar : np.ndarray
        2d numpy array over which retrieving max values for each column.
    initial : np.ndarray
        1d numpy array containing 'initial' values to be considered in max
        evaluation, for each column.
    use_init : bool
        If 'initial' should be used.

    Returns
    -------
    np.ndarray
        Max values per column in 'ar', including 'initial' if 'use_init' is
        true.
    """
    len_ar = len(ar)
    if len_ar > 0:
        if use_init:
            k = 0
            res = initial
        else:
            k = 1
            res = ar[0]
        if len_ar > 1 or k == 0:
            for row in ar[k:]:
                for i, val in ndenumerate(row):
                    if val > res[i]:
                        res[i] = val
        return res
    elif use_init:
        return initial
    else:
        return zeros(ar.shape[1], dtype=ar.dtype)


@njit(
    [int64[:](int64[:, :], int64[:], boolean), float64[:](float64[:, :], float64[:], boolean)],
    cache=True,
    parallel=True,
)
def jmin(ar: ndarray, initial: ndarray, use_init: bool):
    """Jitted min.

    Parameters
    ----------
    ar : np.ndarray
        2d numpy array over which retrieving min values for each column.
    initial : np.ndarray
        1d numpy array containing 'initial' values to be considered in min
        evaluation, for each column.
    use_init : bool
        If 'initial' should be used.

    Returns
    -------
    np.ndarray
        Min values per column in 'ar', including 'initial' if 'use_init' is
        true.
    """
    len_ar = len(ar)
    if len_ar > 0:
        if use_init:
            k = 0
            res = initial
        else:
            k = 1
            res = ar[0]
        if len_ar > 1 or k == 0:
            for row in ar[k:]:
                for i, val in ndenumerate(row):
                    if val < res[i]:
                        res[i] = val
        return res
    elif use_init:
        return initial
    else:
        return zeros(ar.shape[1], dtype=ar.dtype)


@njit(
    [int64[:](int64[:, :], int64[:], boolean), float64[:](float64[:, :], float64[:], boolean)],
    cache=True,
    parallel=True,
)
def jsum(ar: ndarray, initial: ndarray, use_init: bool):
    """Jitted sum.

    Parameters
    ----------
    ar : np.ndarray
        2d numpy array over which assessing sum of values for each column.
    initial : np.ndarray
        1d numpy array containing 'initial' values to be considered in sum
        evaluation, for each column.
    use_init : bool
        If 'initial' should be used.

    Returns
    -------
    np.ndarray
        Sum of values per column in 'ar', including 'initial' if 'use_init' is
        true.
    """
    len_ar = len(ar)
    if len_ar > 0:
        if use_init:
            k = 0
            res = initial
        else:
            k = 1
            res = ar[0]
        if len_ar > 1 or k == 0:
            for row in ar[k:]:
                res += row
        return res
    elif use_init:
        return initial
    else:
        return zeros(ar.shape[1], dtype=ar.dtype)


# Aggregation function ids.
FIRST = "first"
LAST = "last"
MIN = "min"
MAX = "max"
SUM = "sum"
AGG_FUNCS = {FIRST: jfirst, LAST: jlast, MIN: jmin, MAX: jmax, SUM: jsum}


@njit
def jcsagg(
    data: ndarray,  # 2d
    n_cols: int,
    aggs: Tuple[Tuple[Callable, ndarray, ndarray]],
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
    n_cols : int
        Number of columns expected in 'bin_res' and/or 'snap_res'.
    aggs : Tuple[Tuple[Callable, np.ndarray, np.ndarray]
        Tuple of Tuple with 3 items:
          - aggregation function,
          - for related aggregation function, a 1d numpy array listing the
            indices of columns in 'data' to which apply the aggregation
            function.
          - for related aggregration function, and corresponding column in
            'data', the index of the column in 'bin_res' and/or 'snap_res' to
            which recording the result. These indices are listed in a 1d numpy
            array, sorted in the same order than indices of columns in 'data'.
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
    # /!\ WiP for restart
    # check len(bin_indices) == len(bin_res) or if bin_indices[-1] == len(bin_res) -1
    # i.e. check if last bin ends sharp on data, or if last bin is un-finished
    # If last bin is unfinished, and bin_res > 0, then record last state in last row of bin_res.
    # /!\ n_null_bins becomes a n_max_null_bins with restart:
    # the first bin indices can have next_chun_start at 0 (meaning emptybin)
    # but with chunk_res (from previous iter) != 0
    # correct n_null_bin in 'cumsegagg' depending reset state / reset_state should be output
    # is given by pinnu, make a 'return pinnu'
    # pinnu should be renamed 'reset' ?
    data_dtype = data.dtype
    chunk_res = zeros(n_cols, data_dtype)
    bin_start = chunk_start = 0
    bin_res_idx = snap_res_idx = 0
    null_bin_idx = null_snap_idx = 0
    # A 'snapshot' is an 'update' or 'pass-through'. A 'bin' induces a reset.
    if len(snap_res) != 0:
        # Case 'snapshots expected'.
        # Setup identification of bins (vs snapshots).
        some_snaps = True
        # /!\ WiP for restart
        # do not check bin_indices, but len(bin_res)
        # there could be no bin_indices (meaning there is a single unfinished bin in the data)
        # but state at end of data for this single unfinished bin is still to be recorded
        # in bin_res
        # do not record / manage bin only if len(bin_res) is 0.
        # create a corresponding test case with len(bin_indices) = 0 and len(bin_res) = 1
        if len(bin_indices) > 0:
            # Case 'there are bins'.
            # If a 'snapshot' chunk shares same last row than a 'bin'
            # chunk, the 'snapshot' is expected to be listed prior to the
            # 'bin' chunk. This is ensures in the way bin indices are sorted
            # vs snapshot indices. Bin indices are identified thanks to
            # 'bin_indices'.
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
    # 'pinnu' is 'prev_is_non_null_update'.
    # With 'pinnu' True, then cumulate (pass-through) previous results.
    pinnu = False
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
        # Null chunk is identified if no new data since start of 'bin',
        # whatever the chunk is, a 'bin' or a 'snapshot' (update).
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
            chunk = data[chunk_start:next_chunk_start]
            # Step 1: compute results for current chunk.
            # If no data in current chunk, 'chunk_res' is naturally forwarded
            # to next iteration, no need to update it.
            if len(chunk) != 0:
                # for agg in aggs:
                for agg in literal_unroll(aggs):
                    agg_func, cols_data, cols_res = agg
                    chunk_res[cols_res] = agg_func(chunk[:, cols_data], chunk_res[cols_res], pinnu)
            # Step 2: record results.
            if is_update:
                # Case of 'snapshot', record result in 'snap_res'.
                snap_res[snap_res_idx, :] = chunk_res
                # Update local variables and counters.
                snap_res_idx += 1
                pinnu = True
            else:
                # Case of 'bin', record results in 'bin_res'.
                bin_res[bin_res_idx, :] = chunk_res
                # Update local variables and counters to reflect end of bin.
                bin_res_idx += 1
                bin_start = next_chunk_start
                pinnu = False
        chunk_start = next_chunk_start
