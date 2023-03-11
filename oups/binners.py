#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from math import ceil
from typing import Tuple

from numba import boolean
from numba import float64
from numba import guvectorize
from numba import int64
from numpy import NaN as nNaN
from numpy import arange
from numpy import dtype
from numpy import ndarray
from numpy import ndenumerate
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Int64Dtype
from pandas import IntervalIndex
from pandas import NaT as pNaT
from pandas import Series
from pandas import date_range
from pandas.core.resample import _get_timestamp_range_edges as gtre


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_FLOAT64 = dtype("float64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
DTYPE_NULLABLE_INT64 = Int64Dtype()
NULL_INT64_1D_ARRAY = zeros(0, DTYPE_INT64)
NULL_INT64_2D_ARRAY = NULL_INT64_1D_ARRAY.reshape(0, 0)
# Null values.
NULL_DICT = {DTYPE_INT64: pNA, DTYPE_FLOAT64: nNaN, DTYPE_DATETIME64: pNaT}
# Keys for 'bin_by_x_rows'.
KEY_ROWS_IN_LAST_BIN = "rows_in_last_bin"
KEY_LAST_KEY = "last_key"


@guvectorize(
    [
        (int64[:], int64[:], boolean, int64[:], int64[:]),
        (float64[:], float64[:], boolean, int64[:], int64[:]),
    ],
    "(l),(m),(),(n),(o)",
    nopython=True,
)
def _next_chunk_starts(
    data: ndarray,
    right_edges: ndarray,
    right: bool,
    next_chunk_starts: ndarray,
    n_null_chunks: ndarray,
):
    """Return row indices for starts of next chunks.

    Parameters
    ----------
    data: ndarray
        One-dimensional array from which deriving next chunk starts, assuming
        data is sorted (monotonic increasing data).
    right_edges: ndarray
        One-dimensional array of chunk right edges, sorted.
    right : bool
        If `True`, histogram is built considering right-closed bins.
        If `False`, histogram is built considering left-closed bins.

    Returns
    -------
    next_chunk_starts: ndarray
        One-dimensional array, containing row indices for start of next chunk,
        to bin 'data' as per 'right_edges'.
        Size of 'next_chunk_starts' is ``len(right_edges)``.
    n_null_chunks: ndarray
        One-dimensional array of size 1, which single value is the number of
        null chunks identified.
    """
    # Flag for counting null chunks.
    prev_d_idx = 0
    _d_idx = prev_d_idx = 0
    data_max_idx = len(data) - 1
    for (b_idx_loc,), bin_ in ndenumerate(right_edges):
        prev_bin = True
        if right:
            # Right-closed bins.
            for (_d_idx_loc,), val in ndenumerate(data[_d_idx:]):
                if val > bin_:
                    prev_bin = False
                    break
        else:
            # Left-closed bins.
            for (_d_idx_loc,), val in ndenumerate(data[_d_idx:]):
                if val >= bin_:
                    prev_bin = False
                    break
        _d_idx += _d_idx_loc
        if _d_idx == data_max_idx and prev_bin:
            # Array 'data' terminated and loop stayed in previous chunk.
            # Then, last loop has not been accounted for.
            # Hence a '+1' to account for it.
            next_chunk_starts[b_idx_loc:] = _d_idx + 1
            n_null_chunks[0] += len(next_chunk_starts[b_idx_loc:]) - 1
            return
        else:
            next_chunk_starts[b_idx_loc] = _d_idx
            if prev_d_idx == _d_idx:
                n_null_chunks[0] += 1
            else:
                prev_d_idx = _d_idx


def bin_by_time(on: Series, by: Grouper) -> Tuple[IntervalIndex, ndarray, int]:
    """Bin as per pandas Grouper of an ordered date time index.

    Parameters
    ----------
    on : Series
        Ordered date time index over which performing the binning as defined
        per 'by'.
    by : Grouper
        Setup to define binning as a pandas Grouper

    Returns
    -------
    next_chunk_starts : ndarray
        One-dimensional array of `int` specifying the row indices of the
        next-bin starts, for each bin. Successive identical indices implies
        empty bins, except the first bin in the series.
    bins : IntervalIndex
        IntervalIndex defining each bin by its left and right edges, and how
        it is closed, right or left.
    n_null_chunks: int
        Number of null chunks identified in 'on'.
    """
    start, end = gtre(
        first=on.iloc[0],
        last=on.iloc[-1],
        freq=by.freq,
        closed=by.closed,
        origin=by.origin,
        offset=by.offset,
    )
    edges = date_range(start, end, freq=by.freq)
    next_chunk_starts = zeros(len(edges) - 1, dtype=DTYPE_INT64)
    n_null_chunks = zeros(1, dtype=DTYPE_INT64)
    _next_chunk_starts(
        on.to_numpy(copy=False).view(DTYPE_INT64),
        edges[1:].to_numpy(copy=False).view(DTYPE_INT64),
        by.closed == "right",
        next_chunk_starts,
        n_null_chunks,
    )
    return next_chunk_starts, IntervalIndex.from_breaks(edges, closed=by.closed), n_null_chunks[0]


def bin_by_x_rows(data: pDataFrame, buffer: dict = None, x_rows: int = 4):
    """Bin by group of X rows.

    Dummy binning function for testing 'cumsegagg' with 'bin_by' set as a
    Callable.

    Parameters
    ----------
    data : pandas DataFrame
        A pandas DataFrame made either of one or 2 columns, from which deriving
          - the number of rows in 'data',
          - bin labels for each bin,
          - if 'no_snapshot' is False, then 'bin_ends' values are derived from
            the last column of 'data'.
    buffer : dict
        2 parameters are in kept to allow new faultless calls to 'by_x_rows':
          - 'rows_in_last_bin', an int specifying the number of rows in last
            (and possibly incomplete) bin from the previous call to
            'bin_x_rows'.
          - 'last_key', last key of last (possibly incomplete) bin.
    x_rows : int, default 4
        Number of rows in a bin.

    Returns
    -------
    tuple of 5 items
        The first 3 items are used in 'cumsegagg' in all situations.,
          - 'next_chunk_starts', a 1d numpy array of int, specifying for each
             bin the row indice at which starts the next bin.
          - 'bin_labels', a pandas Series specifying for each bin its label.
          - 'n_null_bins', an int indicating the number of null bins, if any.

        The 2 last items are used only if both bins and snapshots are generated
        in 'cumsegagg'.
          - 'bin_closed', a str, ``"right"`` or ``"left"``,  indicating if the
            bin is either right-closed, resp. left-closed.
          - 'bin_ends', a str, made of values from the last columns of 'data'
            (which is either single-column or two-column) and indicating the
            "position" of the bin end (either included or excluded, as per
            'bin_closed')

    """
    len_data = len(data)
    # Derive number of rows in first bins (cannot be 0) and number of bins.
    rows_in_last_bin = (
        buffer[KEY_ROWS_IN_LAST_BIN]
        if (buffer is not None and KEY_ROWS_IN_LAST_BIN in buffer)
        else 0
    )
    rows_in_first_bin = min(
        x_rows - rows_in_last_bin if rows_in_last_bin != x_rows else x_rows, len_data
    )
    n_rows_for_new_bins = len_data - rows_in_first_bin
    n_bins = ceil(n_rows_for_new_bins / x_rows) + 1
    # Define 'next_chunk_starts'.
    next_chunk_starts = arange(
        start=rows_in_first_bin, stop=n_bins * x_rows + rows_in_first_bin, step=x_rows
    )
    # Make a copy and arrange for deriving 'chunk_starts', required for
    # defining bin labels. 'bin_labels' are derived from last column (is then
    # 'ordered_on' and if not, is 'bin_on'). Bin labels are 1st value in bin.
    chunk_starts = next_chunk_starts.copy() - x_rows
    # Correct start of 1st chunk.
    chunk_starts[0] = 0
    bin_labels = data.iloc[chunk_starts, -1].reset_index(drop=True)
    # Adjust 'next_chunk_start' of last bin.
    next_chunk_starts[-1] = len_data
    if buffer is not None:
        # Correct 1st label if not a new bin.
        if KEY_LAST_KEY in buffer and rows_in_first_bin != x_rows:
            bin_labels.iloc[0] = buffer[KEY_LAST_KEY]
        # Update 'buffer[rows_in_last_bin]' with number of rows in last bin for
        # next run.
        buffer[KEY_ROWS_IN_LAST_BIN] = n_rows_for_new_bins % x_rows or x_rows
        # Update'buffer[last_key]' with last bin label.
        buffer[KEY_LAST_KEY] = bin_labels.iloc[-1]
    # 'bin_closed' is "left".
    # 'bin_ends' is same as 'bin_labels' (is excluded)
    return next_chunk_starts, bin_labels, 0, "left", bin_labels
