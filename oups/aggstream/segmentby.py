#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh

"""
from functools import partial
from math import ceil
from math import fmod
from typing import Callable, Dict, Optional, Tuple, Union

from numba import njit
from numpy import arange
from numpy import argsort
from numpy import concatenate
from numpy import diff as ndiff
from numpy import dtype
from numpy import full
from numpy import insert as ninsert
from numpy import ndarray
from numpy import ndenumerate
from numpy import nonzero
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import IntervalIndex
from pandas import Series as pSeries
from pandas import Timedelta
from pandas import concat as pconcat
from pandas import date_range
from pandas.core.resample import TimeGrouper
from pandas.core.resample import _get_timestamp_range_edges as gtre


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
NULL_INT64_1D_ARRAY = zeros(0, DTYPE_INT64)
LEFT = "left"
RIGHT = "right"
# Keys for main buffer.
KEY_BIN = "bin"
KEY_SNAP = "snap"
# Keys for 'by_...' when a Callable.
KEY_LAST_BIN_LABEL = "last_bin_label"
KEY_LAST_BIN_END = "last_bin_end"
KEY_RESTART_KEY = "restart_key"
KEY_LAST_ON_VALUE = "last_on_value"
# Keys for 'bin_by' when a dict
KEY_ON_COLS = "on_cols"
KEY_BIN_BY = "bin_by"
KEY_ORDERED_ON = "ordered_on"
KEY_SNAP_BY = "snap_by"
KEY_BIN_ON = "bin_on"


@njit(
    [
        "Tuple((int64[:], int64, boolean))(int64[:], int64[:], boolean)",
        "Tuple((int64[:], int64, boolean))(float64[:], float64[:], boolean)",
    ],
)
def _next_chunk_starts(
    data: ndarray,
    right_edges: ndarray,
    right: bool,
):
    """
    Return row indices for starts of next chunks.

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
    next_chunk_starts : ndarray
        One-dimensional array, containing row indices for start of next chunk,
        to bin 'data' as per 'right_edges'.
        If last right edges are out of 'data', the 'next chunk starts' for the
        resulting empty bins are not returned.
        Size of 'next_chunk_starts' is smaller than or equal to
        ``len(right_edges)``.
    n_null_chunks : ndarray
        One-dimensional array of size 1, which single value is the number of
        null chunks identified.
    data_traversed : boolean
        Specifies if 'data' has been completely traversed or not.

    """
    # Output variables
    next_chunk_starts = zeros(len(right_edges), dtype=DTYPE_INT64)
    n_null_chunks = 0
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
            next_chunk_starts[b_idx_loc] = _d_idx + 1
            # Previous code to return all bins, including the empty ones
            # defined by the last values in 'right_edges'.
            # next_chunk_starts[b_idx_loc:] = _d_idx + 1
            # n_null_chunks += len(next_chunk_starts[b_idx_loc:]) - 1
            # Do not return empty bins at end of data.
            return next_chunk_starts[: b_idx_loc + 1], n_null_chunks, True
        else:
            next_chunk_starts[b_idx_loc] = _d_idx
            if prev_d_idx == _d_idx:
                n_null_chunks += 1
            else:
                prev_d_idx = _d_idx
    # Array 'right_edges' is terminated, before 'data' is ended.
    return next_chunk_starts, n_null_chunks, False


def by_scale(
    on: pSeries,
    by: Union[TimeGrouper, pSeries, Tuple[pSeries]],
    closed: Optional[str] = None,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, pSeries, int, str, pSeries, bool]:
    """
    Segment an ordered DatetimeIndex or Series.

    Parameters
    ----------
    on : Series
        Ordered date time index over which performing the binning as defined
        per 'by'.
    by : Grouper or Series or tuple of 2 Series
        Setup to define binning as a pandas TimeGrouper, or values contained in
        a Series.
        If a Series, values are used both as ends and labels of chunks.
        If a tuple of 2 Series, values in first Series are labels of chunks,
        and second Series are ends of chunks.
    closed : str, default None
        Optional string, specifying if intervals defined by 'by' are left or
        right closed. This parameter overrides 'by.closed' if 'by' is a pandas
        TimeGrouper.
    buffer : dict
        Dict to keep parameters allowing chaining calls to 'by_scale', with
        ``restart_key``, keeping track of the end of the one-but-last chunk
        from previous iteration, derived from 'by'.

    Returns
    -------
    Tuple[ndarray, Series, int, str, Series, bool]
        The first 3 items are used in 'cumsegagg' in all situations.
          - ``next_chunk_starts``, a one-dimensional array of `int` specifying
            the row indices of the next-bin starts, for each bin. Successive
            identical indices imply empty bins, except the first.
          - ``chunk_labels``, a pandas Series specifying for each bin its
            label. Labels are defined as per 'on' pandas TimeGrouper.
          - ``n_null_chunks``, an int, the number of null chunks identified in
            'on'.

        The 3 following items are used only if both bins and snapshots are
        generated in 'cumsegagg'.
          - ``chunk_closed``, a str, indicating if bins are left or right
            closed, as per 'by' pandas TimeGrouper or 'closed' parameter.
          - ``chunk_ends``, a pandas Series containing bin ends, as per 'by'
            pandas TimeGrouper.
          - ``unknown_last_chunk_end``, a boolean, always `False`, specifying
            that the last chunk end is known. This is because chunk ends are
            always fully specified as per 'by' pandas TimeGrouper or Series.

    Notes
    -----
    If running ``by_scale()`` with a buffer, setting of value for key
    `"restart_key`" depends if last value derived from 'by' (either a
    TimeGrouper or a Series) lies before the last value in 'on'.
      - If it lies before, then this last value derived from 'by' is the
        restart key.
      - If it lies after, then the one-but-last value derived from 'by' is the
        restart key.

    """
    if isinstance(by, TimeGrouper):
        # If 'buffer' is not empty, it necessarily contains 'KEY_RESTART_KEY'.
        first = buffer[KEY_RESTART_KEY] if buffer else on.iloc[0]
        # In case 'by' is for snapshotting, and 'closed' is not set, take care
        # to use 'closed' provided.
        if closed is None:
            closed = by.closed
        # TODO: replace with date_utils.floor_ts() and date_utils.ceil_ts()?
        start, end = gtre(
            first=first,
            last=on.iloc[-1],
            freq=by.freq,
            closed=closed,
            unit=first.unit,
            origin=by.origin,
            offset=by.offset,
        )
        edges = date_range(start, end, freq=by.freq)
        chunk_ends = edges[1:]
        chunk_labels = chunk_ends if by.label == RIGHT else edges[:-1]
    else:
        # Case 'by' is a Series.
        if closed is None:
            raise ValueError(f"'closed' has to be set to {LEFT} or {RIGHT}.")
        if isinstance(by, tuple):
            chunk_labels, chunk_ends = by
            if len(chunk_labels) != len(chunk_ends):
                raise ValueError(
                    "number of chunk labels has to be " "equal to number of chunk ends.",
                )
        else:
            chunk_labels = chunk_ends = by
        if buffer:
            # In case at previous iteration, there has been no snapshot,
            # 'buffer' will not contain 'KEY_RESTART_KEY', but will contain
            # 'KEY_LAST_ON_VALUE'.
            if KEY_RESTART_KEY in buffer and buffer[KEY_RESTART_KEY] != chunk_ends[0]:
                # In case of restart, if first value in 'chunk_ends' is not the
                # the one that was used in last at last iteration, try first to
                # trim values in 'by' that are earlier than 'restart_key'.
                n_chunk_ends_init = len(chunk_ends)
                chunk_ends = chunk_ends[chunk_ends >= buffer[KEY_RESTART_KEY]]
                if buffer[KEY_RESTART_KEY] != chunk_ends[0]:
                    raise ValueError(
                        f"'by' needs to contain value {buffer[KEY_RESTART_KEY]} "
                        "to restart correctly.",
                    )
                n_first_chunks_to_remove = n_chunk_ends_init - len(chunk_ends)
                chunk_labels = chunk_labels[n_first_chunks_to_remove:]
            if KEY_LAST_ON_VALUE in buffer:
                # In the specific case 'on' has not been traversed completely
                # at previous iteration, the chunk for the remaining of the
                # data has no label, and will not appear in the snapshot
                # results. But it will be calculated during the aggregation
                # phase ('cumsegagg()'), and kept in a temporary variable
                # ('chunk_res').
                # In this case, at next iteration, with new chunk ends, a
                # specific check is managed here to ensure correctness of the
                # restart.
                # For this new iteration,
                #  - a new bin has necessarily to be started. Otherwise,
                #    aggregation results for last chunk at previous iteration
                #    will overwrite those of elapsed last bin. This last bin
                #    has been completed at previous iteration. Its results
                #    do not have to be modified.
                #  - this new first bin has to end after the last value in 'on'
                #    from previous iteration. If it is not, then the remaining
                #    aggregated data from previous iteration is not usable, as
                #    it aggregates over several chunks.
                # If there is a single chunk end, then it is that of previous
                # iteration, nothing to check.
                last_on_value = buffer[KEY_LAST_ON_VALUE]
                if len(chunk_ends) > 1 and (
                    (closed == RIGHT and chunk_ends[1] < last_on_value)
                    or (closed == LEFT and chunk_ends[1] <= last_on_value)
                ):
                    raise ValueError(
                        "2nd chunk end in 'by' has to be larger than value "
                        f"{buffer[KEY_LAST_ON_VALUE]} to restart correctly.",
                    )
                if (closed == RIGHT and chunk_ends[0] < last_on_value) or (
                    closed == LEFT and chunk_ends[0] <= last_on_value
                ):
                    # At previous iteration, if last value in 'on' is later
                    # than first chunk end, then this chunk should not be
                    # updated. It is 'done'.
                    # To prevent updating it, this chunk should be removed.
                    # Only the 1st chunk is removed, because it was just
                    # checked that 2nd chunk complies correctly with this
                    # condition.
                    chunk_ends = chunk_ends[1:]
                    chunk_labels = chunk_labels[1:]
                del buffer[KEY_LAST_ON_VALUE]
        if chunk_ends.empty:
            if isinstance(buffer, dict):
                buffer[KEY_LAST_ON_VALUE] = on.iloc[-1]
            return (NULL_INT64_1D_ARRAY, chunk_labels, 0, closed, chunk_ends, False)
    if chunk_ends.dtype == DTYPE_DATETIME64:
        next_chunk_starts, n_null_chunks, data_traversed = _next_chunk_starts(
            on.to_numpy(copy=False).view(DTYPE_INT64),
            chunk_ends.to_numpy(copy=False).view(DTYPE_INT64),
            closed == RIGHT,
        )
    else:
        next_chunk_starts, n_null_chunks, data_traversed = _next_chunk_starts(
            on.to_numpy(copy=False),
            chunk_ends.to_numpy(copy=False),
            closed == RIGHT,
        )
    n_chunks = len(next_chunk_starts)
    # Rationale for selecting the "restart key".
    # - For a correct restart at iteration N+1, the restart point needs to be
    #   that of the last bin at iteration N that has been "in-progress". The
    #   restart is said correct because it restarts on new data, where
    #   aggregation at iteration N stopped. There is no omission of new data,
    #   nor omission of possibly empty bins till new data.
    # - At iteration N,
    #     - if last value derived from 'by' is after last value in
    #       "on", then at next iteration, N+1, new data can be used, which
    #       still lies before this last value derived from 'by' at iteration N.
    #       To make sure this new data is correctly managed, we need to restart
    #       from one-but-last value derived from 'by' at iteration N.
    #     - if last value derived from 'by' is before last value in "on", then
    #       at next iteration, N+1, we are sure no new data will appear before
    #       it. This last value can be safely used as restart value.
    # TODO: when splitting 'by_scale()' into 'by_pgrouper()' and 'by_scale()',
    # for 'by_pgrouper()', then using for 'restart_key' the last value in 'on'
    # complies with whatever the 'closed' parameter is (I think). This
    # simplifies below code.
    if data_traversed:
        chunk_labels = chunk_labels[:n_chunks]
        chunk_ends = chunk_ends[:n_chunks]
        if buffer is not None:
            if closed == LEFT and isinstance(by, TimeGrouper):
                # Use of intricate way to get last or last-but-one element in
                # 'chunk_ends', compatible with both Series and DatetimeIndex.
                if n_chunks > 1:
                    # Get one-but-last element.
                    # Initialize this way if there are more than 2 elements at
                    # least.
                    buffer[KEY_RESTART_KEY] = chunk_ends[n_chunks - 2]
                else:
                    # If there is a single incomplete bin, take first element
                    # in 'on'.
                    buffer[KEY_RESTART_KEY] = on.iloc[0]
            else:
                # Take last end
                # - either if 'by' is a TimeGrouper, as it is enough for
                #   generating edges at next iteration.
                # - or if 'by' is a Series, because Series only needs to
                #   restart from this point then.
                buffer[KEY_RESTART_KEY] = chunk_ends[n_chunks - 1]
    elif buffer is not None:
        # Data is not traversed.
        # This can only happen if 'by' is not a TimeGrouper.
        # Keep last chunk end.
        buffer[KEY_RESTART_KEY] = chunk_ends[n_chunks - 1]
        buffer[KEY_LAST_ON_VALUE] = on.iloc[-1]
    return (
        next_chunk_starts,
        pSeries(chunk_labels),
        n_null_chunks,
        closed,
        chunk_ends,
        False,
    )


def by_x_rows(
    on: Union[pDataFrame, pSeries],
    by: Optional[int] = 4,
    closed: Optional[str] = LEFT,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, pSeries, int, str, pSeries, bool]:
    """
    Segment by group of x rows.

    Dummy binning function for testing 'cumsegagg' with 'bin_by' set as a
    Callable.

    Parameters
    ----------
    on : Union[pDataFrame, pSeries]
        Either a pandas Series or a DataFrame made of two columns, from which
        deriving
          - the number of rows in 'on',
          - bin labels for each bin (from the last column of 'on'),
          - bin ends for each bin (from the last column of 'on').
    by : int, default 4
        Number of rows in a bin.
    closed : str, default "left"
        How is closed the segments, either "left" or "right".
    buffer : dict, default None
        Dict to keep 2 parameters allowing chaining calls to 'by_x_rows':
          - 'restart_key', an int specifying the number of rows in last
            (and possibly incomplete) bin from the previous call to
            'bin_x_rows'.
          - 'last_bin_label', label of the last bin, that will be reused in
            next iteration.

    Returns
    -------
    Tuple[ndarray, Series, int, str, Series, bool]
        The first 3 items are used in 'cumsegagg' in all situations.
          - ``next_chunk_starts``, a one-dimensional numpy array of int,
            specifying for each bin the row indice at which starts the next
            bin.
          - ``bin_labels``, a pandas Series specifying for each bin its label.
            Labels are first value in bin taken in last column of 'on' (which
            is supposed to be an ordered column).
          - ``n_null_bins``, an int, always ``0``.

        The 3 next items are used only if both bins and snapshots are generated
        in 'cumsegagg'.
          - ``bin_closed``, a str, ``"left"`` or ``"right"``, indicating that
            the bins are left or right closed.
          - ``bin_ends``, a pandas Series made of values from the last columns
            of 'on' (which is either single-column or two-column) and
            indicating the "position" of the bin end, which is marked by the
            start of the next bin, excluded. The end of the last bin being
            unknown by definition (because is excluded), the last value is not
            relevant. It is forced anyhow in 'segmentby()' to be last.
          - ``unknown_last_bin_end``, a boolean specifying if the last bin end
            is unknown. It is ``True`` if bins are lef-closed, meaning that
            their end is excluded. Hence, the last bin is always "in-progress".
            It is ``False`` if they are right-closed.

    """
    len_on = len(on)
    if isinstance(on, pDataFrame):
        # Keep only last column, supposed to be `ordered_on` column.
        on = on.iloc[:, -1]
    # Derive number of rows in first bins (cannot be 0) and number of bins.
    if buffer is not None and KEY_RESTART_KEY in buffer:
        # Case 'restart'.
        rows_in_prev_last_bin = buffer[KEY_RESTART_KEY]
        rows_in_continued_bin = (
            min(len_on, by - rows_in_prev_last_bin) if rows_in_prev_last_bin != by else 0
        )
    else:
        # Case 'start from scratch'.
        rows_in_prev_last_bin = 0
        rows_in_continued_bin = 0
    n_rows_for_new_bins = len_on - rows_in_continued_bin
    n_bins = (
        ceil(n_rows_for_new_bins / by) + 1
        if rows_in_continued_bin
        else ceil(n_rows_for_new_bins / by)
    )
    # Define 'next_chunk_starts'.
    first_next_chunk_start = rows_in_continued_bin if rows_in_continued_bin else min(by, len_on)
    next_chunk_starts = arange(
        start=first_next_chunk_start,
        stop=(n_bins - 1) * by + first_next_chunk_start + 1,
        step=by,
    )
    # Make a copy and arrange for deriving 'chunk_starts', required for
    # defining bin labels. 'bin_labels' are derived from last column (is then
    # 'ordered_on' and if not, is 'bin_on'). Bin labels are 1st value in bin.
    chunk_starts = next_chunk_starts.copy() - by
    # Correct start of 1st chunk.
    chunk_starts[0] = 0
    bin_labels = on.iloc[chunk_starts].reset_index(drop=True)
    if n_rows_for_new_bins:
        # Case 'there are new bins'.
        n_rows_in_last_bin = (
            n_rows_for_new_bins
            if n_rows_for_new_bins <= by
            else fmod(n_rows_for_new_bins, by) or by
        )
    else:
        # Case 'there are not'.
        n_rows_in_last_bin = rows_in_continued_bin + rows_in_prev_last_bin
    if closed == LEFT:
        # Case 'left, end is start of next bin, excluded,
        # 'bin_ends' has no end for last bin, because it is unknown.
        # Temporarily adjust 'next_chunk_start' of last bin to last index.
        next_chunk_starts[-1] = len_on - 1
        bin_ends = on.iloc[next_chunk_starts].reset_index(drop=True)
        unknown_last_bin_end = True
    # Reset 'next_chunk_start' of last bin.
    next_chunk_starts[-1] = len_on
    if closed == RIGHT:
        # Case 'right', end is end of current bin, included.
        bin_ends = on.iloc[next_chunk_starts - 1].reset_index(drop=True)
        # Bin end is unknown if last bin does not end exactly.
        unknown_last_bin_end = True if n_rows_in_last_bin != by else False
    # There is likely no empty bin.
    n_null_bins = 0
    if buffer is not None:
        if buffer:
            if rows_in_continued_bin:
                # Correct 1st label if not a new bin.
                bin_labels.iloc[0] = buffer[KEY_LAST_BIN_LABEL]
            else:
                # If a new bin has been created right at start,
                # insert an empty one with label of last bin at prev iteration.
                bin_labels = pconcat(
                    [pSeries([buffer[KEY_LAST_BIN_LABEL]]), bin_labels],
                ).reset_index(drop=True)
                first_bin_end = buffer[KEY_LAST_BIN_END] if closed == RIGHT else on.iloc[0]
                bin_ends = pconcat([pSeries([first_bin_end]), bin_ends]).reset_index(drop=True)
                next_chunk_starts = ninsert(next_chunk_starts, 0, 0)
                # In this case, first bin is empty.
                n_null_bins = 1
        # Update 'buffer[xxx]' parameters for next run.
        buffer[KEY_RESTART_KEY] = n_rows_in_last_bin
        buffer[KEY_LAST_BIN_LABEL] = bin_labels.iloc[-1]
        if closed == RIGHT:
            buffer[KEY_LAST_BIN_END] = bin_ends.iloc[-1]
    return (
        next_chunk_starts,
        bin_labels,
        n_null_bins,
        closed,
        bin_ends,
        unknown_last_bin_end,
    )


def mergesort(
    labels: Tuple[ndarray, ndarray],
    keys: Tuple[ndarray, ndarray],
    force_last_from_second: Optional[bool] = False,
) -> Tuple[ndarray, ndarray]:
    """
    Mergesort labels from keys.

    Parameters
    ----------
    labels : Tuple[ndarray, ndarray]
        2 one-dimensional arrays of labels to be merged together, provided as a
        ``tuple``.
    keys : Tuple[ndarray, ndarray]
        2 one-dimensional arrays of sorted keys according which labels can be
        sorted one with respect to the other.
        ``keys[0]``, resp. ``1``, are keys for ``labels[0]``, resp. ``1``.
    force_last_from_second : bool, default False
        If True, the last label in the resulting sorted array is forced to be
        the last from the second label array.

    Returns
    -------
    Tuple[ndarray, ndarray]
        The first array contains sorted labels from the 2 input arrays.
        The second array contains the insertion indices for labels (i.e. the
        indices in the resulting merged array) from the 2nd input array,

    Notes
    -----
    If a value is found in both input arrays, then value of 2nd input array
    comes after value of 1st input array, as can be checked with insertion
    indices.

    """
    # TODO: transition this to numba.
    labels1, labels2 = labels
    keys1, keys2 = keys
    len_labels1 = len(labels1)
    len_labels2 = len(labels2)
    if len(keys1) != len_labels1:
        raise ValueError(
            "not possible to have arrays of different length for first labels and keys arrays.",
        )
    if len(keys2) != len_labels2:
        raise ValueError(
            "not possible to have arrays of different length for second labels and keys arrays.",
        )
    if force_last_from_second:
        len_tot = len_labels1 + len_labels2
        sort_indices = full(len_tot, len_tot - 1, dtype=DTYPE_INT64)
        sort_indices[:-1] = argsort(concatenate((keys1, keys2[:-1])), kind="mergesort")
    else:
        sort_indices = argsort(concatenate(keys), kind="mergesort")
    return concatenate(labels)[sort_indices], nonzero(len_labels1 <= sort_indices)[0]


def setup_segmentby(
    bin_by: Union[TimeGrouper, Callable],
    bin_on: Optional[str] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[TimeGrouper, pSeries]] = None,
) -> Dict[str, Union[Callable, str]]:
    """
    Check and setup parameters to operate data segmentation.

    Parameters
    ----------
    bin_by : Union[TimeGrouper, Callable]
       A pandas TimeGrouper or a Callable to perform segmentation.
    bin_on : Optional[str]
       Name of the column onto which performing the segmentation.
    ordered_on : Optional[str]
       Name of the column containing ordered data and to use when snapshotting.
       With this column, snapshots (points of observation) can be positioned
       with respect to bin ends.
    snap_by : Optional[Union[TimeGrouper, IntervalIndex]]
       A pandas TimeGrouper or a pandas Series defining the snapshots (points of
       observation).

    Returns
    -------
    Dict[str, Union[Callable, str]]
        A dict with keys
          - ``BIN_BY``, 'bin_by' forced as a Callable,
          - ``ON_COLS``, column name or list of column names to be used for
            segmentation.
          - ``ORDERED_ON``, consolidated value for 'ordered_on' column.

    """
    bin_by_closed = None
    if isinstance(bin_by, TimeGrouper):
        # 'bin_by' is a TimeGrouper.
        bin_by_closed = bin_by.closed
        if bin_by.key:
            if bin_on:
                if bin_by.key != bin_on:
                    raise ValueError(
                        "not possible to set 'bin_by.key' and 'bin_on' to different values.",
                    )
            else:
                bin_on = bin_by.key
        elif not bin_on:
            raise ValueError("not possible to set both 'bin_by.key' and 'bin_on' to `None`.")
        if ordered_on and ordered_on != bin_on:
            raise ValueError(
                "not possible to set 'bin_on' and 'ordered_on' to different values when 'bin_by' is a TimeGrouper.",
            )
        elif not ordered_on:
            # Case 'ordered_on' has not been provided but 'bin_on' has been.
            # Then set 'ordered_on' to 'bin_on'. this is so because 'bin_by' is
            # a TimeGrouper.
            ordered_on = bin_on
        bin_by = partial(by_scale, by=bin_by)
    elif callable(bin_by):
        # 'bin_by' is a Callable.
        if bin_on is None and ordered_on is None:
            raise ValueError("not possible to set both 'bin_on' and 'ordered_on' to `None`.")
    else:
        # 'bin_by' is neither a TimeGrouper, nor a Callable.
        # This is not possible.
        raise ValueError(
            "not possible to have 'bin_by' parameter different "
            "than a pandas TimeGrouper or a Callable.",
        )
    if snap_by is not None:
        if isinstance(snap_by, TimeGrouper):
            if snap_by.key:
                if ordered_on is None:
                    ordered_on = snap_by.key
                elif snap_by.key != ordered_on:
                    raise ValueError(
                        "not possible to set 'ordered_on' and 'snap_by.key' to different values.",
                    )
            if bin_by_closed and snap_by.closed != bin_by_closed:
                raise ValueError(
                    "not possible to set 'bin_by.closed' and 'snap_by.closed' to different values.",
                )
        elif not ordered_on:
            # Case 'snap_by' is not a TimeGrouper.
            raise ValueError(
                "not possible to leave 'ordered_on' to `None` in case of snapshotting.",
            )
    return {
        KEY_BIN_BY: bin_by,
        KEY_ON_COLS: (
            [bin_on, ordered_on]
            if ordered_on and bin_on and ordered_on != bin_on
            else bin_on if bin_on else ordered_on
        ),
        KEY_ORDERED_ON: ordered_on,
        KEY_BIN_ON: bin_on,
        KEY_SNAP_BY: snap_by if isinstance(snap_by, TimeGrouper) else None,
    }


def setup_mainbuffer(buffer: dict, with_snapshot: Optional[bool] = False) -> Tuple[dict, dict]:
    """
    Return 'buffer_bin' and 'buffer_snap' from main buffer.

    Parameters
    ----------
    buffer : dict
        Main buffer, either containing only values for 'buffer_bin', or only
        two keys `"bin"` and `"snap"` providing a separate dict for each of the
        binning and snapshotting processes.
    with_snapshot : bool, default False
        Boolean ``True`` if snapshotting process is requested.

    Returns
    -------
    Tuple[dict, dict]
        The first dict is the binning buffer.
        The second dict is the snapshotting buffer.

    """
    if buffer is not None:
        if KEY_BIN not in buffer:
            buffer[KEY_BIN] = {}
            if with_snapshot:
                buffer[KEY_SNAP] = {}
        if with_snapshot:
            return buffer[KEY_BIN], buffer[KEY_SNAP]
        else:
            return buffer[KEY_BIN], None
    else:
        return None, None


def segmentby(
    data: pDataFrame,
    bin_by: Union[TimeGrouper, Callable, dict],
    bin_on: Optional[str] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[TimeGrouper, IntervalIndex]] = None,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, ndarray, pSeries, int, pSeries, int]:
    """
    Identify starts of segments in data, either bins or optionally snapshots.

    Parameters
    ----------
    data: pDataFrame
        A pandas DataFrame containing the columns to conduct segmentation of
        data.
          - ``bin_on`` column,
          - optionally ``ordered_on`` column (same as ``snap_by.key`` optional
            column, from 'snap_by' parameter, if using snapshots.

        If any of ``ordered_on`` or ``snap_by.key`` parameters are used, the
        column they point to (the same if both parameters are provided) has to
        be ordered.
    bin_by : Union[TimeGrouper, Callable, dict]
        Callable or pandas TimeGrouper to perform binning.
        If a Callable, it is called with following parameters:
        ``bin_by(on, buffer)``
        where:
          - ``on``,
            - either ``ordered_on`` is ``None``. ``on`` is then a pandas Series
              made from ``data[bin_on]`` column.
            - or ``ordered_on`` is provided and is different from ``bin_on``.
              Then ``on`` is a two-column pandas DataFrame made of
              ``data[[bin_on, ordered_on]]``.
              Values from ``data[ordered_on]`` have to be used to define bin
              ends when 'snap_by' is set.
              Also, values from ``data[ordered_on]`` can be used advantageously
              as bin labels.

          - ``buffer``, a dict that has to be modified in-place by 'bin_by' to
            keep internal parameters which allow restart calls to 'bin_by'.

        If a dict, it contains the full setup for conducting the segmentation
        of 'data', as generated by 'setup_segmentby()'.
          - 'on_cols', a str or list of str, to be forwarded to 'bin_by'
            Callable.
          - 'bin_by', a Callable, either the one initially provided, or one
            derived from a pandas TimeGrouper.
          - 'ordered_on', a str, its definitive value.
          - 'snap_by', if a TimeGrouper.

        It has then to return a tuple made of 6 items. There are 3 items used
        whatever if snapshotting is used or not.
          - ``next_chunk_starts``, a one-dimensional array of `int`, specifying
            the row index at which the next bin starts (included) as found in
            ``bin_on``.
            If the same indices appear several times, it means that
            corresponding bins are empty, except the first one. In this case,
            corresponding rows in aggregation result will be filled with null
            values.
            Last value of this array always equals to ``len(on)``.
          - ``bin_labels``, a pandas Series which values are expected to be
            all bin labels, incl. those of empty bins, as they will appear in
            aggregation results. Labels can be of any type.
            In case of restarting the aggregation with new seed data, care
            should be taken so that the label of the first bin is the same as
            that of the last bin from previous iteration if it has been the
            same bin. An exception is raised if not.
          - ``n_null_bins``, an `int` indicating the number of empty bins.

        The 3 next items are used only in case of snapshotting (``snap_by`` is
        different than ``None``).
          - ``bin_closed``, a str, either `'right'` or `'left'`, indicating
            if bins are left or right-closed (i.e. if ``chunk_ends`` is
            included or excluded in the bin).
          - ``bin_ends``, an optional pandas Series, specifying the ends of
            bins with values derived from ``data[ordered_on]`` column. If
            snapshotting, then points of observation (defined by ``snap_by``)
            are positioned with respect to the bin ends. This data allows
            sorting snapshots with respect to bins in case they start/end at
            the same row index in data.
            ``bin_ends`` is not required if no snapshotting. If not used, set
            to None.
          - ``last_bin_end_unknown``, a boolean indicating if the end of the
            last bin is known or not. If bins are left-closed, then it is
            possible the end of the last bin is not known. In this case,
            de-facto, this unknown bin end is supposed to be positioned after
            all snapshots.

    bin_on : Optional[str], default None
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `TimeGrouper`, its `key` parameter is used instead.
        If 'bin_on' is set, its consistence with ``bin_by.key`` parameter is
        then checked.
    ordered_on : Optional[str], default None
        Name of an existing ordered column in 'data'. When setting it, it is
        then forwarded to 'bin_by' Callable.
        This parameter is compulsory if 'snap_by' is set. Values derived from
        'snap_by' (either a TimeGrouper or a Series) are compared to ``bin_ends``,
        themselves derived from ``data[ordered_on]``.
    snap_by : Optional[Union[TimeGrouper, Series]], default None
        Values positioning the points of observation, either derived from a
        pandas TimeGrouper, or contained in a pandas Series.
    buffer : Optional[dict], default None
        Dict of 2 dict.
          - first dict, with key `"bin"` embed values from previous binning
            process, set by 'bin_by' when it is a Callable, or by the internal
            function ``by_scale`` if 'bin_by' is a TimeGrouper. These values are
            required when restarting the binning process with new seed data.
          - second dict, with key `"snap"` embed values from previous
            snapshotting process, set by 'by_scale'. Similarly, these values
            are required to allow restarting the snapshotting process with new
            seed data.

    Returns
    -------
    Tuple made of 6 items.
      - ``next_chunk_starts``,  an ordered one-dimensional numpy array of int,
        specifying for each bin and snapshot the row indice at which starts the
        next one.
      - ``bin_indices``, a one-dimensional array of int, specifying which
        value in ``next_chunk_starts`` relates to a bin (as opposed to a
        snapshot)
      - ``bin_labels``, a pandas Series specifying for each bin its label.
      - ``n_null_bins``, an int, indicating how many bins are empty.
      - ``snap_labels``, a pandas Series specifying for each snapshot its
        label.
      - ``n_max_null_snaps``, an int, specifying how many at most there are
        empty snapshots. This figure is an upper bound.

    Notes
    -----
    When implementing `bin_by` Callable the developer should take care that
    ``next_chunk_starts``, ``chunk_labels`` and``chunk_ends`` that are returned
    by 'bin_by' are expected to be all of the same size, i.e. the total number
    of bins that are expected, including empty ones.

    Also, when implementing it for repetitive calls, care should be taken
    that `bin_by` keeps in the 'buffer' parameter all the data needed to:
      - create the correct number of bins that would be in-between the data
        processed at the previous aggregation iteration, and the new data.
        This has to show in 'next_chunk_starts' array that is returned.
      - start with same bin label as previous iteration when using snapshots.

    Having the same bin label between both iterations when using snapshots will
    ensure:
      - that the bin with previous aggregation results is overwritten (ok, not
        necessarily meaningful if agg results have not changed in case there
        has been no new data in this bin).
      - even if this bin is empty at restart, in the case of snapshotting, it
        is necessary when this bin ends that new empty snapshots before its end
        correctly forward past results, and that new empty snapshots after this
        end are correctly accounted for as empty chunks. For this reason, when
        using snapshots, a check ensures that same bin label is used between
        two successive iterations.

    Still for repetitive calls of 'bin_by', care has to be taken that:
      - the last bin is not an empty one.
      - the last bin does cover the full size of data.

    If not, exceptions will be raised.

    When using snapshots, values defined by ``snap_by`` are considered the
    "points of isolated observation". At such a point, an observation of the
    "on-going" bin is made. In case of snapshot(s) positioned exactly on
    segment(s) ends, at the same row index in data, the observation point will
    always come before the bin end.

    """
    # TODO : split 'by_scale' into 'by_pgrouper' and 'by_scale'.
    # TODO : make some tests validating use of 'by_scale' as 'bin_by' parameter.
    # (when user-provided 'bin_by' is  a Series or a tuple of Series)
    # TODO : consider transitioning 'bin_by' and 'snap_by' into a class.
    # Probably, below initiatialization is to be part of a template class, to
    # be run at child class instantiation.
    if not isinstance(bin_by, dict):
        bin_by = setup_segmentby(bin_by, bin_on, ordered_on, snap_by)
    if bin_by[KEY_SNAP_BY] is not None:
        # 'bin_by[KEY_SNAP_BY]' is not none if 'snap_by' is a TimeGrouper.
        # Otherwise, it can be a DatetimeIndex or a Series.
        snap_by = bin_by[KEY_SNAP_BY]
    buffer_bin, buffer_snap = setup_mainbuffer(buffer, snap_by is not None)
    ordered_on = bin_by[KEY_ORDERED_ON]
    if ordered_on:
        # Check 'ordered_on' is an ordered column.
        if not (
            (
                data[ordered_on].dtype == DTYPE_DATETIME64
                and (data[ordered_on].diff().iloc[1:] >= Timedelta(0)).all()
            )
            or (
                data[ordered_on].dtype != DTYPE_DATETIME64
                and (data[ordered_on].diff().iloc[1:] >= 0).all()
            )
        ):
            raise ValueError(
                f"column '{ordered_on}' is not ordered. It has to be for "
                "'cumsegagg' to operate faultlessly.",
            )
    on = data.loc[:, bin_by[KEY_ON_COLS]]
    # 'bin_by' binning.
    (
        next_chunk_starts,
        bin_labels,
        n_null_bins,
        bin_closed,
        bin_ends,
        unknown_last_bin_end,
    ) = bin_by[KEY_BIN_BY](on=on, buffer=buffer_bin)
    # Check consistency of 'bin_by' results.
    # TODO : consider transitioning 'bin_by' and 'snap_by' into a class.
    # Integrate below checks within a template class.
    # Some checks may probably be managed at class instantiation.
    # Others at runtime.
    if bin_closed != LEFT and bin_closed != RIGHT:
        raise ValueError(f"'bin_closed' has to be set either to '{LEFT}' or to '{RIGHT}'.")
    if not isinstance(bin_labels, pSeries):
        # Because `iloc` is used afterwards, `bin_labels` has to be a pandas
        # Series.
        raise TypeError("'bin_labels' has to be a pandas Series.")
    n_bins = len(next_chunk_starts)
    if n_bins != len(bin_labels):
        raise ValueError("'next_chunk_starts' and 'chunk_labels' have to be of the same size.")
    if n_bins != len(bin_ends):
        raise ValueError("'next_chunk_starts' and 'chunk_ends' have to be of the same size.")
    if isinstance(buffer, dict) and next_chunk_starts[-1] != len(data):
        raise ValueError(
            "series of bins have to cover the full length of 'data'. "
            f"But last bin ends at row {next_chunk_starts[-1]} "
            f"excluded, while size of data is {len(data)}.",
        )
    if buffer is not None:
        # A buffer that is not 'None' means a restart is expected.
        if n_bins > 1 and next_chunk_starts[-2] == len(on):
            # In case a user-provided 'bin_by()' Callable is used, check if there
            # are empty trailing bins. If there are, and that restart are expected
            # (use of 'buffer'), then raise error, this it not allowed, as it would
            # lead to wrong results in 'jcumsegagg()'.
            raise ValueError(
                "there is at least one empty trailing bin. "
                "This is not possible if planning to restart on new "
                "data in a next iteration.",
            )
        if KEY_LAST_BIN_LABEL in buffer and buffer[KEY_LAST_BIN_LABEL] != bin_labels.iloc[0]:
            # When using snapshots, and in case of multiple calls, check that
            # label of last bin (previous iteration) is same than label of
            # first bin (current iteration).
            raise ValueError(
                f"not possible to have label '{buffer[KEY_LAST_BIN_LABEL]}' "
                "of last bin at previous iteration different than label "
                f"'{bin_labels.iloc[0]}' of first bin at current iteration.",
            )
    if snap_by is not None:
        # Define points of observation
        (next_snap_starts, snap_labels, n_max_null_snaps, _, snap_ends, _) = by_scale(
            on=data.loc[:, ordered_on],
            by=snap_by,
            closed=bin_closed,
            buffer=buffer_snap,
        )
        # Consolidate 'next_snap_starts' into 'next_chunk_starts'.
        # If bins are left-closed, the end of the last bin can possibly be
        # unknown yet.
        # If a snapshot (observation point) is also set at end of data
        # (a snapshot position is always known, because it is either
        # derived from a pandas TimeGrouper, or an iterable of ordered values),
        # then 'merge_sorted()' cannot sort them one to the other (end of last
        # bin with last snapshot).
        # In this case, we force the bin end to be after the last snapshot.
        # The logic is that we want both to know the last bin and last
        # snapshot while this last bin is in-progress.
        # Having the bin end before the snapshot would on the opposite
        # reset the data and the resulting snapshot would be a null one.
        next_chunk_starts, bin_indices = mergesort(
            labels=(next_snap_starts, next_chunk_starts),
            keys=(snap_ends, bin_ends),
            force_last_from_second=unknown_last_bin_end,
        )
        # Take indices of 'next_chunk_starts' corresponding to bins that are
        # followed right after by a snapshot.
        # ('append=len(next_chunk_starts)' in 'nonzero()' allows to simulate a
        # configuration in which the last indices in 'next_chunk_starts' is
        # that of a bin, hence to detect if a snapshot is after the actual
        # (real) last bin. Without it, a snapshot after the last bin would not
        # be detected and if needed, accounted for.)
        indices_of_bins_followed_by_a_snap = bin_indices[
            nonzero(ndiff(bin_indices, append=len(next_chunk_starts)) - 1)[0]
        ]
        # Check if the 'next_chunk_starts' for these bins equal that of the
        # snapshot that follows. If yes, then those are potential additional
        # null snapshots.
        n_max_null_snaps += len(
            nonzero(
                (
                    next_chunk_starts[indices_of_bins_followed_by_a_snap]
                    - next_chunk_starts[indices_of_bins_followed_by_a_snap + 1]
                )
                == 0,
            )[0],
        )
    else:
        bin_indices = NULL_INT64_1D_ARRAY
        snap_labels = None
        n_max_null_snaps = 0
    # Keep track of last bin labels for checking at next iteration.
    # Check is managed at upper level in `cumsegagg`.
    if buffer is not None:
        buffer[KEY_LAST_BIN_LABEL] = bin_labels.iloc[-1]
    return (
        next_chunk_starts,
        bin_indices,
        bin_labels,
        n_null_bins,
        snap_labels,
        n_max_null_snaps,
    )
