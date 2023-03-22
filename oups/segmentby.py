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
from numpy import ndarray
from numpy import ndenumerate
from numpy import nonzero
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import IntervalIndex
from pandas import Series
from pandas import Timedelta
from pandas import date_range
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
# Keys for 'by_x_rows'.
KEY_ROWS_IN_LAST_BIN = "rows_in_last_bin"
KEY_LAST_KEY = "last_key"
# Keys for 'bin_by' when a dict
ON_COLS = "on_cols"
BIN_BY = "bin_by"
ORDERED_ON = "ordered_on"
SNAP_BY = "snap_by"


@njit(
    [
        "Tuple((int64[:], int64))(int64[:], int64[:], boolean)",
        "Tuple((int64[:], int64))(float64[:], float64[:], boolean)",
    ],
)
def _next_chunk_starts(
    data: ndarray,
    right_edges: ndarray,
    right: bool,
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
            next_chunk_starts[b_idx_loc:] = _d_idx + 1
            n_null_chunks += len(next_chunk_starts[b_idx_loc:]) - 1
            return next_chunk_starts, n_null_chunks
        else:
            next_chunk_starts[b_idx_loc] = _d_idx
            if prev_d_idx == _d_idx:
                n_null_chunks += 1
            else:
                prev_d_idx = _d_idx
    return next_chunk_starts, n_null_chunks


def by_scale(
    on: Series,
    by: Union[Grouper, Series],
    closed: Optional[str] = None,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, Series, int, str, Series, bool]:
    """Segment an ordered DatetimeIndex as per pandas Grouper .

    Parameters
    ----------
    on : Series
        Ordered date time index over which performing the binning as defined
        per 'by'.
    by : Grouper
        Setup to define binning as a pandas Grouper
    closed : str, default None
        Optional string, specifying if intervals defined by 'by' are left or
        right closed. This parameter is not used if 'by' is a pandas Grouper.
    buffer : dict
        Dict to keep parameters allowing chaining calls to 'by_scale':

    Returns
    -------
    Tuple[ndarray, Series, int, str, Series, bool]
        The first 3 items are used in 'cumsegagg' in all situations.
          - ``next_chunk_starts``, a one-dimensional array of `int` specifying
            the row indices of the next-bin starts, for each bin. Successive
            identical indices implies empty bins, except the first.
          - ``chunk_labels``, a pandas Series specifying for each bin its
            label. Labels are defined as per 'on' pandas Grouper.
          - ``n_null_chunks``, an int, the number of null chunks identified in
            'on'.

        The 3 last items are used only if both bins and snapshots are generated
        in 'cumsegagg'.
          - ``chunk_closed``, a str, indicating if bins are left or right
            closed, as per 'by' pandas Grouper.
          - ``chunk_ends``, a pandas Series containing bin ends, as per 'by'
            pandas Grouper.
          - ``unknown_last_chunk_end``, a boolean, always `False`, specifying
            that the last bin end is known. This is because bin ends are fully
            specified as per 'by' pandas Grouper.

    """
    if isinstance(by, Grouper):
        first = on.iloc[0]
        # In case 'by' is for snapshotting, and 'closed' is not set, take care
        # to use 'closed' provided.
        closed = by.closed if closed is None else closed
        start, end = gtre(
            first=first,
            last=on.iloc[-1],
            freq=by.freq,
            closed=closed,
            origin=by.origin,
            offset=by.offset,
        )
        edges = date_range(start, end, freq=by.freq)
        chunk_ends = edges[1:]
        chunk_labels = chunk_ends if by.label == RIGHT else edges[:-1]
    else:
        # Case 'by' is a Series.
        chunk_ends = by
        chunk_labels = by
    if chunk_ends.dtype == DTYPE_DATETIME64:
        next_chunk_starts, n_null_chunks = _next_chunk_starts(
            on.to_numpy(copy=False).view(DTYPE_INT64),
            chunk_ends.to_numpy(copy=False).view(DTYPE_INT64),
            closed == RIGHT,
        )
    else:
        next_chunk_starts, n_null_chunks = _next_chunk_starts(
            on.to_numpy(copy=False),
            chunk_ends.to_numpy(copy=False),
            closed == RIGHT,
        )
    return next_chunk_starts, chunk_labels, n_null_chunks, closed, chunk_ends, False


def by_x_rows(
    on: Union[pDataFrame, Series],
    by: Optional[int] = 4,
    closed: Optional[str] = LEFT,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, Series, int, str, Series, bool]:
    """Segment by group of x rows.

    Dummy binning function for testing 'cumsegagg' with 'bin_by' set as a
    Callable.

    Parameters
    ----------
    on : Union[pDataFrame, Series]
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
          - 'rows_in_last_bin', an int specifying the number of rows in last
            (and possibly incomplete) bin from the previous call to
            'bin_x_rows'.
          - 'last_key', key of last (possibly incomplete) bin.

    Returns
    -------
    Tuple[ndarray, Series, int, str, Series, bool]
        The first 3 items are used in 'cumsegagg' in all situations.
          - ``next_chunk_starts``, a one-dimensional numpy array of int,
            specifying for each bin the row indice at which starts the next
            bin.
          - ``bin_labels``, a pandas Series specifying for each bin its label.
            Labels are first value in bin taken in last column of 'on'.
          - ``n_null_bins``, an int, always ``0``.

        The 3 last items are used only if both bins and snapshots are generated
        in 'cumsegagg'.
          - ``bin_closed``, a str, always ``"left"``,  indicating that the bins
            are left closed.
          - ``bin_ends``, a pandas Series made of values from the last columns
            of 'on' (which is either single-column or two-column) and
            indicating the "position" of the bin end, which is marked by the
            start of the next bin, excluded. The end of the last bin being
            unknown by definition (because is excluded), the last value is not
            relevant. This information is flagged by ``unknown_last_bin_end```.
          - ``unknown_last_bin_end``, a boolean, always `True`, specifying that
            the last bin end is unknown. This is because bins are lef-closed,
            meaning that their end is excluded. Hence, the last bin is always
            "in-progress".

    """
    len_on = len(on)
    if isinstance(on, pDataFrame):
        # Keep only last column, supposed to be `ordered_on` column.
        on = on.iloc[:, -1]
    # Derive number of rows in first bins (cannot be 0) and number of bins.
    rows_in_last_bin = (
        buffer[KEY_ROWS_IN_LAST_BIN]
        if (buffer is not None and KEY_ROWS_IN_LAST_BIN in buffer)
        else 0
    )
    rows_in_first_bin = min(by - rows_in_last_bin if rows_in_last_bin != by else by, len_on)
    n_rows_for_new_bins = len_on - rows_in_first_bin
    n_bins = ceil(n_rows_for_new_bins / by) + 1
    # Define 'next_chunk_starts'.
    next_chunk_starts = arange(
        start=rows_in_first_bin, stop=n_bins * by + rows_in_first_bin, step=by
    )
    # Make a copy and arrange for deriving 'chunk_starts', required for
    # defining bin labels. 'bin_labels' are derived from last column (is then
    # 'ordered_on' and if not, is 'bin_on'). Bin labels are 1st value in bin.
    chunk_starts = next_chunk_starts.copy() - by
    # Correct start of 1st chunk.
    chunk_starts[0] = 0
    bin_labels = on.iloc[chunk_starts].reset_index(drop=True)
    if buffer is not None:
        # Correct 1st label if not a new bin.
        if KEY_LAST_KEY in buffer and rows_in_first_bin != by:
            bin_labels.iloc[0] = buffer[KEY_LAST_KEY]
        # Update 'buffer[rows_in_last_bin]' with number of rows in last bin for
        # next run.
        buffer[KEY_ROWS_IN_LAST_BIN] = fmod(n_rows_for_new_bins, by) or by
        # Update'buffer[last_key]' with last bin label.
        buffer[KEY_LAST_KEY] = bin_labels.iloc[-1]
    # 'bin_ends' has no end for last bin, because it is unknown.
    # Temporarily adjust 'next_chunk_start' of last bin to last index.
    next_chunk_starts[-1] = len_on - 1
    bin_ends = on.iloc[next_chunk_starts].reset_index(drop=True)
    # Reset 'next_chunk_start' of last bin.
    next_chunk_starts[-1] = len_on
    return next_chunk_starts, bin_labels, 0, closed, bin_ends, True


def mergesort(
    labels: Tuple[ndarray, ndarray],
    keys: Optional[Tuple[ndarray, ndarray]] = None,
    force_last_from_second: Optional[bool] = False,
) -> Tuple[ndarray, ndarray]:
    """Mergesort labels from keys.

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
            "not possible to have arrays of different length for first labels and keys arrays."
        )
    if len(keys2) != len_labels2:
        raise ValueError(
            "not possible to have arrays of different length for second labels and keys arrays."
        )
    if force_last_from_second:
        len_tot = len_labels1 + len_labels2
        sort_indices = full(len_tot, len_tot - 1, dtype=DTYPE_INT64)
        sort_indices[:-1] = argsort(concatenate((keys1, keys2[:-1])), kind="mergesort")
    else:
        sort_indices = argsort(concatenate(keys), kind="mergesort")
    return concatenate(labels)[sort_indices], nonzero(len_labels1 <= sort_indices)[0]


def setup_segmentby(
    bin_by: Union[Grouper, Callable],
    bin_on: Optional[str] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[Grouper, Series]] = None,
) -> Dict[str, Union[Callable, str]]:
    """Check and setup parameters to operate data segmentation.

    Parameters
    ----------
    bin_by : Union[Grouper, Callable]
       A pandas Grouper or a Callable to perform segmentation.
    bin_on : Optional[str]
       Name of the column onto which performing the segmentation.
    ordered_on : Optional[str]
       Name of the column containing ordered data and to use when snapshotting.
       With this column, snapshots (points of observation) can be positioned
       with respect to bin ends.
    snap_by : Optional[Union[Grouper, IntervalIndex]]
       A pandas Grouper or a pandas Series defining the snapshots (points of
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
    if isinstance(bin_by, Grouper):
        # 'bin_by' is a Grouper.
        bin_by_closed = bin_by.closed
        if bin_by.key:
            if bin_on:
                if bin_by.key != bin_on:
                    raise ValueError(
                        "not possible to set 'bin_by.key' and 'bin_on' to different values."
                    )
            else:
                bin_on = bin_by.key
        elif not bin_on:
            raise ValueError("not possible to set both 'bin_by.key' and 'bin_on' to `None`.")
        if ordered_on and ordered_on != bin_on:
            raise ValueError("not possible to set 'bin_on' and 'ordered_on' to different values.")
        elif not ordered_on:
            # Case 'ordered_on' has not been provided but 'bin_on' has been.
            # Then set 'ordered_on' to 'bin_on'. this is so because 'bin_by' is
            # a Grouper.
            ordered_on = bin_on
        bin_by = partial(by_scale, by=bin_by)
    else:
        # 'bin_by' is a Callable.
        if bin_on is None:
            raise ValueError("not possible to set 'bin_on' to `None`.")
    if snap_by is not None:
        if isinstance(snap_by, Grouper):
            if snap_by.key:
                if ordered_on is None:
                    ordered_on = snap_by.key
                elif snap_by.key != ordered_on:
                    raise ValueError(
                        "not possible to set 'ordered_on' and 'snap_by.key' to different values."
                    )
            if bin_by_closed and snap_by.closed != bin_by_closed:
                raise ValueError(
                    "not possible to set 'bin_by.closed' and 'snap_by.closed' to different values."
                )
        elif not ordered_on:
            # Case 'snap_by' is not a Grouper.
            raise ValueError(
                "not possible to leave 'ordered_on' to `None` in case of snapshotting."
            )
    return {
        BIN_BY: bin_by,
        ON_COLS: [bin_on, ordered_on] if ordered_on and ordered_on != bin_on else bin_on,
        ORDERED_ON: ordered_on,
        SNAP_BY: snap_by if isinstance(snap_by, Grouper) else None,
    }


def setup_mainbuffer(buffer: dict) -> Tuple[dict, dict]:
    """Return 'buffer_bin' and 'buffer_snap' from main buffer.

    Parameters
    ----------
    buffer : dict
        Main buffer, either containing only values for 'buffer_bin', or only
        two keys `"bin"` and `"snap"` providing a separate dict for each of the
        binning and snapshotting processes.

    Returns
    -------
    Tuple[dict, dict]
        The first dict is the binning buffer.
        The second dict is the snapshotting buffer.
    """
    if buffer is not None:
        if KEY_SNAP in buffer:
            return buffer[KEY_BIN], buffer[KEY_SNAP]
        elif KEY_BIN in buffer:
            return buffer[KEY_BIN], None
        else:
            return buffer
    else:
        return None, None


def segmentby(
    data: pDataFrame,
    bin_by: Union[Grouper, Callable, dict],
    bin_on: Optional[str] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[Grouper, IntervalIndex]] = None,
    buffer: Optional[dict] = None,
) -> Tuple[ndarray, ndarray, Series, int, Series, int]:
    """Identify starts of segments in data, either bins or optionally snapshots.

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
    bin_by : Union[Grouper, Callable, dict]
        Callable or pandas Grouper to perform binning.
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
            dereived from a pandas Grouper.
          - 'ordered_on', a str, its definitive value.

        It has then to return a tuple made of 6 items. There are 3 items used
        whatever if snapshotting is used or not.
          - ``next_chunk_starts``, a one-dimensional array of `int`, specifying
            the row index at which the next bin starts (included) as found in
            ``bin_on``.
            If the same indices appear several times, it means that
            corresponding bins are empty, except the first one. In this case,
            corresponding rows in aggregation result will be filled with null
            values.
          - ``chunk_labels``, a pandas Series which values are expected to be
            all bin labels, incl. those of empty bins, as they will appear in
            aggregation results. Labels can be of any type.
            In case of restarting the aggregation with new seed data, care
            should be taken so that the label of the first bin is the same as
            that of the last bin from previous iteration if it has been the
            same bin. Having the same label between both iteration will ensure
            that the bin with previous aggregation results is overwritten.
          - ``n_null_chunks``, an `int` indicating the number of empty bins.

        The 3 last items are used only in case of snapshotting (``snap_by`` is
        different than ``None``).
          - ``chunk_closed``, a str, either `'right'` or `'left'`, indicating
            if bins are left or right-closed (i.e. if ``chunk_ends`` is
            included or excluded in the bin).
          - ``chunk_ends``, an optional pandas Series, specifying the ends of
            bins with values derived from ``data[ordered_on]`` column. If
            snapshotting, then points of observation (defined by ``snap_by``)
            are positioned with respect to the bin ends. This data allows
            sorting snapshots with respect to bins in case they start/end at
            the same row index in data.
            ``chunk_ends`` is not required if no snapshotting. If not used, set
            to None.
          - ``last_chunk_end_unknown``, a boolean indicating if the end of the
            last bin is known or not. If bins are left-closed, then it is
            possible the end of the last bin is not known. In this case, care
            is taken to position the last bin end after any last snapshot.

    bin_on : Union[str, None]
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `Grouper`, its `key` parameter is used instead.
        If 'bin_on' is set, its consistence with ``bin_by.key`` parameter is
        then checked.
    ordered_on : Union[str, None]
        Name of an existing ordered column in 'data'. When setting it, it is
        then forwarded to 'bin_by' Callable.
        This parameter is compulsory if 'snap_by' is set. Values derived from
        'snap_by' (either a Grouper or a Series) are compared to ``bin_ends``,
        themselves derived from ``data[ordered_on]``.
    snap_by : Union[Grouper, Series, None]
        Values positioning the points of observation, either derived from a
        pandas Grouper, or contained in a pandas Series.
    buffer : Optional[dict], default None
        Dict of 2 dict.
          - first dict, with key `"bin"` embed values from previous binning
            process, set by 'bin_by' when it is a Callable, or by the internal
            function ``by_scale`` if 'bin_by' is a Grouper. These values are
            required when restarting the binning process with new seed data.
          - second dict, with key `"snap"` embed values from previous
            snapshotting process, set by 'by_scale'. Similarly, these values are
            required to allow restarting the snapshotting process with new seed
            data.

    Returns
    -------
    Tuple made of the 6 items.
      - ``bin_indices``, a one-dimensional array of int
      - ``null_snap_indices``

    Notes
    -----
    When using snapshots, values defined by ``snap_by`` are considered the
    "points of isolated observation". At such a point, an observation of the
    "on-going" bin is made. In case of snapshot(s) positioned exactly on
    segment(s) ends, at the same row index in data, the observation point will
    always come before the bin end.

    When implementing `bin_by` Callable for repetitive calls, the developer
    should take care that `bin_by` keeps in the 'buffer' parameter all the data
    needed to:
      - create the correct number of bins that would be in-between the data
        processed at the previous aggregation iteration, and the new data. This
        has to show in 'next_chunk_starts' array that is returned.
      - appropriately label the first bin.
        - either it is a new bin, different than the last one from previous
          aggregation iteration. Then the label of the new bin has to be
          different than that of the last one from previous iteration.
        - or it is the same bin that is continuing. Then the label has be the
          same. This ensures that when recording the new aggregation result,
          the data from the previous iteration (last bin was in-progress, i.e.
          incomplete) is overwritten.

    Also, ``next_chunk_starts``, ``chunk_labels`` and``chunk_ends`` that are
    returned by 'bin_by' are expected to be all of the same size, i.e. the
    total number of bins that are expected, including empty ones.
    """
    if not isinstance(bin_by, dict):
        bin_by = setup_segmentby(bin_by, bin_on, ordered_on, snap_by)
    buffer_bin, buffer_snap = setup_mainbuffer(buffer)
    ordered_on = bin_by[ORDERED_ON]
    if ordered_on:
        # Check 'ordered_on' is an ordered column.
        if (
            data[ordered_on].dtype == DTYPE_DATETIME64
            and (data[ordered_on].diff() >= Timedelta(0)).all()
            or data[ordered_on].dtype != DTYPE_DATETIME64
            and (data[ordered_on].diff() >= 0).all()
        ):
            raise ValueError(
                f"column '{ordered_on}' is not ordered. It has to be for 'cumsegagg' to operate faultlessly."
            )
    on = data.loc[:, bin_by[ON_COLS]]
    # 'bin_by' binning.
    (
        next_chunk_starts,
        bin_labels,
        n_null_bins,
        bin_closed,
        bin_ends,
        last_bin_end_unknown,
    ) = bin_by[BIN_BY](on=on, buffer=buffer_bin)
    # Some checks.
    if bin_closed != LEFT and bin_closed != RIGHT:
        raise ValueError("'chunk_closed' has to be set either to 'left' or to 'right'.")
    n_bins = len(next_chunk_starts)
    if n_bins != len(bin_labels):
        raise ValueError("'next_chunk_starts' and 'chunk_labels' have to be of the same size.")
    if bin_ends is not None and n_bins != len(bin_ends):
        raise ValueError("'next_chunk_starts' and 'chunk_ends' have to be of the same size.")
    if bin_by[SNAP_BY] is not None:
        # 'bin_by[SNAP_BY]' is not none if 'snap_by' is a Grouper.
        # Otherwise, it can be a DatetimeIndex for instance or a Series.
        snap_by = bin_by[SNAP_BY]
    if snap_by is not None:
        # Define points of observation
        (
            next_snap_starts,
            _,
            n_max_null_snaps,
            _,
            snap_ends,
            _,
        ) = by_scale(on=data.loc[:, ordered_on], by=snap_by, closed=bin_closed, buffer=buffer_snap)
        # Consolidate 'next_snap_starts' into 'next_chunk_starts'.
        # If bins are left-closed, the end of the last bin can possibly be
        # unknown yet.
        # If a snapshot (observation point) is also set at end of data
        # (a snapshot position is always known, because it is either
        # derived from a pandas Grouper, or an iterable of ordered values),
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
            force_last_from_second=last_bin_end_unknown,
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
                == 0
            )[0]
        )
    else:
        bin_indices = NULL_INT64_1D_ARRAY
        snap_ends = None
        n_max_null_snaps = 0
    return next_chunk_starts, bin_indices, bin_labels, n_null_bins, snap_ends, n_max_null_snaps
