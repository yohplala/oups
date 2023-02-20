#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from typing import Callable, Dict, List, Tuple, Union

from numba import boolean
from numba import float64
from numba import guvectorize
from numba import int64
from numpy import NaN as nNaN
from numpy import dtype
from numpy import isin as nisin
from numpy import max as nmax
from numpy import ndarray
from numpy import ndenumerate
from numpy import ones
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import IntervalIndex
from pandas import NaT as pNaT
from pandas import Series
from pandas import Timedelta
from pandas import date_range
from pandas.core.resample import _get_timestamp_range_edges as gtre

from oups.jcumsegagg import AGG_FUNC_IDS
from oups.jcumsegagg import jcsagg
from oups.utils import merge_sorted


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_FLOAT64 = dtype("float64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
NULL_INT64_1D_ARRAY = zeros(0, DTYPE_INT64)
NULL_INT64_2D_ARRAY = NULL_INT64_1D_ARRAY.reshape(0, 0)
# Null values.
NULL_DICT = {DTYPE_INT64: pNA, DTYPE_FLOAT64: nNaN, DTYPE_DATETIME64: pNaT}


def setup_cgb_agg(
    agg: Dict[str, Tuple[str, str]], data_dtype: Dict[str, dtype]
) -> Dict[dtype, Tuple[List[str], List[str], ndarray, ndarray]]:
    """Construct chaingrouby aggregation configuration.

    Parameters
    ----------
    agg : Dict[str, Tuple[str, str]]
        Dict specifying aggregation in the form
        ``'out_col_name' : ('in_col_name', 'function_name')``
    data_dtype : Dict[str, dtype]
        Dict specifying per column name its dtype. Typically obtained with
        ``df.dtypes.to_dict()``

    Returns
    -------
    Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]
         Dict 'cgb_agg_cfg' in the form
         ``{dtype: List[str], 'cols_name_in_data'
                              column name in input data, with this dtype,
                   List[str], 'cols_name_in_res'
                              expected column names in aggregation result,
                   ndarray[int64], 'cols_idx'
                                 Three dimensional array of ``int``, one row
                                 per aggregation function.
                                 Row index for a given aggregation function is
                                 defined by global constants 'ID_FIRST', etc...
                                 Per row (2nd dimension), column indices in
                                 'data' and 'res' to which apply corresponding
                                 aggregation function.
                                 Any value in column past the number of
                                 relevant columns is not used.
                                 In last dimension, index 0 gives indices of
                                 columns in 'data'. Index 1 gives indices of
                                 columns in 'res'.
                   ndarray[int64], 'n_cols'
                                1d-array, length corresponding to the total
                                number of aggregation functions.
                                Row indices correspond aggregation function
                                indices
                                It contains the number of input columns in
                                data, to which apply this aggregation
                                function.
           }``
    """
    cgb_agg_cfg = {}
    n_agg_funcs = len(AGG_FUNC_IDS)
    # Step 1.
    for out_col, (in_col, func) in agg.items():
        if in_col not in data_dtype:
            raise KeyError(f"{in_col} not in input data.")
        else:
            dtype_ = data_dtype[in_col]
        try:
            tup = cgb_agg_cfg[dtype_]
        except KeyError:
            cgb_agg_cfg[dtype_] = [
                [],  # 'cols_name_in_data'
                [],  # 'cols_name_in_res'
                [],  # 'agg_func_idx' (temporary)
                [],  # 'cols_idx'
                # 'n_cols'
                zeros(n_agg_funcs, dtype=DTYPE_INT64),
            ]
            tup = cgb_agg_cfg[dtype_]
        # 'in_col' / name / 1d list.
        cols_name_in_data = tup[0]
        if in_col in cols_name_in_data:
            in_col_idx = cols_name_in_data.index(in_col)
        else:
            in_col_idx = len(cols_name_in_data)
            cols_name_in_data.append(in_col)
        # 'out_col' / name / 1d list.
        cols_name_in_res = tup[1]
        out_col_idx = len(cols_name_in_res)
        cols_name_in_res.append(out_col)
        # Set list of function id (temporary buffer 'agg_func_idx').
        agg_func_idx = tup[2]
        if (func_id := AGG_FUNC_IDS[func]) in agg_func_idx:
            func_idx = agg_func_idx.index(func_id)
        else:
            func_idx = len(agg_func_idx)
            agg_func_idx.append(AGG_FUNC_IDS[func])
        # 'cols_idx'
        cols_idx = tup[3]
        if len(cols_idx) <= func_idx:
            # Create list for this aggregation function.
            cols_idx.append([[in_col_idx, out_col_idx]])
        else:
            # Add this column index for this aggregation function.
            cols_idx[func_idx].append([in_col_idx, out_col_idx])
    # Step 2.
    for conf in cgb_agg_cfg.values():
        # Remove 'agg_func_idx'.
        agg_func_idx = conf.pop(2)
        n_funcs = len(agg_func_idx)
        cols_idx = conf[2]
        # 'n_cols': 1d array of length the total number of existing agg funcs.
        n_cols = conf[3]
        for func_idx in range(n_funcs):
            # Retrieve number of columns in input data for this aggregation
            # function.
            # 'agg_func_idx[func_idx]' is aggregation function id.
            n_cols[agg_func_idx[func_idx]] = len(cols_idx[func_idx])
        # Transform list of list into 2d array.
        max_cols = nmax(n_cols)
        cols = zeros((n_agg_funcs, max_cols, 2), dtype=DTYPE_INT64)
        for func_idx in range(n_funcs):
            # 'agg_func_idx[func_idx]' is aggregation function id.
            func_id = agg_func_idx[func_idx]
            cols[func_id, : n_cols[func_id], :] = cols_idx[func_idx]
        conf[2] = cols
    return cgb_agg_cfg


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
            # Array 'data' terminated and loop stayed in previous bin.
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
    bins : IntervalIndex
        IntervalIndex defining each bin by its left and right edges, and how
        it is closed, right or left.
    next_chunk_starts : ndarray
        One-dimensional array of `int` specifying the row indices of the
        next-bin starts, for each bin. Successive identical indices implies
        empty bins, except the first bin in the series.
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
    return IntervalIndex.from_breaks(edges, closed=by.closed), next_chunk_starts, n_null_chunks[0]


def cumsegagg(
    data: pDataFrame,
    agg: Union[Dict[str, Tuple[str, str]], Dict[dtype, list]],
    bin_by: Union[Grouper, Callable],
    bin_on: Union[str, None] = None,
    binning_buffer: dict = None,
    ordered_on: Union[str, None] = None,
    snap_by: Union[Grouper, IntervalIndex, None] = None,
    allow_bins_snaps_disalignment: bool = False,
) -> Union[pDataFrame, Tuple[pDataFrame, pDataFrame]]:
    """Cumulative segmented aggregations, with optional snapshotting.

    In this function, "snapshotting" is understood as the action of making
    isolated observations. When using snapshots, values derived from
    ``snap_by`` Grouper (or contained in ``snap_by`` IntervalIndex) are
    considered the "points of isolated observation". At a given point, an
    observation of the "on-going" segment (aka bin) is made.
    Because segments are continuous, any row of the dataset falls in a segment.

    Parameters
    ----------
    data: pDataFrame
        A pandas Dataframe containing the columns over which binning (relying
        on ``bin_on`` column), performing aggregations and optionally
        snapshotting (relying on column pointed by ``snap_by.key`` or
        ``snap_by.name`` depending if a Grouper or an IntervalIndex
        respectively).
        If using snapshots ('snap_by' parameter), then the column pointed by
        ``snap_by.key`` (or ``snap_by.name``) has to be ordered.
    agg : dict
        Definition of aggregation.
        If in the form ``Dict[str, Tuple[str, str]]`` (typically a form
        compatible with pandas aggregation), then it is transformed in the 2nd
        form ``Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]]``.

        - in the form ``Dict[str, Tuple[str, str]]``

          - keys are ``str``, requested output column name
          - values are ``tuple`` with 1st component a ``str`` for the input
            column name, and 2nd component a ``str`` for aggregation function
            name.

        - the 2nd form is that returned by the function ``setup_cgb_agg``.

    bin_by : Union[Grouper, Callable]
        Callable or pandas Grouper to perform binning.
        If a Callable, is called with following parameters:
        ``bin_by(on, binning_buffer)``
        where:

          - ``on``,

            - either ``ordered_on`` is ``None``. ``on`` is then a pandas Series
              made from ``data[bin_on]`` column.
            - or ``ordered_on`` is provided and is different than ``bin_on``.
              Then ``on`` is a 2-column pandas Dataframe made of
              ``data[[bin_on, ordered_on]``. Values from ``data[ordered_on]``
              can be used advantageously as bin labels.
              Also, values from ``data[ordered_on]`` have to be used when
              'snap_by' is set. See below.

          - ``binning_buffer``, see corresponding parameter of this function.

        It has then to return a Tuple made of 3 or 5 items. There are 3 items
        if no snapshotting.

          - ``bin_labels``, a pandas Series or one-dimensional array, which
            values are expected to be all bin labels, incl. for empty
            bins, as they will appear in aggregation results. Labels can be of
            any type.

          - ``next_chunk_starts``, a one-dimensional array of `int`, specifying
            the row index at which the next bin starts (included) as found in
            ``bin_on``.
            If the same index appears several time, it means that corresponding
            bins are empty, except the first one. In this case, corresponding
            rows in aggregation result will be filled with null values.

          - ``n_null_bins``, an `int` indicating the number of empty bins.

        In case of snapshotting (``snap_by`` is different than ``None``), the 2
        additional items are:

          - ``bin_ends``, a one dimensional array, specifying with values
            derived from ``data[ordered_on]`` column the ends of bins.
            If snapshotting, then points of observation (defined by
            ``snap_by``) can then be positioned with respect to the bin ends.
            This data allows most notably sorting snapshots with respect to
            bins in case they start/end at the same row index in data (most
            notably possible in case of empty snapshots and/or empty bins).

          - ``bin_closed``, a str, either `'right'` or `'left'`, indicating if
            bins are left or right-closed.

    bin_on : Union[str, None]
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `Grouper`, its `key` parameter is used instead,
        and 'bin_on' is ignored.
    binning_buffer : Union[dict, None]
        User-chosen values from previous binning process, that can be required
        when restarting the binning process with new seed data.
    ordered_on : Union[str, None]
        Name of an existing ordered column in 'data'. When setting it, it is
        then forwarded to 'bin_by' Callable.
        This parameter is compulsory if 'snap_by' is set. Values derived from
        'snap_by' (either a Grouper or a Series) are compared to ``bin_ends``,
        themselves derived from ``data[ordered_on]``.
    snap_by : Union[Grouper, IntervalIndex, None]
        Values positioning the points of observation, either derived from a
        pandas Grouper, or contained in a pandas IntervalIndex.
        In case 'snap_by' is an IntervalIndex, it should contain one Interval
        per point of observation. Only the "ends", i.e. the right edges are
        retrieved to serve as locations for points of observation.
        Additionally, ``snap_by.closed`` has to be set, either to `left` or
        `right`. As a convention, at point of observation, if

              - `left`, then values at point of observation are excluded.
              - `right`, then values at point of observation are included.

    allow_bins_snaps_disalignment : bool, default False
        By default, check that ``bin_by.closed`` and ``snap_by.closed`` are set
        to the same value. If not, an error is raised.
        As a result of the logic when setting 'bins.closed' and 'snaps.closed'
        to different values, incomplete snapshots can be created. The relevance
        of such a use is not clear and for safety, this combination is not
        possible by default.
        To make it possible, set 'allow_bins_snaps_disalignment' `True`.

    Returns
    -------
    pDataFrame
        A pandas DataFrame with aggregation results. Its index is composed of
        the bin labels.

    Notes
    -----
    When using snapshots, values derived from ``snap_by`` Grouper (or right
    edges of ``snap_by`` IntervalIndex) are considered the "points of isolated
    observation". At such a point, an observation of the "on-going" bin is
    made. In case of snapshot(s) positioned exactly on segment(s) ends, at the
    same row index in data, if

      - the bins are left-closed, `[(`,

          - if points of observations are excluded, then snapshot(s) will come
            before said bin(s), that is to say, these snapshots will be equal
            to the first bin (subsequent bins are then "empty" ones).
          - if points of observations are included, then snapshot(s) will come
            after said bin(s), that is to say, these will be "empty" snapshots.

      - the bins are right-closed, `)]`, whatever if the points of observations
        are included or excluded, snapshot(s) will come before said bin(s),
        that is to say, these snapshots will be equal to the first bin
        (subsequent bins are then "empty" ones).

    """
    if ordered_on:
        if (
            data[ordered_on].dtype == DTYPE_DATETIME64
            and (data[ordered_on].diff() >= Timedelta(0)).all()
            or data[ordered_on].dtype != DTYPE_DATETIME64
            and (data[ordered_on].diff() >= 0).all()
        ):
            # 'ordered_on' is not an ordered column.
            raise ValueError(f"column '{ordered_on}' is not ordered.")
    if isinstance(next(iter(agg.keys())), str):
        # Reshape aggregation definition.
        agg = setup_cgb_agg(agg, data.dtypes.to_dict())
    # All 'bin_xxx' parameters are expected to be the size of the resulting
    # aggregated array from 'groupby' operation, i.e. including empty bins.
    # 'bin_labels', 'next_chunk_starts', and if defined, 'bin_ends'.
    if isinstance(bin_by, Grouper):
        # 'bin_by' is a Grouper.
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
        bins, next_chunk_starts, n_null_bins = bin_by_time(data.loc[:, bin_on], bin_by)
        if snap_by:
            bin_ends = bins.right
        bin_labels = bins.right if bin_by.label == "right" else bins.left
        bin_closed = bin_by.closed
    else:
        # 'bin_by' is a Callable.
        if bin_on is None:
            raise ValueError("not possible to set 'bin_on' to `None`.")
        on = (
            data.loc[:, [bin_on, ordered_on]]
            if ordered_on and ordered_on != bin_on
            else data.loc[:, bin_on]
        )
        # 'bin_by' binning, possibly jitted.
        bin_by_res = bin_by(on, binning_buffer)
        if snap_by:
            bin_labels, next_chunk_starts, n_null_bins, bin_ends, bin_closed = bin_by_res
        else:
            bin_labels, next_chunk_starts, n_null_bins = bin_by_res
    # Parameters related to bins.
    n_bins = len(bin_labels)
    null_bin_indices = zeros(n_null_bins, dtype=DTYPE_INT64)
    # Initiate dict of result columns.
    bin_res = {}
    if snap_by:
        if not ordered_on:
            raise ValueError("not possible to set 'ordered_on' to `None` in case of snapshotting.")
        if isinstance(snap_by, Grouper):
            if snap_by.key and snap_by.key != ordered_on:
                raise ValueError(
                    "not possible to set 'ordered_on' and 'snap_by.key' to different values."
                )
        snap_on = ordered_on
        if not allow_bins_snaps_disalignment:
            if bins.closed != snap_by.closed:
                raise ValueError(
                    "as a result of the logic when setting 'bin_by.closed' and 'snap_by.closed' to "
                    "different values, incomplete snapshots can be created. The relevance of "
                    "such a use is not clear and for safety, this combination is not possible "
                    "by default. To make it possible, set 'allow_bins_snaps_disalignment' `True`."
                )
        # Define points of observation
        if isinstance(snap_by, Grouper):
            # 'snap_by' is an Grouper.
            snaps, next_snap_starts, n_max_null_snaps = bin_by_time(data.loc[:, snap_on], snap_by)
        else:
            # 'snap_by' is an IntervalIndex.
            snaps = snap_by
            next_snap_starts = zeros(len(snaps), dtype=DTYPE_INT64)
            n_max_null_snaps = zeros(1, dtype=DTYPE_INT64)
            right_edges = snap_by.right
            is_datetime = right_edges.dtype == DTYPE_DATETIME64
            _next_chunk_starts(
                snap_on.to_numpy(copy=False).view(DTYPE_INT64)
                if is_datetime
                else snap_on.to_numpy(copy=False),
                right_edges.to_numpy(copy=False).view(DTYPE_INT64)
                if is_datetime
                else right_edges.to_numpy(copy=False),
                snaps.closed == "right",
                next_snap_starts,
                n_max_null_snaps,
            )
            n_max_null_snaps = n_max_null_snaps[0]
        # The 1st edge (out of the values) is removed.
        snap_closed = snap_by.closed
        if snap_closed != "left" and snap_closed != "right":
            raise ValueError("'snap_by.closed' has to be set either to 'left' or to 'right'.")
        if bin_closed != "left" and bin_closed != "right":
            raise ValueError(
                "'bin_by.closed' if a Grouper or 'bin_closed' if a Callable has to be set either to 'left' or to 'right'."
            )
        # 'snaps" become the labels to use for each snapshot.
        snap_labels = snaps.right
        # Parameters related to bins.
        n_snaps = len(snap_labels)
        # Initialize 'null_snap_indices' to -1, to identify easily those
        # which are not set, and remove them.
        null_snap_indices = -ones(n_max_null_snaps, dtype=DTYPE_INT64)
        # Initiate dict of result columns.
        snap_res = {}
        # Consolidate 'next_snap_starts' into 'next_chunk_starts'.
        if bin_closed == "left" and snap_closed == "right":
            # Bins are left-closed,and observation points are included.
            # If sharing the same "next_starts", snapshot come after the bin.
            next_chunk_starts, bin_indices = merge_sorted(
                labels=(next_chunk_starts, next_snap_starts),
                keys=(bin_ends, snap_labels),
                ii_for_first=True,
            )
        else:
            # If sharing the same "next_starts", snapshot come before the bin.
            next_chunk_starts, bin_indices = merge_sorted(
                labels=(next_snap_starts, next_chunk_starts),
                keys=(snap_labels, bin_ends),
                ii_for_first=False,
            )
    else:
        # Case 'no snapshotting'.
        bin_indices = NULL_INT64_1D_ARRAY
        null_snap_indices = NULL_INT64_1D_ARRAY
        snap_res_single_dtype = NULL_INT64_2D_ARRAY
    for dtype_, (
        cols_name_in_data,
        cols_name_in_res,
        cols_idx,
        n_cols,  # 1d
    ) in agg.items():
        data_single_dtype = (
            data.loc[:, cols_name_in_data].to_numpy(copy=False)
            if len(cols_name_in_data) > 1
            else data.loc[:, cols_name_in_data].to_numpy(copy=False).reshape(-1, 1)
        )
        n_cols_single_dtype = len(cols_name_in_res)
        bin_res_single_dtype = zeros((n_bins, n_cols_single_dtype), dtype=dtype_)
        bin_res.update(
            {name: bin_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)}
        )
        if snap_by:
            snap_res_single_dtype = zeros((n_snaps, n_cols_single_dtype), dtype=dtype_)
            snap_res.update(
                {name: snap_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)}
            )
        if dtype_ == DTYPE_DATETIME64:
            data_single_dtype = data_single_dtype.view(DTYPE_INT64)
            bin_res_single_dtype = bin_res_single_dtype.view(DTYPE_INT64)
            if snap_by:
                snap_res_single_dtype = snap_res_single_dtype.view(DTYPE_INT64)
        # 'data' is a numpy array, with columns in 'expected order',
        # as defined in 'cols_idx'.
        # 'cols_idx[ID_AGG_FUNC, :, 0]' contains indices for cols in data
        # 'cols_idx[ID_AGG_FUNC, :, 1]' contains indices for cols in res
        jcsagg(
            data_single_dtype,  # 2d
            n_cols,  # 1d
            cols_idx,  # 3d
            next_chunk_starts,  # 1d
            bin_indices,  # 1d
            bin_res_single_dtype,  # 2d
            snap_res_single_dtype,  # 2d
            null_bin_indices,  # 1d
            null_snap_indices,  # 1d
        )
    # Assemble 'bin_res' as a pandas DataFrame.
    bin_res = pDataFrame(bin_res, index=bin_labels, copy=False)
    bin_res.index.name = ordered_on if ordered_on else bin_on
    # Set null values.
    if n_null_bins != 0:
        null_bin_labels = bin_labels[null_bin_indices]
        for dtype_, (
            _,
            cols_name_in_res,
            _,
            _,
        ) in agg.items():
            bin_res.loc[null_bin_labels, cols_name_in_res] = NULL_DICT[dtype_]
    if snap_by:
        snap_res = pDataFrame(snap_res, index=snap_labels, copy=False)
        snap_res.index.name = ordered_on
        # Set null values.
        if n_max_null_snaps != 0:
            # Remove -1 indices.
            null_snap_labels = snap_labels[null_snap_indices[~nisin(null_snap_indices, -1)]]
            if DTYPE_INT64 in agg:
                # As of pandas 1.5.3, use "Int64" dtype to work with nullable 'int'.
                # (it is a pandas dtype, not a numpy one)
                snap_res[agg[DTYPE_INT64][1]] = snap_res[agg[DTYPE_INT64][1]].astype("Int64")
            for dtype_, (
                _,
                cols_name_in_res,
                _,
                _,
            ) in agg.items():
                snap_res.loc[null_snap_labels, cols_name_in_res] = NULL_DICT[dtype_]
        return bin_res, snap_res
    else:
        return bin_res
