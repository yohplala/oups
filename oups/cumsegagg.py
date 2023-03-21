#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from typing import Callable, Dict, List, Tuple, Union

from numpy import NaN as nNaN
from numpy import array
from numpy import dtype
from numpy import full
from numpy import isin as nisin
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Int64Dtype
from pandas import IntervalIndex
from pandas import NaT as pNaT

from oups.jcumsegagg import AGG_FUNCS
from oups.jcumsegagg import jcsagg
from oups.segmentby import segmentby


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_FLOAT64 = dtype("float64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
DTYPE_NULLABLE_INT64 = Int64Dtype()
NULL_INT64_1D_ARRAY = zeros(0, DTYPE_INT64)
NULL_INT64_2D_ARRAY = NULL_INT64_1D_ARRAY.reshape(0, 0)
# Null values.
NULL_DICT = {DTYPE_INT64: pNA, DTYPE_FLOAT64: nNaN, DTYPE_DATETIME64: pNaT}


def setup_cumsegagg(
    agg: Dict[str, Tuple[str, str]], data_dtype: Dict[str, dtype]
) -> Dict[dtype, Tuple[List[str], List[str], Tuple, int]]:
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
    Dict[dtype,
         Tuple[List[str],
               List[str],
               Tuple[Tuple[Callable, ndarray[int64], ndarray[int64]]],
               int64
               ]
         ]
         Dict 'cgb_agg_cfg' in the form
         ``{dtype: List[str], 'cols_name_in_data'
                              column name in input data, with this dtype,
                   List[str], 'cols_name_in_res'
                              expected column names in aggregation result,
                   Tuple[Tuple[Callable, ndarray[int64], ndarray[int64]]],
                              'aggs'
                              Tuple of Tuple. One inner Tuple per aggregation
                              function. Each one contain 3 items,
                                - a Callable, the aggregation function
                                - a 1st 1d numpy array with indices of columns
                                  in 'data', to which has to be applied the
                                  aggregation function.
                                - a 2nd 1d numpy array with indices of columns
                                  in 'res', to which are recoreded aggrgation
                                  results
                   int64, 'n_cols'
                              Total number of columns in 'res' (summing for all
                              aggregation function).
           }``
    """
    cgb_agg_cfg = {}
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
                [],  # 'cols_data' (temporary)
                [],  # 'cols_res' (temporary)
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
        # Set list of agg functions (temporary buffer).
        agg_funcs = tup[2]
        if (agg_func := AGG_FUNCS[func]) in agg_funcs:
            func_idx = agg_funcs.index(agg_func)
        else:
            func_idx = len(agg_funcs)
            agg_funcs.append(AGG_FUNCS[func])
        # 'cols_idx'
        cols_data = tup[3]
        cols_res = tup[4]
        if len(cols_data) <= func_idx:
            # Create list for this aggregation function.
            cols_data.append([in_col_idx])
            cols_res.append([out_col_idx])
        else:
            # Add this column index for this aggregation function.
            cols_data[func_idx].append(in_col_idx)
            cols_res[func_idx].append(out_col_idx)
    # Step 2.
    for conf in cgb_agg_cfg.values():
        # Remove 'agg_funcs' & 'cols_idx'.
        agg_funcs = conf.pop(2)
        cols_data = conf.pop(2)
        cols_res = conf.pop(2)
        n_cols = sum(map(len, cols_res))
        # Add back 'aggs', as tuple of tuple.
        conf.append(tuple(zip(agg_funcs, map(array, cols_data), map(array, cols_res))))
        # 'n_cols'.
        conf.append(n_cols)
    return cgb_agg_cfg


def cumsegagg(
    data: pDataFrame,
    agg: Union[Dict[str, Tuple[str, str]], Dict[dtype, Tuple[List[str], List[str], Tuple, int]]],
    bin_by: Union[Grouper, Callable],
    bin_on: Union[str, None] = None,
    buffer: dict = None,
    ordered_on: Union[str, None] = None,
    snap_by: Union[Grouper, IntervalIndex, None] = None,
    allow_bins_snaps_disalignment: bool = False,
    error_on_0: bool = True,
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
        A pandas DataFrame containing the columns over which binning (relying
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
            - either ``ordered_on`` is ``None``. ``on`` is then a 1-column
              pandas DataFrame made from ``data[bin_on]`` column.
            - or ``ordered_on`` is provided and is different than ``bin_on``.
              Then ``on`` is a 2-column pandas DataFrame made of
              ``data[[bin_on, ordered_on]]``. Values from ``data[ordered_on]``
              can be used advantageously as bin labels.
              Also, values from ``data[ordered_on]`` have to be used when
              'snap_by' is set to define bin ends. See below.

          - ``binning_buffer``, a dict that has to be modified in-place by
             'bin_by' to keep internal parameters and to allow new faultless
             calls to 'bin_by'.

        It has then to return a Tuple made of 3 or 5 items. There are 3 items
        if no snapshotting.
          - ``next_chunk_starts``, a one-dimensional array of `int`, specifying
            the row index at which the next bin starts (included) as found in
            ``bin_on``.
            If the same index appears several time, it means that corresponding
            bins are empty, except the first one. In this case, corresponding
            rows in aggregation result will be filled with null values.
          - ``bin_labels``, a pandas Series or one-dimensional array, which
            values are expected to be all bin labels, incl. for empty
            bins, as they will appear in aggregation results. Labels can be of
            any type.
          - ``n_null_bins``, an `int` indicating the number of empty bins.

        In case of snapshotting (``snap_by`` is different than ``None``), the 2
        additional items are:
          - ``bin_closed``, a str, either `'right'` or `'left'`, indicating if
            bins are left or right-closed (i.e. if ``bin_ends`` in included or
            excluded).
          - ``bin_ends``, a one dimensional array, specifying the ends of bins
            with values derived from ``data[ordered_on]`` column.
            If snapshotting, then points of observation (defined by
            ``snap_by``) can then be positioned with respect to the bin ends.
            This data allows most notably sorting snapshots with respect to
            bins in case they start/end at the same row index in data (most
            notably possible in case of empty snapshots and/or empty bins).

    bin_on : Union[str, None]
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `Grouper`, its `key` parameter is used instead,
        and 'bin_on' is ignored.
    buffer : Union[dict, None]
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
        By default, check that ``bin_by.closed`` and ``snap_by.closed`` are not
        set simulatenously to 'right', resp. 'left'.
        If not, an error is raised.
        As a result of the logic when setting 'bins.closed' and 'snaps.closed'
        to 'right', resp. 'left', incomplete snapshots can be created. The
        relevance of such a use is not clear and for safety, this combination
        is not possible by default.
        To make it possible, set 'allow_bins_snaps_disalignment' `True`.
    error_on_0 : bool, default True
        By default, check that there is no `0` values (either int or float) in
        aggregation results (bins and snapshots). ``cumsegagg()`` is
        experimental and a `0` value is likely to hint a bug. If raised, the
        result should be double checked. Ultimately, please, report the use
        case that is raising this error, and what would be the expected
        behavior.

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

    When using 'cumsegagg' through 'chainagg' function (i.e. for chained calls
    to 'cumsegagg') and if setting `bin_by` as a Callable, the developer should
    take care that `bin_by` keeps in the 'buffer' parameter all the data needed
    to:
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

    """
    len_data = len(data)
    if not len_data:
        # 'data' is empty. Simply return.
        return
    if isinstance(next(iter(agg.keys())), str):
        # Reshape aggregation definition.
        agg = setup_cumsegagg(agg, data.dtypes.to_dict())
    (
        next_chunk_starts,
        bin_indices,
        bin_labels,
        n_null_bins,
        snap_labels,
        n_max_null_snaps,
    ) = segmentby(
        data=data,
        bin_by=bin_by,
        bin_on=bin_on,
        ordered_on=ordered_on,
        snap_by=snap_by,
        buffer=buffer,
    )
    # Initiate dict of result columns.
    n_bins = len(bin_labels)
    n_snaps = len(snap_labels)
    null_bin_indices = zeros(n_null_bins, dtype=DTYPE_INT64)
    bin_res = {}
    snap_res = {}
    if snap_by is None:
        snap_res_single_dtype = NULL_INT64_2D_ARRAY
    # Initialize 'null_snap_indices' to -1, to identify easily those which
    # are not set. they will be removed in a post-processing step.
    null_snap_indices = full(n_max_null_snaps, -1, dtype=DTYPE_INT64)
    for dtype_, (
        cols_name_in_data,
        cols_name_in_res,
        aggs,
        n_cols,
    ) in agg.items():
        data_single_dtype = (
            data.loc[:, cols_name_in_data].to_numpy(copy=False)
            if len(cols_name_in_data) > 1
            else data.loc[:, cols_name_in_data].to_numpy(copy=False).reshape(-1, 1)
        )
        bin_res_single_dtype = zeros((n_bins, n_cols), dtype=dtype_)
        bin_res.update(
            {name: bin_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)}
        )
        if snap_by is not None:
            snap_res_single_dtype = zeros((n_snaps, n_cols), dtype=dtype_)
            snap_res.update(
                {name: snap_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)}
            )
        if dtype_ == DTYPE_DATETIME64:
            data_single_dtype = data_single_dtype.view(DTYPE_INT64)
            bin_res_single_dtype = bin_res_single_dtype.view(DTYPE_INT64)
            if snap_by is not None:
                snap_res_single_dtype = snap_res_single_dtype.view(DTYPE_INT64)
        # 'data' is a numpy array, with columns in 'expected order',
        # as defined in 'cols_data' & 'cols_res' embedded in 'aggs'.
        jcsagg(
            data_single_dtype,  # 2d
            n_cols,
            aggs,
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
    if snap_by is not None:
        snap_res = pDataFrame(snap_res, index=snap_labels, copy=False)
        snap_res.index.name = ordered_on
        # Set null values.
        if n_max_null_snaps != 0:
            # Remove -1 indices.
            null_snap_labels = snap_labels[null_snap_indices[~nisin(null_snap_indices, -1)]]
            if DTYPE_INT64 in agg:
                # As of pandas 1.5.3, use "Int64" dtype to work with nullable 'int'.
                # (it is a pandas dtype, not a numpy one)
                snap_res[agg[DTYPE_INT64][1]] = snap_res[agg[DTYPE_INT64][1]].astype(
                    DTYPE_NULLABLE_INT64
                )
            for dtype_, (
                _,
                cols_name_in_res,
                _,
                _,
            ) in agg.items():
                snap_res.loc[null_snap_labels, cols_name_in_res] = NULL_DICT[dtype_]
    if error_on_0:
        if snap_by is not None and snap_res.eq(0).any().any():
            raise ValueError(
                "at least one null value exists in 'snap_res' which is likely to hint a bug."
            )
        if bin_res.eq(0).any().any():
            raise ValueError(
                "at least one null value exists in 'bin_res' which is likely to hint a bug."
            )
    if snap_by is not None:
        return bin_res, snap_res
    else:
        return bin_res
