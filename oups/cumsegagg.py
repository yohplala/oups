#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

from numpy import NaN as nNaN
from numpy import array
from numpy import dtype
from numpy import full
from numpy import isin as nisin
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Int64Dtype
from pandas import NaT as pNaT
from pandas import Series

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
            raise ValueError(f"column '{in_col}' does not exist in input data.")
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
    bin_on: Optional[str] = None,
    buffer: Optional[dict] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[Grouper, Series, DatetimeIndex]] = None,
    error_on_0: Optional[bool] = True,
) -> Union[pDataFrame, Tuple[pDataFrame, pDataFrame]]:
    """Cumulative segmented aggregations, with optional snapshotting.

    In this function, "snapshotting" is understood as the action of making
    isolated observations. When using snapshots, values derived from
    ``snap_by`` Grouper (or contained in ``snap_by`` Series) are considered the
    "points of isolated observation".
    At a given point, an observation of the "on-going" segment (aka bin) is
    made. Because segments are contiguous, any row of the dataset falls in a
    segment.

    Parameters
    ----------
    data: pDataFrame
        A pandas DataFrame containing the columns over which binning (relying
        on ``bin_on`` column), performing aggregations and optionally
        snapshotting (relying on column pointed by 'ordered_on' and optionally
        ``snap_by.key`` if is a Grouper).
        If using snapshots ('snap_by' parameter), then the column pointed by
        ``snap_by.key`` has to be ordered.
    agg : dict
        Definition of aggregation.
        If in the form ``Dict[str, Tuple[str, str]]`` (typically a form
        compatible with pandas aggregation), then it is transformed in the 2nd
        form ``Dict[dtype, Tuple[List[str], List[str], Tuple, int]]``.
          - in the form ``Dict[str, Tuple[str, str]]``
            - keys are ``str``, requested output column name
            - values are ``tuple`` with 1st component a ``str`` for the input
              column name, and 2nd component a ``str`` for aggregation function
              name.

          - the 2nd form is that returned by the function ``setup_cumsegagg``.

    bin_by : Union[Grouper, Callable]
        Callable or pandas Grouper to perform binning.
        If a Callable, is called with following parameters, please see
        description in 'segmentby' docstring.
    bin_on : Optional[str], default None
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `Grouper`, its `key` parameter is used instead,
        and 'bin_on' is ignored.
    buffer : Optional[dict], default None
        User-chosen values from previous binning process, that can be required
        when restarting the binning process with new seed data.
    ordered_on : Union[str, None]
        Name of an existing ordered column in 'data'. When setting it, it is
        then forwarded to 'bin_by' Callable.
        This parameter is compulsory if 'snap_by' is set. Values derived from
        'snap_by' (either a Grouper or a Series) are compared to ``bin_ends``,
        themselves derived from ``data[ordered_on]``.
    snap_by : Optional[Union[Grouper, Series, DatetimeIndex]], default None
        Values positioning the points of observation, either derived from a
        pandas Grouper, or contained in a pandas Series.
        In case 'snap_by' is a Series, values  serve as locations for points of
        observation.
        Additionally, ``closed`` value defined by 'bin_on' specifies if points
        of observations are included or excluded. As "should be logical", if
          - `left`, then values at points of observation are excluded.
          - `right`, then values at points of observation are included.

    error_on_0 : bool, default True
        By default, check that there is no `0` value (either int or float) in
        aggregation results (bins and snapshots). ``cumsegagg()`` is
        experimental and a `0` value is likely to hint a bug. If raised, the
        result should be double checked. Ultimately, please, report the use
        case that is raising this error, and what would be the expected
        behavior.

    Returns
    -------
    Union[pDataFrame, Tuple[pDataFrame, pDataFrame]]
        A pandas DataFrame with aggregation results. Its index is composed of
        the bin labels.
        If a tuple, then the first DataFrame is that for the bins, and the
        second that for the snapshots.

    Notes
    -----
    When using snapshots, values derived from ``snap_by`` are considered the
    "points of isolated observation". At such a point, an observation of the
    "on-going" bin is made. In case of snapshot(s) positioned exactly on
    segment(s) ends, at the same row index in data, snapshot will come "before"
    the bin.

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
    n_snaps = len(snap_labels) if snap_labels is not None else 0
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
        # /!\ WiP for restart, need to tell jcsagg about pinnu
        #     also rework in jcsagg: need to always print on exit in last bin the
        #     content of temporary buffer 'chunk_res'.
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
    #
    # Development notes regarding restart.
    # Managing a restart is how managing the first bin which was likely
    # incomplete at previous iteration.
    #
    # Segmentation ('segmentby()')
    # ----------------------------
    #
    # 1/ From this step, 'next_chunk_starts' should not end with an empty bin.
    # It will be check in 'segmentby()'. All empty bins at end of
    # 'next_chunk_starts' have to be trimmed.
    # Rationale is that when restarting after several empty bins, it *may* be
    # that some new data was actually in these bins, empty at previous
    # iteration.
    #
    # 2/ At this step, the 'restart_key' is a key that enables a restart from
    # last not empty bin of previous iteration, that was likely incomplete.
    # That it was actually complete or not does not raise trouble. It will be
    # check at next iteration in 'cumsegagg()'. If there is no new data in this
    # bin at new iteration, it will be discarded from new iteation.
    #
    # 3/ If using snapshots, in consolidated 'next_chunk_starts' &
    # 'bin_indices',
    #  - the two last indices in 'next_chunk_starts' have to cover the full
    #    data (this should be ensured directly by 'bin_by()' functions or
    #    an error should be raised).
    #  - the last index should be forced to be that of a bin (to avoid during
    #    the cumulative aggregation step that the last snapshot is empty)
    #
    # Cumulative segmented aggregation ('cumsegagg()')
    # ------------------------------------------------
    #
    # 1/ Before starting 'jcumsegagg()', the first bin is checked to be empty
    # or not (last bin from previous iteration was then complete).
    # - if it is empty in this new iteration, it is discarded, i.e. following
    #   parameters are forwarded to 'jcumsegagg()':
    #    - trimmed 'next_chunk_starts[1:]'
    #    - 'bin_res' reduced by one element (and bin_labels)
    #    - 'n_null_bin' reduced by one.
    # - to check it is empty, 2 checks are required:
    #     size is different than 0 AND label is different than label of last
    #     bin at prrevious iteration.
    #     ('by_x_rows' restarts directly on a new bin if the previous
    #      ended right on the last value of previous iter.)
    # - if it is not empty in this new iteration, previous aggregation results
    #   'chunk_res', need to be re-used and 'pinnu' has to be set to ``True``.
    #
    # 2/ In the specific case data is not traversed completely during
    # segmentation step, with 'by_scale()' (in case 'by' is a Series ending in
    # within 'on'.)
    # We don't know what label to provide the remaining of the data in 'on'.
    # In this case, no corresponding bin (or snapshot) is generated
    # in 'next_chunk_starts', but in 'cumsegagg()', 'chunk_res' does be updated
    # with this remaining data and the updated value is recoded (but this
    # value does not appear in 'bin_res' and 'snap_res')
    # In this case, a specific check in 'by_scale()' is required to ensure
    # correct restart: at next iteration, there should be no bin end before the
    # last timestamp at previous iteration in 'on'. (because this data will
    # then not be correctly accounted)
    # At a upper level, if this appears, one can re-use 'restart_key' to trim
    # correctly the data (with additional data from previous iteration), and
    # force a restart from scratch in 'cumsegagg()' (with 'pinnu' set False)
