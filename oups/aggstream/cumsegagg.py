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
from pandas import Int64Dtype
from pandas import NaT as pNaT
from pandas import Series
from pandas.core.resample import TimeGrouper

from oups.aggstream.jcumsegagg import AGG_FUNCS
from oups.aggstream.jcumsegagg import jcsagg
from oups.aggstream.segmentby import KEY_BIN_ON
from oups.aggstream.segmentby import KEY_LAST_BIN_LABEL
from oups.aggstream.segmentby import KEY_ORDERED_ON
from oups.aggstream.segmentby import KEY_SNAP_BY
from oups.aggstream.segmentby import segmentby
from oups.aggstream.segmentby import setup_segmentby


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_FLOAT64 = dtype("float64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
DTYPE_NULLABLE_INT64 = Int64Dtype()
NULL_INT64_1D_ARRAY = zeros(0, DTYPE_INT64)
NULL_INT64_2D_ARRAY = NULL_INT64_1D_ARRAY.reshape(0, 0)
# Null values.
NULL_DICT = {DTYPE_INT64: pNA, DTYPE_FLOAT64: nNaN, DTYPE_DATETIME64: pNaT}
# Key for buffer.
KEY_LAST_CHUNK_RES = "last_chunk_res"


def setup_cumsegagg(
    agg: Dict[str, Tuple[str, str]],
    data_dtype: Dict[str, dtype],
) -> Dict[dtype, Tuple[List[str], List[str], Tuple, int]]:
    """
    Construct chaingrouby aggregation configuration.

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
                                  in 'res', to which are recoreded aggregation
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
        try:
            if (agg_func := AGG_FUNCS[func]) in agg_funcs:
                func_idx = agg_funcs.index(agg_func)
            else:
                func_idx = len(agg_funcs)
                agg_funcs.append(AGG_FUNCS[func])
        except KeyError:
            raise ValueError(f"`{func}` aggregation function is unknown.")
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


def setup_chunk_res(agg: Dict[dtype, tuple]) -> pDataFrame:
    """
    Initialize one-row DataFrame for storing the first 'chunk_res'.
    """
    chunk_res = {}
    for dtype_, (
        _,
        cols_name_in_res,
        _,
        n_cols,
    ) in agg.items():
        chunk_res_single_dtype = zeros(n_cols, dtype=dtype_)
        chunk_res.update(
            {name: chunk_res_single_dtype[i : i + 1] for i, name in enumerate(cols_name_in_res)},
        )
    return pDataFrame(chunk_res, copy=False)


def cumsegagg(
    data: pDataFrame,
    agg: Union[Dict[str, Tuple[str, str]], Dict[dtype, Tuple[List[str], List[str], Tuple, int]]],
    bin_by: Union[TimeGrouper, Callable, dict],
    bin_on: Optional[str] = None,
    buffer: Optional[dict] = None,
    ordered_on: Optional[str] = None,
    snap_by: Optional[Union[TimeGrouper, Series, DatetimeIndex]] = None,
    error_on_0: Optional[bool] = True,
) -> Union[pDataFrame, Tuple[pDataFrame, pDataFrame]]:
    """
    Cumulative segmented aggregations, with optional snapshotting.

    In this function, "snapshotting" is understood as the action of making
    isolated observations. When using snapshots, values derived from
    ``snap_by`` TimeGrouper (or contained in ``snap_by`` Series) are considered
    the "points of isolated observation".
    At a given point, an observation of the "on-going" segment (aka bin) is
    made. Because segments are contiguous, any row of the dataset falls in a
    segment.

    Parameters
    ----------
    data: pDataFrame
        A pandas DataFrame containing the columns over which binning (relying
        on ``bin_on`` column), performing aggregations and optionally
        snapshotting (relying on column pointed by 'ordered_on' and optionally
        ``snap_by.key`` if is a TimeGrouper).
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

    bin_by : Union[TimeGrouper, Callable, dict]
        Callable or pandas TimeGrouper to perform binning.
        If a Callable, please see signature requirements in 'segmentby'
        docstring.
        If a dict, it contains the full setup for conducting the segmentation
        of 'data', as generated by 'setup_segmentby()'.
    bin_on : Optional[str], default None
        Name of the column in `data` over which performing the binning
        operation.
        If 'bin_by' is a pandas `TimeGrouper`, its `key` parameter is used instead,
        and 'bin_on' is ignored.
        If not provided, and 'ordered_on' parameter is, then 'ordered_on' value
        is also used to specify the column name onto which performing binning.
    buffer : Optional[dict], default None
        Buffer containing data for restarting the binning process with new seed
        data:
        - from previous segmentation step,
        - from previous aggregation step.
    ordered_on : Union[str, None]
        Name of an existing ordered column in 'data'. When setting it, it is
        then forwarded to 'bin_by' Callable.
        This parameter is compulsory if 'snap_by' is set. Values derived from
        'snap_by' (either a TimeGrouper or a Series) are compared to ``bin_ends``,
        themselves derived from ``data[ordered_on]``.
    snap_by : Optional[Union[TimeGrouper, Series, DatetimeIndex]], default None
        Values positioning the points of observation, either derived from a
        pandas TimeGrouper, or contained in a pandas Series.
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

    Notes on design for allowing 'restart'
    --------------------------------------
    Current implementation may present limitations from inadequate design
    choices, not challenged so far.
    To minimize memory footprint, segmentation step is expected to provide
    start indices of the next bin (as opposed to providing the status for each
    row of the input data, individually).
    Because historically, the aggregation function is expected to use all data
    so as to provide the actual status of the last, in-progress bin, its end
    is de-facto the end of the input data.
    Because of this, the internal flag 'preserve_res'  is always set to
    ``False`` when reaching the end of input data.
    This is a limitation. It should be ``False`` only to mark the actual end of
    the bins. As a result, this internal flag 'preserve_res' cannot be output
    for reuse in a next restart step.
    An option to circumvent this is to use snapshots instead of bins as media
    to output aggregation results for last, in-progress bin.
    This option has not been implemented.

    In current implementation, the limitation presented above is circumvented
    by assuming that last bin is never empty, that is to say, 'chunk_res'
    parameter which contains aggregation results for the last, in-progress bin,
    always has relevant results to preserve. This is true, as long as the last,
    in-progress bin is not empty.
    Would this last bin be empty, then 'chunk_res' would still contain
    aggregation results for the last not empty bin it was used. In this case,
    we would need to make sure if the last row in input data matches the end of
    a bin or not, to not preserve 'chunk_res' or preserve it.
    Now, if we assume the last, in-progress bin is not empty, we can wait for
    the restart to check if the bin has ended before the start of input data,
    and then close this bin which was the last, in-progress bin at previous
    iteration.
    To bring more freedom to this implementation, a 'preserve_res' flag is
    expected from the segmentation phase. This flag is set ``False`` to allow
    restarting right on a new, next bin, if at the previous iteration, the last
    bin was complete.
    In current implementation, the limitation is thus that from the
    segmentation, the last bin cannot be empty. All empty trailing bins have to
    be trimmed, otherwise an exception is raised.

    The following thoughts have been investigated in current implementation.
      - **segmentation step ('segmentby()')**
        -1 From this step, 'next_chunk_starts' should not end with an empty
           bin, as mentioned above. A complementary thought is that when
           restarting after several empty bins, it *may* be that some new data
           was actually in these bins, empty at previous iteration.
           A check is then managed in 'segmentby()'. All empty bins at end of
           'next_chunk_starts' have to be trimmed or an exception will be
           raised.
        -2 If restarting, bins produced by the user-defined 'bin_by()' have to
           cover the full size of data, meaning last item in
           'next_chunk_starts' is equal to length of data.
           As mentioned above, this rationale is from history.
           Additionally, it *may* be that if no bin goes till the end of data,
           then we are not sure the next bin (at next iteration) will not lie
           within these last values in data at current iteration.
           A check is then performed and an exception is raised if this
           situation occurs.
           This requirement is not applied to 'snap_by' (in case using a
           Series). Because it is applied to 'bin_by', then 'chunk_res' will
           contain aggregation results over the last values in data, it is not
           lost.
           In the existing 'snap_by' (either by TimeGrouper or by Series),
             - either if a TimeGrouper, then last snapshot ends after end of data
             - or if a Series, at restart, if 2nd snapshot ends before last
               value in data at previous iteration, then an exception is
               raised.

        -3 At next iteration, the first bin has to be the continuation of the
           last one from previous iteration. A check is made using bin label.
           This is the case even if the bin is empty. Thus, if it is preceded /
           followed by empty snapshots, content of these snapshots will be set
           appropriately. For empty snapshots that precede this bin end, past
           results are forwarded. For empty snapshots that follow this bin end,
           this results in empty snapshots.

      - **cumulative segmented aggregation ('cumsegagg()')**
        -1 'preserve_res' parameter is used to indicate if aggregation
           calculations start from scratch (first iteration) or reuse past
           aggregation results (following iterations).
           Aggregation results from last, in-progress bin can then be
           forwarded.

    """
    # TODO: create a test case with restart, that has no snapshot in 1st
    # iteration (with 'by_scale' using a Series). Target is to check that even
    # without snapshot in 1st iteration, an empty 'snap_res' gets returned
    # nonetheless and that concatenation can be managed with subsequent
    # 'snap_res' from next iterations.
    # TODO: if requesting snapshots, bin aggregation results are not necessary.
    # Consider just outputting label of bins in snapshot results, without
    # agrgegation results for bins? (memory savings).
    len_data = len(data)
    if not len_data:
        # 'data' is empty. Simply return.
        return
    if not isinstance(next(iter(agg.values())), list):
        # Reshape aggregation definition.
        agg = setup_cumsegagg(agg, data.dtypes.to_dict())
    if buffer is None:
        # Single run agg.
        preserve_res = False
    else:
        # Agg iteration with possible restart.
        # Detection of 1st iteration is managed below with test if a new bin
        # is started.
        preserve_res = True
        prev_last_bin_label = buffer[KEY_LAST_BIN_LABEL] if KEY_LAST_BIN_LABEL in buffer else None
    if not isinstance(bin_by, dict):
        bin_by = setup_segmentby(bin_by, bin_on, ordered_on, snap_by)
    # Following 'setup_segmentby', parameters 'ordered_on', 'bin_on'  have to
    # be retrieved from it.
    ordered_on = bin_by[KEY_ORDERED_ON]
    # 'bin_by' as a dict may contain 'snap_by' if it is a TimeGrouper.
    if bin_by[KEY_SNAP_BY] is not None:
        # 'bin_by[KEY_SNAP_BY]' is not none if 'snap_by' is a TimeGrouper.
        # Otherwise, it can be a DatetimeIndex or a Series.
        snap_by = bin_by[KEY_SNAP_BY]
    # In case of restart, 'n_max_null_bins' is a max because 1st null bin may
    # well be continuation of last in-progress bin, without result in current
    # iteration, but with results from previous iteration.
    (
        next_chunk_starts,
        bin_indices,
        bin_labels,
        n_max_null_bins,
        snap_labels,
        n_max_null_snaps,
    ) = segmentby(
        data=data,
        bin_by=bin_by,
        snap_by=snap_by,
        buffer=buffer,
    )
    if preserve_res and prev_last_bin_label != bin_labels.iloc[0]:
        # A new bin has been started. Do not preserve past results.
        # This behavior is only possible in case no snapshot is used.
        preserve_res = False
    # Initiate dict of result columns.
    # Setup 'chunk_res'.
    chunk_res_prev = (
        buffer[KEY_LAST_CHUNK_RES]
        if isinstance(buffer, dict) and KEY_LAST_CHUNK_RES in buffer
        else setup_chunk_res(agg)
    )
    chunk_res = {}
    # Setup 'bin_res'.
    n_bins = len(bin_labels)
    null_bin_indices = full(n_max_null_bins, -1, dtype=DTYPE_INT64)
    bin_res = {}
    # Setup 'snap_res', & preserve_res
    snap_res = {}
    if snap_by is None:
        snap_res_single_dtype = NULL_INT64_2D_ARRAY
        null_snap_indices = NULL_INT64_1D_ARRAY
    else:
        # Initialize 'null_snap_indices' to -1, to identify easily those which
        # are not set. they will be removed in a post-processing step.
        n_snaps = len(snap_labels)
        null_snap_indices = full(n_max_null_snaps, -1, dtype=DTYPE_INT64)
    # Loop.
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
        # Setup 'chunk_res_single_dtype'.
        chunk_res_single_dtype = (
            chunk_res_prev.loc[:, cols_name_in_res].to_numpy(copy=False).reshape(n_cols)
        )
        chunk_res.update(
            {name: chunk_res_single_dtype[i : i + 1] for i, name in enumerate(cols_name_in_res)},
        )
        # Setup 'bin_res_single_dtype'.
        bin_res_single_dtype = zeros((n_bins, n_cols), dtype=dtype_)
        bin_res.update(
            {name: bin_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)},
        )
        # Setup 'snap_res_single_dtype'.
        if snap_by is not None:
            snap_res_single_dtype = zeros((n_snaps, n_cols), dtype=dtype_)
            snap_res.update(
                {name: snap_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_res)},
            )
        if dtype_ == DTYPE_DATETIME64:
            data_single_dtype = data_single_dtype.view(DTYPE_INT64)
            bin_res_single_dtype = bin_res_single_dtype.view(DTYPE_INT64)
            chunk_res_single_dtype = chunk_res_single_dtype.view(DTYPE_INT64)
            if snap_by is not None:
                snap_res_single_dtype = snap_res_single_dtype.view(DTYPE_INT64)
        # 'data' is a numpy array, with columns in 'expected order',
        # as defined in 'cols_data' & 'cols_res' embedded in 'aggs'.
        # TODO: if extending 'jcsagg()' to process last chunk in data (even if
        # not a bin or a snap, so as to make possible that bins really only end
        # on end of bins, and that end of 'data' is not systematically a bin
        # end as well), then output from 'jcsagg()' 'preserve_res' parameter.
        # When inputting it for the next iteration, 'preserve_res' parameter
        # is then ``not first_bin_is_new and preserve_res``.
        # With this feature, empty trailing bins are then possible to manage.
        jcsagg(
            data_single_dtype,  # 2d
            aggs,
            next_chunk_starts,  # 1d
            bin_indices,  # 1d
            preserve_res,
            chunk_res_single_dtype,
            bin_res_single_dtype,  # 2d
            snap_res_single_dtype,  # 2d
            null_bin_indices,  # 1d
            null_snap_indices,  # 1d
        )
    # Record last aggregation results for a restart.
    if isinstance(buffer, dict):
        buffer[KEY_LAST_CHUNK_RES] = pDataFrame(chunk_res, copy=False)
    # Assemble 'bin_res' as a pandas DataFrame.
    bin_res = pDataFrame(bin_res, index=bin_labels, copy=False)
    bin_res.index.name = ordered_on if ordered_on else bin_by[KEY_BIN_ON]
    # Set null values.
    if n_max_null_bins != 0:
        null_bin_labels = bin_labels.iloc[null_bin_indices[~nisin(null_bin_indices, -1)]]
        if not null_bin_labels.empty:
            if DTYPE_INT64 in agg:
                # As of pandas 1.5.3, use "Int64" dtype to work with nullable 'int'.
                # (it is a pandas dtype, not a numpy one)
                bin_res[agg[DTYPE_INT64][1]] = bin_res[agg[DTYPE_INT64][1]].astype(
                    DTYPE_NULLABLE_INT64,
                )
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
            # TODO: is not necessary to re-create an array without the -1.
            # Only indices above 0 should be used.
            # Alternatively, output number of empty snaps from 'jcumsegagg()'?
            null_snap_labels = snap_labels[null_snap_indices[~nisin(null_snap_indices, -1)]]
            if not null_snap_labels.empty:
                if DTYPE_INT64 in agg:
                    # As of pandas 1.5.3, use "Int64" dtype to work with nullable 'int'.
                    # (it is a pandas dtype, not a numpy one)
                    snap_res[agg[DTYPE_INT64][1]] = snap_res[agg[DTYPE_INT64][1]].astype(
                        DTYPE_NULLABLE_INT64,
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
                "at least one null value exists in 'snap_res' which is likely to hint a bug.",
            )
        if bin_res.eq(0).any().any():
            raise ValueError(
                "at least one null value exists in 'bin_res' which is likely to hint a bug.",
            )
    if snap_by is not None:
        return bin_res, snap_res
    else:
        return bin_res
