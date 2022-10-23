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
from numpy import array
from numpy import count_nonzero
from numpy import dtype
from numpy import max as nmax
from numpy import min as nmin
from numpy import ndarray
from numpy import ndenumerate
from numpy import sum as nsum
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import NaT as pNaT
from pandas import Series
from pandas import date_range
from pandas.core.resample import _get_timestamp_range_edges as gtre

from oups.chainagg import FIRST
from oups.chainagg import LAST
from oups.chainagg import MAX
from oups.chainagg import MIN
from oups.chainagg import SUM


# Some constants.
DTYPE_INT64 = dtype("int64")
DTYPE_FLOAT64 = dtype("float64")
DTYPE_DATETIME64 = dtype("datetime64[ns]")
ZEROS_AR_FLOAT64 = zeros(0, dtype=DTYPE_FLOAT64)
ZEROS_AR_INT64 = zeros(0, dtype=DTYPE_INT64)
# Aggregation functions
ID_FIRST = 0
ID_LAST = 1
ID_MIN = 2
ID_MAX = 3
ID_SUM = 4
AGG_FUNC_IDS = {FIRST: ID_FIRST, LAST: ID_LAST, MIN: ID_MIN, MAX: ID_MAX, SUM: ID_SUM}


def setup_cgb_agg(
    agg: Dict[str, Tuple[str, str]], data_dtype: Dict[str, dtype]
) -> Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]:
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
         Dict in the form
         ``{dtype: ndarray[int64], 'agg_func_idx'
                                   1d-array, aggregation function indices,
                   ndarray[int64], 'n_cols'
                                   1d-array, number of input columns in data,
                                   to which apply this aggregation function,
                   List[str], 'cols_name_in_data'
                              column name in input data, with this dtype,
                   ndarray[int], 'cols_idx_in_data'
                                 2d-array, column indices in input data,
                                 per aggregation function,
                   List[str], 'cols_name_in_agg_res'
                              expected column names in aggregation result,
                   ndarray[int], 'cols_idx_in_agg_res'
                                 2d-array, column indices in aggregation
                                 results, per aggregation function.
           }``
    """
    cgb_agg_cfg = {}
    for out_col, (in_col, func) in agg.items():
        dtype_ = data_dtype[in_col]
        try:
            tup = cgb_agg_cfg[dtype_]
        except KeyError:
            cgb_agg_cfg[dtype_] = [[], [], [], [], [], []]
            tup = cgb_agg_cfg[dtype_]
        # function id / 1d list.
        agg_func_idx = tup[0]
        if (func_id := AGG_FUNC_IDS[func]) in agg_func_idx:
            func_idx = agg_func_idx.index(func_id)
        else:
            func_idx = len(agg_func_idx)
            agg_func_idx.append(AGG_FUNC_IDS[func])
        # 'in_col' / name / 1d list.
        cols_name_in_data = tup[2]
        if in_col in cols_name_in_data:
            in_col_idx = cols_name_in_data.index(in_col)
        else:
            in_col_idx = len(cols_name_in_data)
            cols_name_in_data.append(in_col)
        # 'in_col' / idx / 2d-array
        cols_idx_in_data = tup[3]
        if len(cols_idx_in_data) <= func_idx:
            # Create list for this aggregation function.
            cols_idx_in_data.append([in_col_idx])
        else:
            # Add this column index for this aggregation function.
            cols_idx_in_data[func_idx].append(in_col_idx)
        # 'out_col' / name / 1d list.
        cols_name_in_agg_res = tup[4]
        out_col_idx = len(cols_name_in_agg_res)
        cols_name_in_agg_res.append(out_col)
        # 'out_col' / idx / 2d-array.
        cols_idx_in_agg_res = tup[5]
        if len(cols_idx_in_agg_res) <= func_idx:
            # Create list for this aggregation function.
            cols_idx_in_agg_res.append([out_col_idx])
        else:
            # Add this column index for this aggregation function.
            cols_idx_in_agg_res[func_idx].append(out_col_idx)
    for conf in cgb_agg_cfg.values():
        n_func = len(conf[0])
        conf[1] = zeros(n_func, dtype=DTYPE_INT64)
        n_cols = conf[1]
        cols_idx_in_data = conf[3]
        for func_idx in range(n_func):
            # Retrieve number of columns in input data for this aggregation
            # function.
            n_cols[func_idx] = len(cols_idx_in_data[func_idx])
        # Transform 'agg_func_id' list into 1d array.
        conf[0] = array(conf[0], dtype=DTYPE_INT64)
        # Transform list of list into 2d array.
        max_cols = nmax(n_cols)
        for idx in (3, 5):
            # 3: 'in_col' idx
            # 5: 'out_col' idx
            ar = zeros((n_func, max_cols), dtype=DTYPE_INT64)
            for func_idx in range(n_func):
                ar[func_idx, : n_cols[func_idx]] = conf[idx][func_idx]
            conf[idx] = ar
    return cgb_agg_cfg


@guvectorize(
    [(int64[:], int64[:], boolean, int64[:]), (float64[:], float64[:], boolean, int64[:])],
    "(l),(m),(),(n)",
    nopython=True,
)
def _histo_on_sorted(data: ndarray, bins: ndarray, right: bool, histo: ndarray):
    """Histogram on sorted data.

    Parameters
    ----------
    data: ndarray
        One-dimensional array from which deriving the histogram, assumed
        sorted (monotonic increasing data).
    bins: ndarray
        One-dimensional array of bin edges, sorted.
    right : bool
        If `True`, histogram is built considering right-closed bins.
        If `False`, histogram is built considering left-closed bins.

    Returns
    -------
    histo: ndarray
        One-dimensional array, histogram of 'data' as per 'bins'.
        Size of 'hist' is ``len(bins)-1``.
    """
    bin_ = bins[0]
    data_len = len(data)
    if right:
        # Right-closed bins.
        for (_d_idx,), val in ndenumerate(data):
            if val > bin_:
                break
    else:
        # Left-closed bins.
        for (_d_idx,), val in ndenumerate(data):
            if val >= bin_:
                break
    if _d_idx + 1 == data_len:
        # No data in bins.
        return
    for (b_idx_loc,), bin_ in ndenumerate(bins[1:]):
        if right:
            # Right-closed bins.
            for (_d_idx_loc,), val in ndenumerate(data[_d_idx:]):
                if val > bin_:
                    break
        else:
            # Left-closed bins.
            for (_d_idx_loc,), val in ndenumerate(data[_d_idx:]):
                if val >= bin_:
                    break
        _d_idx += _d_idx_loc
        histo[b_idx_loc] = _d_idx_loc
        if _d_idx >= data_len:
            # Array 'data' terminated.
            return


def by_time(bin_on: Series, by: Grouper) -> Tuple[ndarray, ndarray, int]:
    """Bin as per pandas Grouper of an ordered date time index.

    Parameters
    ----------
    bin_on : Series
        Ordered date time index over performing the binning as defined per
        'by'.
    by : Grouper
        Setup to define time binning as a pandas Grouper.
    """
    start, end = gtre(
        first=bin_on.iloc[0],
        last=bin_on.iloc[-1],
        freq=by.freq,
        closed=by.closed,
        origin=by.origin,
        offset=by.offset,
    )
    bins = date_range(start, end, freq=by.freq)
    group_keys = bins[1:] if by.label == "right" else bins[:-1]
    group_sizes = zeros(len(group_keys))
    _histo_on_sorted(bin_on, bins, by.closed == "right", group_sizes)
    # Count zeros.
    n_nan_groups = count_nonzero(group_sizes == 0)
    return group_keys, group_sizes, n_nan_groups


def _jitted_agg_func_router(
    data: ndarray,  # 2d
    row_start: int,
    row_end: int,
    agg_func: ndarray,  # 1d
    n_cols: ndarray,  # 1d
    cols_in_data: ndarray,  # 2d
    cols_in_agg_res: ndarray,  # 2d
    agg_res_idx: int,
    agg_res: ndarray,  # 2d
):
    """Aggregate data.

    Parameters
    ----------
    data : ndarray
        Two dimensional array, containing the data to aggregate (agrgegation
        per column).
    row_start : int
        1st row index of bin in data.
    row_end : int
        Last row index of bin in data.
    agg_func : ndarray[int]
        One dimensional array of ``int``, containing the indices of the
        aggregation functions to apply. Say length 'f'.
    n_cols : ndarray[int]
        One dimensional array of ``int``, with size 'f', one row per
        aggregation function. Each row specifies for related aggregation
        function the number of columns in data to which applying the related
        aggregation function, hence the same number of columns resulting in
        'agg_res' for related aggregation function.
    cols_in_data : ndarray
        One dimensional array of ``int``, with size 'f', one row per
        aggregation function.
        If number of columns specified in 'n_cols' is ``n``, then the slice
        ``[:n]`` in the row provides the column indices in 'data'
        to which applying related aggregation function.
    cols_in_agg_res : ndarray
        One dimensional array of ``int``, with size 'f', one row per
        aggregation function.
        If number of columns specified in 'n_cols' is ``n``, then the slice
        ``[:n]`` in the row provides the column indices in 'agg_res'
        into which recording aggregation results for related aggregation
        function.
    agg_res_idx : ndarray
        Index of row in 'agg_res' into which recording aggregation result.

    Returns
    -------
    agg_res : ndarray
        Two dimensional array, to contain the aggregation results.
    """
    for idx_func, func in ndenumerate(agg_func):
        n_cols_ = n_cols[idx_func]
        data_chunk = data[row_start:row_end, cols_in_data[idx_func, :n_cols_]]
        if func == ID_FIRST:
            agg_res[agg_res_idx, cols_in_agg_res[idx_func, :n_cols_]] = data_chunk[0]
        elif func == ID_LAST:
            agg_res[agg_res_idx, cols_in_agg_res[idx_func, :n_cols_]] = data_chunk[-1]
        elif func == ID_MIN:
            agg_res[agg_res_idx, cols_in_agg_res[idx_func, :n_cols_]] = nmin(data_chunk, axis=0)
        elif func == ID_MAX:
            agg_res[agg_res_idx, cols_in_agg_res[idx_func, :n_cols_]] = nmax(data_chunk, axis=0)
        elif func == ID_SUM:
            agg_res[agg_res_idx, cols_in_agg_res[idx_func, :n_cols_]] = nsum(data_chunk, axis=0)


def _jitted_cgb(
    group_sizes: ndarray,
    data_float: ndarray,  # 2d
    agg_func_float: ndarray,  # 1d
    n_cols_float: ndarray,  # 1d
    cols_in_data_float: ndarray,  # 2d
    cols_in_agg_res_float: ndarray,  # 2d
    agg_res_float: ndarray,  # 2d
    data_int: ndarray,  # 2d
    agg_func_int: ndarray,  # 1d
    n_cols_int: ndarray,  # 1d
    cols_in_data_int: ndarray,  # 2d
    cols_in_agg_res_int: ndarray,  # 2d
    agg_res_int: ndarray,  # 2d
    data_dte: ndarray,  # 2d
    agg_func_dte: ndarray,  # 1d
    n_cols_dte: ndarray,  # 1d
    cols_in_data_dte: ndarray,  # 2d
    cols_in_agg_res_dte: ndarray,  # 2d
    agg_res_dte: ndarray,  # 2d
    nan_group_indices: ndarray,  # 1d
):
    """Group assuming contiguity.

    Parameters
    ----------
    group_sizes : ndarray
        Array of int, indicating the size of the groups. May contain ``0`` if
        a group key without any value is in resulting aggregation array.
    data_float : ndarray
        Array of ``float`` over which performing aggregation functions.
    agg_float_func : ndarray
        One dimensional array of ``int``, specifying the aggregation function
        ids.
    n_cols_float : ndarray
        One dimensional array of ``int``, specifying per aggregation function
        the number of columns to which applying related aggregation function
        (and consequently the number of columns in 'agg_res' to which recording
        the aggregation results).
    cols_in_data_float : ndarray
        Two dimensional array of ``int``, one row per aggregation function.
        Per row, column indices in 'data_float' to which apply corresponding
        aggregation function.
        Any value in column past the number of relevant columns is not used.
    cols_in_agg_res_float :  ndarray
        Two dimensional array of ``int``, one row per aggregation function.
        Per row, column indices in 'agg_res_float' into which storing the
        aggregation results.
        Any value in column past the number of relevant columns is not used.

    Returns
    -------
    agg_res_float : ndarray
        Results from aggregation, ``float`` dtype
    nan_group_indices : ndarray
        One dimensional array containing row indices in 'agg_res' that
        correspond to "empty" groups, i.e. for which group size has been set to
        0.
    """
    data_row_start = 0
    nan_group_idx = 0
    for agg_res_idx, size in ndenumerate(group_sizes):
        (agg_res_idx,) = agg_res_idx
        if size != 0:
            data_row_end = data_row_start + size
            if len(agg_func_float) != 0:
                # Manage float.
                _jitted_agg_func_router(
                    data_float,
                    data_row_start,
                    data_row_end,
                    agg_func_float,
                    n_cols_float,
                    cols_in_data_float,
                    cols_in_agg_res_float,
                    agg_res_idx,
                    agg_res_float,
                )
            if len(agg_func_int) != 0:
                # Manage int.
                _jitted_agg_func_router(
                    data_int,
                    data_row_start,
                    data_row_end,
                    agg_func_int,
                    n_cols_int,
                    cols_in_data_int,
                    cols_in_agg_res_int,
                    agg_res_idx,
                    agg_res_int,
                )
            if len(agg_func_dte) != 0:
                # Manage int.
                _jitted_agg_func_router(
                    data_dte,
                    data_row_start,
                    data_row_end,
                    agg_func_dte,
                    n_cols_dte,
                    cols_in_data_dte,
                    cols_in_agg_res_dte,
                    agg_res_idx,
                    agg_res_dte,
                )
            data_row_start = data_row_end
        else:
            nan_group_indices[nan_group_idx] = agg_res_idx
            nan_group_idx += 1


def chaingroupby(
    by: Union[Grouper, Callable],
    bin_on: ndarray,
    binning_buffer: dict,
    agg: Union[Dict[str, Tuple[str, str]], Dict[dtype, list]],
    data: pDataFrame,
    null_float=nNaN,
    null_int=pNA,
    null_dte=pNaT,
) -> pDataFrame:
    """Group as per 'by', assuming group keys are ordered.

    Parameters
    ----------
    by : Union[Grouper, Callable]
        Callable or pandas Grouper to perform binning.
        If a Callable, is called with following parameters:
        ``by(binning_buffer, bin_on, group_keys, n_groups, n_nan_groups)``
        where:

          - ``bin_on``, same parameter as for ``chaingroupby``.
          - ``binning_buffer``, same parameter as for ``chaingroupby``.

        It has then to return:

          - ``group_keys``, a one-dimensional array containing all individual
            group keys, as they will appear in aggregation results. Keys can be
            of any type.
          - ``group_sizes``, a one-dimensional array of `int`, specifying the
            number of consecutive rows for a given group as found in
            ``bin_on``. A group size, can be 0, meaning corresponding
            row in aggregation result will be filled with null values.
          - ``n_groups``, an `int`, the number of different groups, that ``by``
            has to output.
          - ``n_nan_groups``, an `int`, the number of groups with a null row in
            aggregation results.

    by : Union[np.ndarray, Series]
        Array of group keys, of the same size as the input array. It does not
        contain directly key values, but the indices of each key expected in
        the resulting aggregated array.
        These indices are expected sorted, and the resulting aggregated array
        will be of length defined by ``by[-1] - by[0]``.
        This means there can be holes within the resulting aggregated array if
        there are holes in the indices in ``by``. However, no hole can be at
        start or end of array.
    agg : dict
        Definition of aggregation.
        If in the form ``Dict[str, Tuple[str, str]]``, then it is reworked to
        be in the 2nd form ``Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]]``.

        - in the form ``Dict[str, Tuple[str, str]]``,

          - keys are ``str``, requested output column name
          - values are ``tuple`` with 1st component a ``str`` for the input
            column name, and 2nd component a ``str`` for aggregation function
            name.

        - the 2nd form is that returned by the function ``setup_cgb_agg``.

    """
    if isinstance(next(iter(agg.keys())), str):
        # Reshape aggregation definition.
        agg = setup_cgb_agg(agg, data.dtypes.to_dict())
    # Both 'group_keys' and 'group_sizes' are expected to be the size of the
    # resulting aggregated array from 'groupby' operation.
    if isinstance(by, Grouper):
        group_keys, group_sizes, n_groups, n_nan_groups = by_time(bin_on, by)
    else:
        # 'by' binning, possibly jitted.
        group_keys, group_sizes, n_groups, n_nan_groups = by(bin_on, binning_buffer)
    # Initialize input parameters.
    # 'agg_res' contain rows for possible empty bins.
    nan_group_indices = zeros(n_nan_groups, dtype=DTYPE_INT64)
    if DTYPE_FLOAT64 in agg:
        # Manage float.
        (
            agg_func_idx_float,
            n_cols_float,
            cols_name_in_data_float,
            cols_idx_in_data_float,
            cols_name_in_agg_res_float,
            cols_idx_in_agg_res_float,
        ) = agg[DTYPE_FLOAT64]
        if n_cols_float:
            data_float = data.loc[:, [cols_name_in_data_float]].to_numpy(copy=False)
            agg_res_float = zeros((n_groups, len(cols_name_in_agg_res_float)), dtype=DTYPE_FLOAT64)
        else:
            data_float = ZEROS_AR_FLOAT64
            agg_res_float = ZEROS_AR_FLOAT64
    if DTYPE_INT64 in agg:
        # Manage int.
        (
            agg_func_idx_int,
            n_cols_int,
            cols_name_in_data_int,
            cols_idx_in_data_int,
            cols_name_in_agg_res_int,
            cols_idx_in_agg_res_int,
        ) = agg[DTYPE_INT64]
        if n_cols_int:
            data_int = data.loc[:, [cols_name_in_data_int]].to_numpy(copy=False)
            agg_res_int = zeros((n_groups, len(cols_name_in_agg_res_int)), dtype=DTYPE_INT64)
        else:
            data_int = ZEROS_AR_INT64
            agg_res_int = ZEROS_AR_INT64
    if DTYPE_DATETIME64 in agg:
        # Manage datetime.
        (
            agg_func_idx_dte,
            n_cols_dte,
            cols_name_in_data_dte,
            cols_idx_in_data_dte,
            cols_name_in_agg_res_dte,
            cols_idx_in_agg_res_dte,
        ) = agg[DTYPE_DATETIME64]
        if n_cols_dte:
            data_dte = data.loc[:, [cols_name_in_data_dte]].to_numpy(copy=False).view(DTYPE_INT64)
            agg_res_dte = zeros((n_groups, len(cols_name_in_agg_res_dte)), dtype=DTYPE_INT64)
        else:
            data_dte = ZEROS_AR_INT64
            agg_res_dte = ZEROS_AR_INT64
    # 'data_xxx' are numpy arrays, with columns in 'expected order', as defined
    # in 'cols_idx_in_data_xxx'.
    _jitted_cgb(
        group_sizes=group_sizes,
        data_float=data_float,  # 2d
        agg_func_float=agg_func_idx_float,  # 1d
        n_cols_float=n_cols_float,  # 1d
        cols_in_data_float=cols_idx_in_data_float,  # 2d
        cols_in_agg_res_float=cols_idx_in_agg_res_float,  # 2d
        agg_res_float=agg_res_float,  # 2d
        data_int=data_int,  # 2d
        agg_func_int=agg_func_idx_int,  # 1d
        n_cols_int=n_cols_int,  # 1d
        cols_in_data_int=cols_idx_in_data_int,  # 2d
        cols_in_agg_res_int=cols_idx_in_agg_res_int,  # 2d
        agg_res_int=agg_res_int,  # 2d
        data_dte=data_dte,  # 2d
        agg_func_dte=agg_func_idx_dte,  # 1d
        n_cols_dte=n_cols_dte,  # 1d
        cols_in_data_dte=cols_idx_in_data_dte,  # 2d
        cols_in_agg_res_dte=cols_idx_in_agg_res_dte,  # 2d
        agg_res_dte=agg_res_dte,  # 2d
        nan_group_indices=nan_group_indices,  # 1d
    )

    # set 'NaN value in agg_res to include NaN of NaT value if meaning full
    # depending dtype, if nan_n_groups != 0.

    # /!\ WiP /!\
    # 'agg_res' sent to 'jitted_cgb' of the expanded size already, with a default
    # 0 values, that preserves dtype
    # then default value for missing data is set depending final dtype expected:
    #   timestamp (if NaT - forced to NaT if missing rows, not keeping 0 as not possible),
    #   float (if NaN)

    # /!\ WiP /!\ tester également 'group_indices' if relevant


#   if group_idx[-1] != len(agg_res):
# initialize 'agg' with new size, depending type


# WiP
# Rename columns as expected.
#    return pDataFrame(index=group_keys)
