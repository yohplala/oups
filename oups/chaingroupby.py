#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from typing import Callable, Dict, List, Tuple, Union

from numpy import array
from numpy import dtype
from numpy import max as nmax
from numpy import min as nmin
from numpy import ndarray
from numpy import ndenumerate
from numpy import sum as nsum
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Timestamp as pTimestamp

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


def time_bin(grouper: Grouper, bin_on: ndarray) -> Union[pTimestamp, None]:
    """Bin timestamp in corresponding time bin.

    Return a 2d array with 2 columns, 1st columns
    """


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


def chaingroupby(
    by: Union[Grouper, Callable],
    bin_on: ndarray,
    binning_buffer: dict,
    agg: Union[
        Dict[str, Tuple[str, str]], Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]
    ],
    data: pDataFrame,
) -> pDataFrame:
    """Group as per by, assuming input group keys are contiguous and sorted.

    Parameters
    ----------
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
    # WiP /!\ REMOVE comments /!\
    # - if 'by' a Grouper,
    #       create array of group keys (with keys for hole) -> agg_res_len
    #       create array of indices for each row in input array,
    #       starting at 0, and with last value = agg_res_len - 1
    #       but of size = len(input_array_to_bin) (with possibly hole)
    #       -> move to 'time_bin' function
    # - if 'by' a Callable (possibly jitted),
    #       signature  bin_on: Union[np:ndarray, Series],
    #                  buffer: dict,
    #                  group_keys: np.ndarray (size of input array)
    #    #                  group_indices: np.ndarray (size of input array,
    #    #                                 indicating which row belong to which group)
    #       len(group_keys) = group_indices[-1]
    #

    # Both 'group_keys' and 'group_sizes' are expected to be the size of the
    # resulting aggregated array from 'groupby' operation.
    if isinstance(by, Grouper):
        group_keys, group_sizes, n_groups, n_nan_groups = time_bin(by, bin_on)
    else:
        # Bin, possibly jitted.
        group_keys = zeros(len(bin_on))
        by(binning_buffer, bin_on, group_keys, n_groups, n_nan_groups)
        # Here, all keys are supposed to be with non-null value in aggregation
        # results.
        # WiP
        # Recreate 'np.unique' function here, using n_unique_keys, that returns
        # - list of unique keys (all keys -> agg_res len)
        # - group_sizes
        # group_keys, group_sizes = unique(group_keys, return_counts=True)
        # len_reduced_agg_res = len(group_keys)

    # n_groups = len(group_keys)
    # for dtype_ in agg:
    # Create 'agg_res'.
    #        if dtype_ != :

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
    # Forward 'data' as numpy array, with columns in 'expected order'.
    jitted_cgb(
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


def jitted_agg_func_router(
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


def jitted_cgb(
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
    nan_group_indices: ndarray,  # 1d
):
    """Group as per by, assuming input group keys are contiguous and sorted.

    Parameters
    ----------
    group_sizes : ndarray
        Array of int, indicating the size of the groups. May contain ``0`` if
        a group key without any value is in resulting aggregation array.
    data_float : ndarray
        Array of ``float`` over which performaing aggregation functions.
    agg_float_func : ndarray
        One dimensional array of ``int``, specifying the aggregation function
        ids.
    n_cols_float : ndarray
        One dimensional array of ``int``, specifying per aggregation function
        the number of columns to which applying related aggregation function
        (and consequently the number of columns in 'agg_res' to which recording
         the aggregation results).
    cols_in_data_float : ndarray
        One row per aggregation function. Per row, column indices in
        'data_float' to which apply corresponding aggregation function.
        Any value in column past the number of relevant columns is not used.
    cols_in_agg_res_float :  ndarray
        One row per aggregation function. Per row, column indices in
        'agg_res_float' into which storing the agrgegation result.
        Any value in column past the number of relevant columns is not used.

    Returns
    -------
    agg_res_float : ndarray
        Results from aggregation, ``float`` dtype
    agg_res_int : ndarray
        Results from aggregation, ``int`` dtype
    nan_group_indices : ndarray
        Index of row in 'agg_res' to which 'NaN' values are to be set.
    """
    data_row_start = 0
    nan_group_idx = 0
    for agg_res_idx, size in ndenumerate(group_sizes):
        (agg_res_idx,) = agg_res_idx
        if size != 0:
            data_row_end = data_row_start + size
            if len(agg_func_float) != 0:
                # Manage float.
                jitted_agg_func_router(
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
                jitted_agg_func_router(
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
            data_row_start = data_row_end
        else:
            nan_group_indices[nan_group_idx] = agg_res_idx
            nan_group_idx += 1
