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
# Null values.
NULL_DICT = {DTYPE_INT64: pNA, DTYPE_FLOAT64: nNaN, DTYPE_DATETIME64: pNaT}
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
        if in_col not in data_dtype:
            raise KeyError(f"{in_col} not in input data.")
        else:
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
def _histo_on_ordered(data: ndarray, bins: ndarray, right: bool, histo: ndarray):
    """Histogram on ordered data.

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
        histo[b_idx_loc] = _d_idx_loc
        if _d_idx + 1 == data_len and prev_bin:
            # Array 'data' terminated and loop stayed in previous bin.
            # Then, last loop has not been accounted for.
            # Hence a '+1' to account for it.
            histo[b_idx_loc] = _d_idx_loc + 1
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

    Returns
    -------
    group_keys : ndarray
        One-dimensional array of keys (labels) of each group.
    group_sizes : ndarray
        One-dimensional array of `int` specifying the number of rows in
        'bin_on' for each group. An empty group has a size 0.
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
    group_sizes = zeros(len(group_keys), dtype=DTYPE_INT64)
    _histo_on_ordered(
        bin_on.to_numpy(copy=False).view(DTYPE_INT64),
        bins.to_numpy(copy=False).view(DTYPE_INT64),
        by.closed == "right",
        group_sizes,
    )
    return group_keys, group_sizes


@guvectorize(
    [
        (
            int64[:],
            int64[:, :],
            int64[:],
            int64[:],
            int64[:, :],
            int64[:, :],
            boolean,
            int64[:, :],
            int64[:],
        ),
        (
            int64[:],
            float64[:, :],
            int64[:],
            int64[:],
            int64[:, :],
            int64[:, :],
            boolean,
            float64[:, :],
            int64[:],
        ),
    ],
    "(l),(m,n),(o),(o),(o,p),(o,p),(),(l,k),(i)",
    nopython=True,
)
def _jitted_cgb(
    group_sizes: ndarray,  # 1d
    data: ndarray,  # 2d
    agg_func: ndarray,  # 1d
    n_cols: ndarray,  # 1d
    cols_in_data: ndarray,  # 2d
    cols_in_agg_res: ndarray,  # 2d
    assess_null_group_indices: bool,
    agg_res: ndarray,  # 2d
    null_group_indices: ndarray,  # 1d
):
    """Group assuming contiguity.

    Parameters
    ----------
    group_sizes : ndarray
        One dimensional array of ``int``, indicating the size of the groups.
        May contain ``0`` if a group key without any value is in resulting
        aggregation array.
    data : ndarray
        Array over which performing aggregation functions.
    agg_func : ndarray
        One dimensional array of ``int``, specifying the aggregation function
        ids.
    n_cols : ndarray
        One dimensional array of ``int``, specifying per aggregation function
        the number of columns to which applying related aggregation function
        (and consequently the number of columns in 'agg_res' to which recording
        the aggregation results).
    cols_in_data : ndarray
        Two dimensional array of ``int``, one row per aggregation function.
        Per row, column indices in 'data' to which apply corresponding
        aggregation function.
        Any value in column past the number of relevant columns is not used.
    cols_in_agg_res :  ndarray
        Two dimensional array of ``int``, one row per aggregation function.
        Per row, column indices in 'agg_res' into which storing the
        aggregation results.
        Any value in column past the number of relevant columns is not used.
    assess_null_group_indices : bool
       If `True`, assess row indices of null groups.

    Returns
    -------
    agg_res : ndarray
        Results from aggregation, with same `dtype` than 'data' array.
    null_group_indices : ndarray
        One dimensional array containing row indices in 'agg_res' that
        correspond to "empty" groups, i.e. for which group size has been set to
        0.
    """
    data_row_start = 0
    null_group_idx = 0
    for (agg_res_idx,), size in ndenumerate(group_sizes):
        if size != 0:
            data_row_end = data_row_start + size
            for (idx_func,), func in ndenumerate(agg_func):
                n_cols_ = n_cols[idx_func]
                data_chunk = data[data_row_start:data_row_end, cols_in_data[idx_func, :n_cols_]]
                if func == ID_FIRST:
                    for col_i in range(n_cols_):
                        agg_res[agg_res_idx, cols_in_agg_res[idx_func, col_i]] = data_chunk[
                            0, col_i
                        ]
                elif func == ID_LAST:
                    for col_i in range(n_cols_):
                        agg_res[agg_res_idx, cols_in_agg_res[idx_func, col_i]] = data_chunk[
                            -1, col_i
                        ]
                elif func == ID_MIN:
                    for col_i in range(n_cols_):
                        agg_res[agg_res_idx, cols_in_agg_res[idx_func, col_i]] = nmin(
                            data_chunk[:, col_i]
                        )
                elif func == ID_MAX:
                    for col_i in range(n_cols_):
                        agg_res[agg_res_idx, cols_in_agg_res[idx_func, col_i]] = nmax(
                            data_chunk[:, col_i]
                        )
                elif func == ID_SUM:
                    for col_i in range(n_cols_):
                        agg_res[agg_res_idx, cols_in_agg_res[idx_func, col_i]] = nsum(
                            data_chunk[:, col_i]
                        )
            data_row_start = data_row_end
        elif assess_null_group_indices:
            null_group_indices[null_group_idx] = agg_res_idx
            null_group_idx += 1


def chaingroupby(
    by: Union[Grouper, Callable],
    agg: Union[Dict[str, Tuple[str, str]], Dict[dtype, list]],
    data: pDataFrame,
    bin_on: Union[Series, str, None] = None,
    binning_buffer: dict = None,
) -> pDataFrame:
    """Group as per 'by', assuming group keys are ordered.

    Parameters
    ----------
    by : Union[Grouper, Callable]
        Callable or pandas Grouper to perform binning.
        If a Callable, is called with following parameters:
        ``by(bin_on, binning_buffer)``
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

    bin_on : Union[Series, str, None]
        A pandas Series over which performing the binning operation.
        If 'by' is a pandas `Grouper`, its `key` parameter is used instead, and
        'bin_on' can be left to `None` as per default.
        If a `str`, then corresponding column in `data` is used.
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

    data: pDataFrame
        A pandas Dataframe (pandas) containing the columns over which
        performing aggregations and with same length as that of ``bin_on``.
    binning_buffer : Union[dict, None]
        User-chosen values from previous binning process, that can be required
        when restarting the binning process with new seed data.

    Returns
    -------
    pDataFrame
        A pandas DataFrame with aggregation results. Its index is composed of
        the group keys.
    """
    if isinstance(next(iter(agg.keys())), str):
        # Reshape aggregation definition.
        agg = setup_cgb_agg(agg, data.dtypes.to_dict())
    # Both 'group_keys' and 'group_sizes' are expected to be the size of the
    # resulting aggregated array from 'groupby' operation.
    if isinstance(by, Grouper):
        if by.key:
            bin_on = data.loc[:, by.key]
        group_keys, group_sizes = by_time(bin_on, by)
    else:
        if isinstance(bin_on, str):
            bin_on = data.loc[:, bin_on]
        elif bin_on is None:
            raise ValueError("not possible to have 'bin_on' set to `None`.")
        # 'by' binning, possibly jitted.
        group_keys, group_sizes = by(bin_on, binning_buffer)
    # Initialize input parameters.
    n_groups = len(group_keys)
    # 'agg_res' contain rows for possible empty bins.
    # Count zeros.
    n_null_groups = count_nonzero(group_sizes == 0)
    null_group_indices = zeros(n_null_groups, dtype=DTYPE_INT64)
    assess_null_group_indices = True if n_null_groups else False
    # Initiate dict of result columns.
    agg_res = {}
    for dtype_, (
        agg_func_idx,  # 1d
        n_cols,  # 1d
        cols_name_in_data,
        cols_idx_in_data,
        cols_name_in_agg_res,
        cols_idx_in_agg_res,
    ) in agg.items():
        data_single_dtype = (
            data.loc[:, cols_name_in_data].to_numpy(copy=False)
            if len(cols_name_in_data) > 1
            else data.loc[:, cols_name_in_data].to_numpy(copy=False).reshape(-1, 1)
        )
        agg_res_single_dtype = zeros((n_groups, len(cols_name_in_agg_res)), dtype=dtype_)
        agg_res.update(
            {name: agg_res_single_dtype[:, i] for i, name in enumerate(cols_name_in_agg_res)}
        )
        if dtype_ == DTYPE_DATETIME64:
            data_single_dtype = data_single_dtype.view(DTYPE_INT64)
            agg_res_single_dtype = agg_res_single_dtype.view(DTYPE_INT64)
        # 'data' are numpy arrays, with columns in 'expected order', as defined
        # in 'cols_idx_in_data'.
        _jitted_cgb(
            group_sizes,  # 1d
            data_single_dtype,  # 2d
            agg_func_idx,  # 1d
            n_cols,  # 1d
            cols_idx_in_data,  # 2d
            cols_idx_in_agg_res,  # 2d
            assess_null_group_indices,  # bool
            agg_res_single_dtype,  # 2d
            null_group_indices,  # 1d
        )
        assess_null_group_indices = False
    # Assemble 'agg_res' as a pandas DataFrame.
    agg_res = pDataFrame(agg_res, index=group_keys, copy=False)
    agg_res.index.name = bin_on.name
    # Set null values.
    if n_null_groups != 0:
        null_group_keys = group_keys[null_group_indices]
        for dtype_, (
            _,
            _,
            _,
            _,
            cols_name_in_agg_res,
            _,
        ) in agg.items():
            agg_res.loc[null_group_keys, cols_name_in_agg_res] = NULL_DICT[dtype_]
    return agg_res
