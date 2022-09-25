#!/usr/bin/env python3
"""
Created on Wed Dec  4 21:30:00 2021.

@author: yoh
"""
from typing import Callable, Dict, List, Tuple, Union

from numpy import dtype
from numpy import ndarray
from numpy import unique
from numpy import zeros
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Timestamp as pTimestamp


def time_bin(grouper: Grouper, bin_on: ndarray) -> Union[pTimestamp, None]:
    """Bin timestamp in corresponding time bin.

    Return a 2d array with 2 columns, 1st columns
    """


def setup_cgb_agg(
    agg: Dict[str, Tuple[str, str]], data_dtype: Dict[str, dtype]
) -> Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]:
    """Construct chaingrouby agg configuration.

    Parameters
    ----------
    agg: Dict[str, Tuple[str, str]]
        Dict specifying aggregation in the form
        ``'out_col_name' : ('in_col_name', 'function_name')``
    data_dtype: Dict[str, dtype]
        Dict specifying per column name its dtype. Typically obtained with
        ``df.dtypes.to_dict()``

    Returns
    -------
    Dict[dtype, Tuple[List[str], Tuple[int, str], List[str]]]
         Dict in the form list in form
         ``{dtype: List[str], column name in input data,
                   List[Tuple[int, str]], column posiiton in input data, function name
                   List[str], column name axpected in aggregation result}``
    """
    cgb_agg_cfg = {}
    for out_col, (in_col, func) in agg.items():
        dtype = data_dtype[in_col]
        try:
            tup = cgb_agg_cfg[dtype]
        except KeyError:
            cgb_agg_cfg[dtype] = [[], [], []]
            tup = cgb_agg_cfg[dtype]
        if in_col in tup[0]:
            in_col_idx = tup[0].index(in_col)
        else:
            in_col_idx = len(tup[0])
            tup[0].append(in_col)
        tup[1].append((in_col_idx, func))
        tup[2].append(out_col)

    return cgb_agg_cfg


def chaingroupby(
    by: Union[Grouper, Callable],
    bin_on: ndarray,
    binning_buffer: dict,
    agg: Dict[str, Tuple[str, str]],
    data: pDataFrame,
    agg_col_types: Dict[str, List[str]] = None,
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

    agg_col_types: Dict
    """
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

    # Whatever 'by', resulting from this step,
    # - output 'group_keys' is expected to be the size of the resulting
    #   aggregated array from 'groupby' operation.
    # - output 'group_sizes' is expected to be the size of the resulting
    #   aggregated array from 'groupby' operation.
    if isinstance(by, Grouper):
        group_keys, group_sizes = time_bin(by, bin_on)
    else:
        # Bin, possibly jitted.
        group_keys = zeros(len(bin_on))
        by(binning_buffer, bin_on, group_keys)
        group_keys, group_sizes = unique(group_keys, return_counts=True)

    # WiP
    # Transform agg into suitable form.
    # Dict[str, Tuple[str, str]] -> Tuple[int, str]

    # Create 'agg_res'.
    agg_res = zeros((len(group_keys), len(agg)))

    # WiP
    # Loop per dtype.
    # Forward 'data' as numpy array, only with column in 'expected order'.
    jitted_cgb(group_sizes, agg, data, agg_res)

    # WiP
    # Rename columns as expected.
    return pDataFrame(index=group_keys)


def jitted_cgb(group_indices: ndarray, agg: List[Tuple[int, str]], data: ndarray, agg_res: ndarray):
    """Group as per by, assuming input group keys are contiguous and sorted.

    Parameters
    ----------
    by : Union[ndarray, Series]
        Array of group keys, of the same size as the input array. It does not
        contain directly key values, but the indices of each key expected in
        the resulting aggregated array.
        These indices are expected sorted, and the resulting aggregated array
        will be of length defined by ``by[-1] - by[0]``.
        This means there can be holes within the resulting aggregated array if
        there are holes in the indices in ``by``. However, no hole can be at
        start or end of array.
    """

    # Beware that group_indices provide index to which recording aggregated result
    # in resulting array.
