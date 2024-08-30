#!/usr/bin/env python3
"""
Created on Wed Jan 24 21:30:00 2024.

@author: yoh

"""
import operator

from numpy import ndarray
from numpy import ones
from numpy import zeros
from pandas import DataFrame


ops = {
    "==": operator.eq,
    "=": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def dataframe_filter(df: DataFrame, filters) -> ndarray:
    """
    Produce a column filter of the input dataframe.

    Parameters
    ----------
    df : pDataFrame
        DataFrame to filter.
    filters : List[List[Tuple]]
       To filter out data from seed.
       Filter syntax: [[(column, op, val), ...],...]
       where op is [==, =, >, >=, <, <=, !=, in, not in, ~]
       The innermost tuples are transposed into a set of filters applied
       through an `AND` operation.
       The outer list combines these sets of filters through an `OR` operation.
       A single list of tuples can also be used, meaning that no `OR` operation
       between set of filters is to be conducted.

    Returns
    -------
    numpy 1D-array
       List of rows to keep.

    """
    if isinstance(filters[0], tuple):
        raise ValueError(
            "not possible to have a 'filters' parameter without at least an inner list.",
        )
    out = zeros(len(df), dtype=bool)
    for or_part in filters:
        and_part = ones(len(df), dtype=bool)
        for name, op, val in or_part:
            if op == "in":
                and_part &= df[name].isin(val).values
            elif op == "not in":
                and_part &= ~df[name].isin(val).values
            elif op in ops:
                and_part &= ops[op](df[name].values, val)
            elif op == "~":
                and_part &= ~df[name].values
            else:
                # Unknown operator.
                raise ValueError(f"operator '{op}' is not supported.")
        out |= and_part
    return out
