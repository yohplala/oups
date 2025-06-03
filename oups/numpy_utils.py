#!/usr/bin/env python3
"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""
from typing import Tuple, Union

from numpy import array
from numpy import ones
from numpy import searchsorted


def isnotin_ordered(
    sorted_array: array,
    query_elements: array,
    return_insert_positions: bool = False,
) -> Union[array, Tuple[array, array]]:
    """
    Check if query elements are not present in a sorted array.

    Parameters
    ----------
    sorted_array : array
        Sorted array in which to search for elements.
        Must be sorted in ascending order.
    query_elements : array
        Array of elements to search for.
        Must be sorted in ascending order if containing elements which are
        are larger than the largest element in 'sorted_array'.
    return_insert_positions : bool, optional
        If True, also return the insert positions where unfound elements
        could be inserted into 'sorted_array', maintaining sort order.
        Default is False.

    Returns
    -------
    Union[array, Tuple[array, array]]
        If 'return_insert_positions' is False:
            Array of booleans with same length as 'query_elements', where
            True indicates the element is not found in 'sorted_array'.
        If 'return_insert_positions' is True:
            Tuple containing:
            - Array of booleans indicating which query elements are not found,
            - Array of insert positions for unfound elements (where they could
              be inserted into 'sorted_array', maintaining sort order)

    Examples
    --------
    >>> sorted_arr = np.array([1, 3, 5, 7, 9])
    >>> queries = np.array([2, 3, 6, 7])
    >>> isnotin_ordered(sorted_arr, queries)
    array([True, False, True, False])

    >>> not_found, insert_positions = isnotin_ordered(sorted_arr, queries, return_insert_positions=True)
    >>> not_found
    array([True, False, True, False])
    >>> insert_positions
    array([1, 3])  # positions where 2 and 6 could be inserted

    """
    insert_idx = searchsorted(sorted_array, query_elements, side="left")
    # Check if elements are found: they exist if insert position is valid
    # and the element at that position matches the query
    found_max_idx = searchsorted(insert_idx, len(sorted_array))
    is_not_found = ones(len(query_elements), dtype=bool)
    is_not_found[:found_max_idx] = (
        sorted_array[insert_idx[:found_max_idx]] != query_elements[:found_max_idx]
    )
    return (is_not_found, insert_idx[is_not_found]) if return_insert_positions else is_not_found
