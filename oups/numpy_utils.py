#!/usr/bin/env python3
"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""
from numpy import array
from numpy import searchsorted


def isin_ordered(element: array, test_elements: array, invert=False) -> array:
    """
    Check if array of elements is in a sorted array.

    Parameters
    ----------
    element : array
        Array of elements to check. Has to be sorted.
    test_elements : array
        Array of elements to test. Not necessarily sorted.
    invert : bool, optional
        If True, return the elements that are not in the 'element' array.

    Returns
    -------
    array
        Array of booleans of same length as 'test_elements' indicating if the
        test elements are in the 'element' array.

    """
    return (
        (
            searchsorted(element, test_elements, side="left")
            != searchsorted(element, test_elements, side="right") - 1
        )
        if invert
        else (
            searchsorted(element, test_elements, side="left")
            == searchsorted(element, test_elements, side="right") - 1
        )
    )
