#!/usr/bin/env python3
"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""
import pytest
from numpy import array
from numpy import array_equal
from numpy import nan

from oups.numpy_utils import bfill1d
from oups.numpy_utils import ffill1d
from oups.numpy_utils import isnotin_ordered


@pytest.mark.parametrize(
    "expected_insert_positions",
    [
        None,
        array([0, 5]),
    ],
)
def test_isnotin_ordered(expected_insert_positions):
    sorted_array = array([1, 2, 3, 4, 5])
    query_elements = array([0, 1, 5, 10])
    res = isnotin_ordered(
        sorted_array=sorted_array,
        query_elements=query_elements,
        return_insert_positions=expected_insert_positions is not None,
    )
    expected_isnotin = array([True, False, False, True])
    if expected_insert_positions is None:
        assert array_equal(
            res,
            expected_isnotin,
        )
    else:
        assert array_equal(
            res[0],
            expected_isnotin,
        )
        assert array_equal(
            res[1],
            expected_insert_positions,
        )


def test_ffill1d():
    arr = ffill1d(array([nan, 1, nan, 3, nan, 5]))
    expected = array([nan, 1, 1, 3, 3, 5])
    assert array_equal(arr, expected, equal_nan=True)


def test_bfill1d():
    arr = array([1, 2, 3, 4, 5])
    expected = array([1, 2, 3, 4, 5])
    assert array_equal(bfill1d(arr), expected)
