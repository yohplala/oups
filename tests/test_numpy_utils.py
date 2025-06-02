#!/usr/bin/env python3
"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""

from numpy import array
from numpy import array_equal

from oups.numpy_utils import isin_ordered


def test_isin_ordered():
    element = array([1, 2, 3, 4, 5])
    test_elements = array([0, 1, 2])
    assert array_equal(
        isin_ordered(element=element, test_elements=test_elements),
        array([False, True, True]),
    )
    assert array_equal(
        isin_ordered(element=element, test_elements=test_elements, invert=True),
        array([True, False, False]),
    )
