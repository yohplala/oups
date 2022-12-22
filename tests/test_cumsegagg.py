#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import array

from oups.cumsegagg import jmax
from oups.cumsegagg import jmin
from oups.cumsegagg import jsum


INT64 = "int64"
FLOAT64 = "float64"


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jmax(dtype_):
    # Test 'jmax()'.
    ar = array([], dtype=dtype_)
    assert jmax(ar) is None
    assert jmax(ar, 3) == 3
    ar = array([2], dtype=dtype_)
    assert jmax(ar) == 2
    assert jmax(ar, 5) == 5
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jmax(ar) == 3
    assert jmax(ar, 4) == 4


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jmin(dtype_):
    # Test 'jmin()'.
    ar = array([], dtype=dtype_)
    assert jmin(ar) is None
    assert jmin(ar, 3) == 3
    ar = array([8], dtype=dtype_)
    assert jmin(ar) == 8
    assert jmin(ar, 5) == 5
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jmin(ar) == -1
    assert jmin(ar, -3) == -3


@pytest.mark.parametrize(
    "dtype_",
    [FLOAT64, INT64],
)
def test_jsum(dtype_):
    # Test 'jsum()'.
    ar = array([], dtype=dtype_)
    assert jsum(ar) is None
    assert jsum(ar, 3) == 3
    ar = array([8], dtype=dtype_)
    assert jsum(ar) == 8
    assert jsum(ar, 5) == 13
    ar = array([1, 3, 2, -1], dtype=dtype_)
    assert jsum(ar) == 5
    assert jsum(ar, -3) == 2
