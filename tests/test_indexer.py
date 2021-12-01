#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import dataclass
import pytest

from oups import top_level
from oups.indexer import DIR_SEP


def test_top_level_to_str():
    # Test without parameter.
    @top_level
    class Test:
        mu : int
        nu : str
    test = Test(3, 'oh')
    assert str(test) == '3-oh'
    
    # Test with 'fields_sep' parameter.
    @top_level(fields_sep='.')
    class Test:
        mu : int
        nu : str
    test = Test(3, 'oh')
    assert str(test) == '3.oh'

def test_top_level_equality_and_dict():    
    # Test (un)equality.
    @top_level
    class Test:
        mu : int
        nu : str
    test1 = Test(3, 'oh')
    test2 = Test(3, 'oh')
    assert test1 == test2
    test3 = Test(4, 'oh')
    assert test1 != test3
    
    # Test dict.
    di = {test1:[1,2,3], test2:[7,8,9]}
    assert len(di) == 1
    assert di[Test(3, 'oh')] == [7,8,9]

def test_top_level_nested_dataclass():
    # Test 'normal' use case.
    # (only int, float, str or dataclass instance in last position)
    @dataclass(order=True, frozen=True)
    class SubLevel2:
        ma: str
        to: int
        ou: float    

    @dataclass(order=True, frozen=True)
    class SubLevel1:
        pu: float
        il: str
        iv: SubLevel2

    @top_level
    class TopLevel:
        ma: str
        to: int
        fo: SubLevel1

    sl2 = SubLevel2('ou', 3, 7.2)
    sl1 = SubLevel1(5.6, 'oh', sl2)
    # Should not raise an exception.
    tl = TopLevel('aha', 2, sl1)
    to_str_ref = 'aha-2-5.6-oh-ou-3-7.2'
    assert str(tl) == to_str_ref
    to_str_ref = DIR_SEP.join(['aha-2','5.6-oh','ou-3-7.2'])
    assert tl.to_path == to_str_ref

    # Test validation with wrong data type.
    @dataclass(order=True, frozen=True)
    class SubLevel1:
        pu: float
        il: dict
        iv: SubLevel2

    sl1 = SubLevel1(5.6, {5:['ah']}, sl2)
    with pytest.raises(TypeError, match='^Field type'):
        tl = TopLevel('aha', 2, sl1)

    # Test validation with several dataclass instance at same level.
    @dataclass(order=True, frozen=True)
    class SubLevel1:
        pu: float
        il: str
        iv: SubLevel2
        po: SubLevel2

    sl2_ = SubLevel2('fi', 9, 2.8)
    sl1 = SubLevel1(5.6, 'oh', sl2, sl2_)
    with pytest.raises(TypeError, match='^A dataclass instance is only'):
        tl = TopLevel('aha', 2, sl1)

    # Test validation with a single dataclass instance in a level.
    @dataclass(order=True, frozen=True)
    class SubLevel1:
        iv: SubLevel2

    sl1 = SubLevel1(sl2)
    with pytest.raises(TypeError, match='^A dataclass instance cannot be'):
        tl = TopLevel('aha', 2, sl1)
