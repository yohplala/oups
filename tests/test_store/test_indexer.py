#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
from dataclasses import FrozenInstanceError
from dataclasses import asdict
from dataclasses import fields
from os.path import sep
from pathlib import Path

import pytest
from cloudpickle import dumps
from cloudpickle import loads

from oups import is_toplevel
from oups import sublevel
from oups import toplevel
from oups.store.indexer import DEFAULT_FIELD_SEP


def test_toplevel_is_toplevel():
    @toplevel
    class Test:
        mu: int
        nu: str

    assert is_toplevel(Test)

    @sublevel
    class Test:
        mu: int
        nu: str

    assert not is_toplevel(Test)


def test_toplevel_to_str():
    # Test without parameter.
    @toplevel
    class Test:
        mu: int
        nu: str

    test = Test(3, "oh")
    assert str(test) == "3-oh"

    # Test with 'field_sep' parameter.
    @toplevel(field_sep=".")
    class Test:
        mu: int
        nu: str

    test = Test(3, "oh")
    assert str(test) == "3.oh"
    # Test 'toplevel' is a dataclass only exposing attribute defined by user,
    # not those added in 'top_level' definition like 'field_sep'.
    attrs = list(asdict(test))
    assert attrs == ["mu", "nu"]


def test_toplevel_equality_and_dict():
    # Test (un)equality.
    @toplevel
    class Test:
        mu: int
        nu: str

    test1 = Test(3, "oh")
    test2 = Test(3, "oh")
    assert test1 == test2
    test3 = Test(4, "oh")
    assert test1 != test3

    # Test dict.
    di = {test1: [1, 2, 3], test2: [7, 8, 9]}
    assert len(di) == 1
    assert di[Test(3, "oh")] == [7, 8, 9]


# Test material for nested dataclass.
@sublevel
class SubLevel2:
    ma: str
    to: int
    ou: int


@sublevel
class SubLevel1:
    pu: int
    il: str
    iv: SubLevel2


@toplevel
class TopLevel:
    ma: str
    to: int
    of: SubLevel1


def test_toplevel_nested_dataclass_to_str():
    # Test 'normal' use case.
    # (only int, str or dataclass instance in last position)
    sl2 = SubLevel2("ou", 3, 7)
    sl1 = SubLevel1(5, "oh", sl2)
    # Should not raise an exception.
    tl = TopLevel("aha", 2, sl1)
    to_str_ref = "aha-2-5-oh-ou-3-7"
    assert str(tl) == to_str_ref

    # Test serialization.
    unserialized = loads(dumps(tl))
    assert unserialized == tl


def test_toplevel_nested_dataclass_attributes():
    @toplevel(field_sep=".")
    class TopLevel:
        ma: str
        to: int
        of: SubLevel1

    sl2 = SubLevel2("ou", 3, 7)
    sl1 = SubLevel1(5, "oh", sl2)
    tl = TopLevel("aha", 2, sl1)
    assert len(fields(tl)) == 3
    assert TopLevel._field_sep == "."
    assert TopLevel._depth == 3
    assert TopLevel.depth == 3
    assert tl._field_sep == "."
    assert tl._depth == 3
    assert tl.depth == 3
    # Check frozen instance error.
    with pytest.raises(FrozenInstanceError, match="^cannot assign"):
        tl.field_sep = "-"
    with pytest.raises(FrozenInstanceError, match="^cannot assign"):
        tl.depth = 4
    with pytest.raises(AttributeError, match="^can't set"):
        TopLevel.field_sep = "-"
    with pytest.raises(AttributeError, match="^can't set"):
        TopLevel.depth = 4


def test_sublevel_single_attribute_to_path():
    @sublevel
    class SubLevel1:
        pu: str

    sl1 = SubLevel1("oh")
    tl = TopLevel("ah", 5, sl1)
    ref_path = Path(f"ah{DEFAULT_FIELD_SEP}5", "oh")
    assert tl.to_path() == ref_path


def test_toplevel_nested_dataclass_validation():
    # Test validation with wrong data type (dict).
    @sublevel
    class SubLevel1:
        pu: int
        il: dict
        iv: SubLevel2

    sl2 = SubLevel2("ou", 3, 7)
    sl1 = SubLevel1(5, {5: ["ah"]}, sl2)
    with pytest.raises(TypeError, match="^field type"):
        TopLevel("aha", 2, sl1)

    # Test validation with several dataclass instance at same level.
    @sublevel
    class SubLevel1:
        pu: int
        il: str
        iv: SubLevel2
        po: SubLevel2

    sl2_ = SubLevel2("fi", 9, 2)
    sl1 = SubLevel1(5, "oh", sl2, sl2_)
    with pytest.raises(TypeError, match="^a dataclass instance is only"):
        TopLevel("aha", 2, sl1)

    # Test validation with a single dataclass instance in a level.
    @sublevel
    class SubLevel1:
        iv: SubLevel2

    sl1 = SubLevel1(sl2)
    with pytest.raises(TypeError, match="^a dataclass instance cannot be"):
        TopLevel("aha", 2, sl1)

    # Test validation with a string embedding a forbidden character: directory
    # separator.
    @sublevel
    class SubLevel1:
        pu: int
        il: str
        iv: SubLevel2

    sl1 = SubLevel1(4, f"6{sep}2", sl2)
    with pytest.raises(ValueError, match="^use of a forbidden"):
        TopLevel("aha", 2, sl1)

    # Test validation with a string embedding a forbidden character: field_sep.
    sl1 = SubLevel1(4, f"6{DEFAULT_FIELD_SEP}2", sl2)
    with pytest.raises(ValueError, match="^use of a forbidden"):
        TopLevel("aha", 2, sl1)


def test_toplevel_nested_dataclass_str_roundtrip_3_levels():
    # Test '._to_path', '_from_str' and '_from_path'.
    sl2 = SubLevel2("ou", 3, 7)
    sl1 = SubLevel1(5, "oh", sl2)
    tl = TopLevel("aha", 2, sl1)
    path_res = tl.to_path()
    path_ref = Path("aha-2", "5-oh", "ou-3-7")
    assert path_res == path_ref
    tl_from_path = TopLevel.from_path(path_res)
    assert tl == tl_from_path
    # Checking with only 'field_sep' in string.
    str_res = str(tl)
    tl_from_str = TopLevel.from_str(str_res)
    assert tl == tl_from_str


def test_toplevel_nested_dataclass_str_roundtrip_2_levels():
    # Testing 2 levels.
    @sublevel
    class SubLevel1:
        pu: int
        il: str

    @toplevel
    class TopLevel:
        ma: str
        of: SubLevel1

    # Test '._to_path', '_from_str' and '_from_path'.
    sl1 = SubLevel1(5, "oh")
    tl = TopLevel("aha", sl1)
    path_res = tl.to_path()
    path_ref = Path("aha", "5-oh")
    assert path_res == path_ref
    tl_from_path = TopLevel.from_path(path_res)
    assert tl == tl_from_path
    # Checking with only 'field_sep' in string.
    str_res = str(tl)
    tl_from_str = TopLevel.from_str(str_res)
    assert tl == tl_from_str

    # Test 'from_path' returning None because of incorrect 'field_sep'.
    @toplevel(field_sep=".")
    class TopLevel:
        ma: str
        of: SubLevel1

    tl_from_path = TopLevel.from_path(path_res)
    assert tl_from_path is None
