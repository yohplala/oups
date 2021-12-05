#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from os import path as os_path
import pytest
import zipfile

from oups import ParquetSet, sublevel, toplevel


TEST_DATA = 'test-data'

@sublevel
class SpaceTime:
    area:str
    season:str

def test_parquet_set_init(tmp_path):
    fn = os_path.join(TEST_DATA, 'dummy_store.zip')
    with zipfile.ZipFile(fn, 'r') as zip_ref:
        zip_ref.extractall(tmp_path)
    @toplevel
    class WeatherEntry:
        capital:str
        quantity:str
        spacetime:SpaceTime
    basepath = os_path.join(tmp_path, 'store')
    ps = ParquetSet(basepath, WeatherEntry)
    assert ps.basepath == basepath
    # 'keys' is empty as 'fields_sep' in example directories is '.',
    # while default 'fields_sep' is '-'.
    assert len(ps) == 0
    # Re-do with 'fields_sep' set to '.' to comply with example directories.
    @toplevel(fields_sep='.')
    class WeatherEntry:
        capital:str
        quantity:str
        spacetime:SpaceTime
    ps = ParquetSet(basepath, WeatherEntry)
    assert len(ps) == 3
    repr_ref = 'london.temperature.greenwich.summer\nparis.temperature.bastille.summer\nstockholm.pressure.skansen.fall'
    assert repr(ps) == repr_ref
    

   



# check error message when using a class not being a toplevel.
# check with a directory that can't be materialize into a key.
