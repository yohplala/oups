#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from os import path as os_path
import pytest
import zipfile

import pandas as pd
from fastparquet import ParquetFile

from oups import ParquetSet, sublevel, toplevel


TEST_DATA = 'test-data'

@sublevel
class SpaceTime:
    area:str
    season:str

def test_parquet_set_init(tmp_path):
    # Test parquet set 'discovery' from existing directories.
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
    # Test '__repr__'.
    repr_ref = ('london.temperature.greenwich.summer\n'
                'paris.temperature.bastille.summer\n'
                'stockholm.pressure.skansen.fall')
    assert repr(ps) == repr_ref
    # Test '__contains__'.
    key = WeatherEntry('london','temperature', SpaceTime('greenwich','summer'))
    assert key in ps

def test_exception_key_not_correct_indexer(tmp_path):
    # Test with class not being 'toplevel'.
    class WeatherEntry:
        capital:str
        quantity:str
        spacetime:SpaceTime
    basepath = os_path.join(tmp_path, 'store')
    with pytest.raises(TypeError, match='^WeatherEntry'):
        ParquetSet(basepath, WeatherEntry)

@toplevel
class WeatherEntry:
    capital:str
    quantity:str
    spacetime:SpaceTime

def test_set_parquet(tmp_path):
    # Initialize a parquet dataset.
    basepath = os_path.join(tmp_path, 'store')    
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry('paris', 'temperature', SpaceTime('notredame', 'winter'))
    df = pd.DataFrame({'timestamp': pd.date_range('2021/01/01 08:00',
                                                  '2021/01/01 10:00',
                                                  freq='2H'),
                       'temperature': [8.4, 5.3]})
    ps[we] = df
    assert we in ps
    res = ParquetFile(os_path.join(basepath, we.to_path)).to_pandas()
    assert res.equals(df)

def test_set_parquet_with_config(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, 'store')    
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry('paris', 'temperature', SpaceTime('notredame', 'winter'))
    df = pd.DataFrame({'timestamp': pd.date_range('2021/01/01 08:00',
                                                  '2021/01/01 14:00',
                                                  freq='2H'),
                       'temperature': [8.4, 5.3, 4.9, 2.3]})
    rg_size=2
    config = {'row_group_size':rg_size}
    ps[we] = config, df
    assert we in ps
    # Load only first row group.
    res = ParquetFile(os_path.join(basepath, we.to_path))[0].to_pandas()
    assert res.equals(df.loc[:rg_size-1])

def test_exception_config_not_a_dict(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, 'store')    
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry('paris', 'temperature', SpaceTime('notredame', 'winter'))
    df = pd.DataFrame({'timestamp': pd.date_range('2021/01/01 08:00',
                                                  '2021/01/01 14:00',
                                                  freq='2H'),
                       'temperature': [8.4, 5.3, 4.9, 2.3]})
    config = [] # First silly thing that comes to mind.
    with pytest.raises(TypeError, match='^First item'):
        ps[we] = config, df

def test_exception_data_not_a_dataframe(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, 'store')    
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry('paris', 'temperature', SpaceTime('notredame', 'winter'))
    df = [] # First silly thing that comes to mind.
    with pytest.raises(TypeError, match='^Data should'):
        ps[we] = df
