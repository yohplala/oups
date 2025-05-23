#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

Test utils.
- Initialize path:
tmp_path = os_path.expanduser('~/Documents/code/data/oups')

"""
import zipfile
from os import path as os_path

import pytest
from fastparquet import ParquetFile
from pandas import DataFrame as pDataFrame
from pandas import MultiIndex
from pandas import date_range

from oups import ParquetSet
from oups import sublevel
from oups import toplevel
from oups.aggstream.segmentby import KEY_ORDERED_ON
from oups.store.write import KEY_ROW_GROUP_TARGET_SIZE

from .. import TEST_DATA


@sublevel
class SpaceTime:
    area: str
    season: str


def test_parquet_set_init(tmp_path):
    # Test parquet set 'discovery' from existing directories.
    fn = os_path.join(TEST_DATA, "dummy_store.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    assert ps.basepath == basepath
    # 'keys' is empty as 'fields_sep' in example directories is '.',
    # while default 'fields_sep' is '-'.
    assert len(ps) == 0

    # Re-do with 'fields_sep' set to '.' to comply with example directories.
    @toplevel(fields_sep=".")
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    ps = ParquetSet(basepath, WeatherEntry)
    assert len(ps) == 3
    # Test '__repr__'.
    repr_ref = (
        "london.temperature.greenwich.summer\n"
        "paris.temperature.bastille.summer\n"
        "stockholm.pressure.skansen.fall"
    )
    assert repr(ps) == repr_ref
    # Test '__contains__'.
    key = WeatherEntry("london", "temperature", SpaceTime("greenwich", "summer"))
    assert key in ps


def test_exception_key_not_correct_indexer(tmp_path):
    # Test with class not being 'toplevel'.
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    basepath = os_path.join(tmp_path, "store")
    with pytest.raises(TypeError, match="^WeatherEntry"):
        ParquetSet(basepath, WeatherEntry)


@toplevel
class WeatherEntry:
    capital: str
    quantity: str
    spacetime: SpaceTime


def test_set_parquet(tmp_path):
    # Initialize a parquet dataset.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 10:00", freq="2h"),
            "temperature": [8.4, 5.3],
        },
    )
    ps[we] = {KEY_ORDERED_ON: "timestamp"}, df
    assert we in ps
    res = ParquetFile(os_path.join(basepath, we.to_path)).to_pandas()
    assert res.equals(df)


def test_set_parquet_with_config(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    rg_size = 2
    config = {KEY_ORDERED_ON: "timestamp", KEY_ROW_GROUP_TARGET_SIZE: rg_size}
    ps[we] = config, df
    assert we in ps
    # Load only first row group.
    res = ParquetFile(os_path.join(basepath, we.to_path))[0].to_pandas()
    assert res.equals(df.loc[: rg_size - 1])


def test_exception_config_not_a_dict(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    config = []  # First silly thing that comes to mind.
    with pytest.raises(TypeError, match="^first item"):
        ps[we] = config, df


def test_exception_data_not_a_dataframe(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = []  # First silly thing that comes to mind.
    with pytest.raises(TypeError, match="^data should"):
        ps[we] = df


def test_iterator(tmp_path):
    # Test `__iter__`.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    we2 = WeatherEntry("london", "temperature", SpaceTime("greenwich", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    ps[we1], ps[we2] = ({KEY_ORDERED_ON: "timestamp"}, df), ({KEY_ORDERED_ON: "timestamp"}, df)
    for key in ps:
        assert key in (we1, we2)


def test_set_and_get_roundtrip_pandas(tmp_path):
    # Set and get data, roundtrip.
    ps = ParquetSet(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 2.9, 6.4],
        },
    )
    config = {KEY_ORDERED_ON: "timestamp", KEY_ROW_GROUP_TARGET_SIZE: 2}
    ps[we] = config, df
    df_res = ps[we].pdf
    assert df_res.equals(df)


def test_set_pandas_and_get_vaex(tmp_path):
    # Set and get data, roundtrip.
    ps = ParquetSet(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 2.9, 6.4],
        },
    )
    config = {KEY_ORDERED_ON: "timestamp", KEY_ROW_GROUP_TARGET_SIZE: 2}
    ps[we] = config, df
    vdf = ps[we].vdf
    assert vdf.to_pandas_df().equals(df)


def test_set_pandas_and_get_parquet_file(tmp_path):
    # Set and get data, roundtrip.
    ps = ParquetSet(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 2.9, 6.4],
        },
    )
    config = {KEY_ORDERED_ON: "timestamp", KEY_ROW_GROUP_TARGET_SIZE: 2}
    ps[we] = config, df
    pf = ps[we].pf
    assert len(pf.row_groups) == 2
    assert sorted(df.columns) == sorted(pf.columns)


def test_set_cmidx_get_vaex(tmp_path):
    # Write column multi-index in pandas, retrieve in vaex.
    pdf = pDataFrame(
        {
            ("ts", ""): date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            ("temp", "1"): [8.4, 5.3, 2.9, 6.4],
            ("temp", "2"): [8.4, 5.3, 2.9, 6.4],
        },
    )
    pdf.columns = MultiIndex.from_tuples(
        [("ts", ""), ("temp", "1"), ("temp", "2")],
        names=["component", "point"],
    )
    ps = ParquetSet(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    config = {KEY_ORDERED_ON: ("ts", "")}
    ps[we] = config, pdf
    vdf = ps[we].vdf
    assert list(map(str, pdf.columns)) == vdf.get_column_names()
    df_res = vdf.to_pandas_df()
    df_res.columns = pdf.columns
    assert df_res.equals(pdf)


def test_dataset_removal(tmp_path):
    # Test `__delitem__`.
    basepath = os_path.join(tmp_path, "store")
    ps = ParquetSet(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    we2 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "summer"))
    we3 = WeatherEntry("london", "temperature", SpaceTime("greenwich", "winter"))
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    config = {KEY_ORDERED_ON: "timestamp"}
    ps[we1], ps[we2], ps[we3] = (config, df), (config, df), (config, df)
    # Delete london-related data.
    we3_path = os_path.join(basepath, we3.to_path)
    assert os_path.exists(we3_path)
    assert len(ps) == 3
    del ps[we3]
    assert not os_path.exists(we3_path)
    assert len(ps) == 2
    # Delete paris-summer-related data.
    we2_path = os_path.join(basepath, we2.to_path)
    assert os_path.exists(we2_path)
    del ps[we2]
    assert not os_path.exists(we2_path)
    assert len(ps) == 1
    # Check paris-winter-related data still exists.
    we1_path = os_path.join(basepath, we1.to_path)
    assert os_path.exists(we1_path)


def test_11_rgs_pandas_to_vaex(tmp_path):
    # With 11 row groups, 'bug' related to the way sort files in lexicographic
    # order to read them is apparent.
    ps = ParquetSet(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    temp = range(10, 21)
    df = pDataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", freq="2h", periods=len(temp)),
            "temperature": temp,
        },
    )
    config = {KEY_ORDERED_ON: "timestamp", KEY_ROW_GROUP_TARGET_SIZE: 1}
    ps[we] = config, df
    vdf = ps[we].vdf
    assert vdf.to_pandas_df().equals(df)
