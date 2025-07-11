#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
import zipfile
from os import path as os_path
from pathlib import Path

import pytest
from pandas import DataFrame
from pandas import MultiIndex
from pandas import date_range

from oups import Store
from oups import sublevel
from oups import toplevel
from oups.store.ordered_parquet_dataset.metadata_filename import get_md_filepath

from ... import TEST_DATA


@sublevel
class SpaceTime:
    area: str
    season: str


def test_store_init(tmp_path):
    # Test store 'discovery' from existing directories.
    fn = Path(TEST_DATA, "dummy_store.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)

    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    assert ps.basepath == basepath
    # 'keys' is empty as 'field_sep' in example directories is '.',
    # while default 'field_sep' is '-'.
    assert len(ps) == 0

    # Re-do with 'field_sep' set to '.' to comply with example directories.
    @toplevel(field_sep=".")
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    ps = Store(basepath, WeatherEntry)
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


def test_exception_store_key_not_correct_indexer(tmp_path):
    # Test with class not being 'toplevel'.
    class WeatherEntry:
        capital: str
        quantity: str
        spacetime: SpaceTime

    basepath = tmp_path / "store"
    with pytest.raises(TypeError, match="^WeatherEntry"):
        Store(basepath, WeatherEntry)


@toplevel
class WeatherEntry:
    capital: str
    quantity: str
    spacetime: SpaceTime


def test_store_write_getitem(tmp_path):
    # Initialize a parquet dataset.
    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 10:00", freq="2h"),
            "temperature": [8.4, 5.3],
        },
    )
    ps[we].write(ordered_on="timestamp", df=df)
    assert ps._needs_keys_refresh
    assert we in ps
    assert not ps._needs_keys_refresh
    res = ps[we].to_pandas()
    assert res.equals(df)


def test_store_write_getitem_with_config(tmp_path):
    # Initialize a parquet dataset with config.
    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    rg_size = 2
    ps[we].write(ordered_on="timestamp", df=df, row_group_target_size=rg_size)
    assert we in ps
    # Load only first row group.
    res = ps[we][0].to_pandas()
    assert res.equals(df.loc[: rg_size - 1])


def test_store_iterator(tmp_path):
    # Test `__iter__`.
    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    we2 = WeatherEntry("london", "temperature", SpaceTime("greenwich", "winter"))
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    ps[we1].write(ordered_on="timestamp", df=df)
    ps[we2].write(ordered_on="timestamp", df=df)
    for key in ps:
        assert key in (we1, we2)


def test_store_pandas_write_read_roundtrip(tmp_path):
    # Set and get data, roundtrip.
    ps = Store(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 2.9, 6.4],
        },
    )
    ps[we].write(ordered_on="timestamp", df=df, row_group_target_size=2)
    df_res = ps[we].to_pandas()
    assert df_res.equals(df)


def test_store_write_column_multi_index(tmp_path):
    # Write column multi-index in pandas, retrieve in vaex.
    pdf = DataFrame(
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
    ps = Store(tmp_path, WeatherEntry)
    we = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    ps[we].write(ordered_on=("ts", ""), df=pdf, row_group_target_size=2)
    df_res = ps[we].to_pandas()
    df_res.columns = pdf.columns
    assert df_res.equals(pdf)


def test_store_delitem(tmp_path):
    # Test `__delitem__`.
    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "winter"))
    we2 = WeatherEntry("paris", "temperature", SpaceTime("notredame", "summer"))
    we3 = WeatherEntry("london", "temperature", SpaceTime("greenwich", "winter"))
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8.4, 5.3, 4.9, 2.3],
        },
    )
    ps[we1].write(ordered_on="timestamp", df=df)
    ps[we2].write(ordered_on="timestamp", df=df)
    ps[we3].write(ordered_on="timestamp", df=df)
    # Delete london-related data.
    we3_path = basepath / we3.to_path()
    assert os_path.exists(we3_path)
    assert os_path.exists(get_md_filepath(we3_path))
    assert len(ps) == 3
    del ps[we3]
    assert not os_path.exists(we3_path)
    assert not os_path.exists(get_md_filepath(we3_path))
    assert len(ps) == 2
    # Delete paris-summer-related data.
    we2_path = basepath / we2.to_path()
    assert os_path.exists(we2_path)
    assert os_path.exists(get_md_filepath(we2_path))
    del ps[we2]
    assert not os_path.exists(we2_path)
    assert not os_path.exists(get_md_filepath(we2_path))
    assert len(ps) == 1
    # Check paris-winter-related data still exists.
    we1_path = basepath / we1.to_path()
    assert os_path.exists(we1_path)
    assert os_path.exists(get_md_filepath(we1_path))


def test_store_iter_intersections(tmp_path):
    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str

    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature")
    we2 = WeatherEntry("london", "temperature")
    df1 = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8, 5, 4, 2],
        },
    )
    ps[we1].write(ordered_on="timestamp", df=df1, row_group_target_size=2)
    df2 = DataFrame(
        {
            "timestamp": date_range("2021/01/01 09:00", "2021/01/01 15:00", freq="2h"),
            "temperature": [18, 15, 14, 12],
        },
    )
    ps[we2].write(ordered_on="timestamp", df=df2, row_group_target_size=2)
    res = list(ps.iter_intersections(keys=[we1, we2]))
    assert len(res) == 3
    assert res[0][we1].equals(df1.iloc[:2].reset_index(drop=True))
    assert res[1][we1].equals(df1.iloc[2:3].reset_index(drop=True))
    assert res[2][we1].equals(df1.iloc[3:].reset_index(drop=True))
    assert res[0][we2].equals(df2.iloc[:2].reset_index(drop=True))
    assert res[1][we2].equals(df2.iloc[2:2].reset_index(drop=True))
    assert res[2][we2].equals(df2.iloc[2:].reset_index(drop=True))


def test_exception_store_delitem(tmp_path):
    """
    Test that Store raises KeyError for non-existent keys in get, __getitem__, and
    __delitem__.
    """

    @toplevel
    class Indexer:
        category: str
        item: str

    store = Store(tmp_path, Indexer)
    # Create a valid key and add some data
    valid_key = Indexer(category="data", item="dataset1")
    df = DataFrame({"timestamp": [1, 2, 3], "value": [10, 20, 30]})
    store[valid_key].write(df=df, ordered_on="timestamp")
    # Create a new key not in store.
    invalid_key = Indexer(category="missing", item="notfound")
    # Test that __delitem__ raises KeyError.
    with pytest.raises(KeyError, match="not found"):
        del store[invalid_key]
    # Verify valid key still works.
    assert valid_key in store
    assert len(store[valid_key].row_group_stats) > 0
    # Test after deletion, accessing should raise KeyError.
    del store[valid_key]
    with pytest.raises(KeyError, match="not found"):
        del store[valid_key]


def test_store_iter_intersections_exception_handling_releases_locks(tmp_path):
    """
    Test that locks are released when exception occurs during Store.iter_intersections.
    """

    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str

    store = Store(tmp_path, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature")
    we2 = WeatherEntry("london", "temperature")
    # Setup data
    df1 = DataFrame(
        {
            "timestamp1": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8, 5, 4, 2],
        },
    )
    df2 = df1.rename(columns={"timestamp1": "timestamp2"})
    store[we1].write(ordered_on="timestamp1", df=df1)
    store[we2].write(ordered_on="timestamp2", df=df2)
    # Verify exception is raised and locks are still released
    with pytest.raises(ValueError, match="inconsistent 'ordered_on' columns"):
        list(store.iter_intersections([we1, we2]))
    # After exception, locks should be released - verify we can access datasets
    assert store.get(we1, lock_timeout=1) is not None
    assert store.get(we2, lock_timeout=1) is not None


def test_store_iter_intersections_concurrent_access_blocking(tmp_path):
    """
    Test that Store.iter_intersections blocks concurrent access to same datasets.
    """

    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str

    store = Store(tmp_path, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature")
    we2 = WeatherEntry("london", "temperature")
    # Setup data
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8, 5, 4, 2],
        },
    )
    store[we1].write(ordered_on="timestamp", df=df)
    store[we2].write(ordered_on="timestamp", df=df)

    # Start iteration (this will acquire locks via cached_datasets)
    iter_obj = store.iter_intersections([we1, we2])
    next(iter_obj)  # Locks are now held
    # Try concurrent access - should fail
    with pytest.raises(TimeoutError):
        store.get(we1, lock_timeout=1)

    # Complete iteration by exhausting iterator to release locks.
    list(iter_obj)
    # Now access should work.
    assert store.get(we1, lock_timeout=1) is not None
