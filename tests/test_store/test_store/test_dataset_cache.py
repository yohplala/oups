"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""

import pytest
from pandas import DataFrame
from pandas import date_range

from oups import Store
from oups import toplevel
from oups.store.ordered_parquet_dataset.ordered_parquet_dataset import OrderedParquetDataset
from oups.store.store.dataset_cache import cached_datasets


def test_cached_datasets_releases_locks_properly(tmp_path):
    """
    Test that cached_datasets context manager releases locks properly.
    """

    @toplevel
    class WeatherEntry:
        capital: str
        quantity: str

    basepath = tmp_path / "store"
    ps = Store(basepath, WeatherEntry)
    we1 = WeatherEntry("paris", "temperature")
    we2 = WeatherEntry("london", "temperature")
    df = DataFrame(
        {
            "timestamp": date_range("2021/01/01 08:00", "2021/01/01 14:00", freq="2h"),
            "temperature": [8, 5, 4, 2],
        },
    )
    ps[we1].write(ordered_on="timestamp", df=df, row_group_target_size=2)
    ps[we2].write(ordered_on="timestamp", df=df, row_group_target_size=2)
    # Use cached_datasets and verify locks are released after
    with cached_datasets(ps, [we1, we2]):
        # Verify datasets are locked during context
        with pytest.raises(TimeoutError):
            OrderedParquetDataset(
                ps.basepath / we1.to_path,
                lock_timeout=1,
            )

    # After context exit, locks should be released
    opd1 = OrderedParquetDataset(ps.basepath / we1.to_path, lock_timeout=1)
    opd2 = OrderedParquetDataset(ps.basepath / we2.to_path, lock_timeout=1)
    assert isinstance(opd1, OrderedParquetDataset)
    assert isinstance(opd2, OrderedParquetDataset)
