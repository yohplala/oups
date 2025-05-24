#!/usr/bin/env python3
"""
Created on Sat May 24 18:35:00 2025.

@author: yoh

"""
import pytest
from pandas import Timestamp

from oups.date_utils import ceil_ts
from oups.date_utils import floor_ts


@pytest.mark.parametrize(
    "test_id, ts, freq, expected_floor, expected_ceil",
    [
        (
            "month_start_exact",
            Timestamp("2024-03-01"),  # month start already
            "MS",  # month start
            Timestamp("2024-03-01 00:00:00"),  # floor to month start
            Timestamp("2024-04-01 00:00:00"),  # ceil to next month start
        ),
        (
            "month_end_exact",
            Timestamp("2024-03-31"),  # month end already
            "MS",  # month start
            Timestamp("2024-03-01 00:00:00"),  # floor to month start
            Timestamp("2024-04-01 00:00:00"),  # ceil to next month start
        ),
        (
            "month_start",
            Timestamp("2024-03-15 14:30:00"),  # mid-month timestamp
            "MS",  # month start
            Timestamp("2024-03-01 00:00:00"),  # floor to month start
            Timestamp("2024-04-01 00:00:00"),  # ceil to next month start
        ),
        (
            "month_end",
            Timestamp("2024-03-15 14:30:00"),  # mid-month timestamp
            "ME",  # month end
            Timestamp("2024-02-29 00:00:00"),  # floor to month end
            Timestamp("2024-03-31 00:00:00"),  # ceil to next month end
        ),
        (
            "semi_month_start",
            Timestamp("2024-03-17 14:30:00"),  # mid-month timestamp
            "SMS",  # semi-month start (1st and 15th)
            Timestamp("2024-03-15 00:00:00"),  # floor to 15th
            Timestamp("2024-04-01 00:00:00"),  # ceil to next month start
        ),
        (
            "semi_month_start_early",
            Timestamp("2024-03-07 14:30:00"),  # early month timestamp
            "SMS",  # semi-month start (1st and 15th)
            Timestamp("2024-03-01 00:00:00"),  # floor to month start
            Timestamp("2024-03-15 00:00:00"),  # ceil to 15th
        ),
        (
            "semi_month_end",
            Timestamp("2024-03-17 14:30:00"),  # mid-month timestamp
            "SME",  # semi-month end (15th and last day)
            Timestamp("2024-03-15 00:00:00"),  # floor to 15th
            Timestamp("2024-03-31 00:00:00"),  # ceil to month end
        ),
        (
            "semi_month_end_early",
            Timestamp("2024-03-07 14:30:00"),  # early month timestamp
            "SME",  # semi-month end (14th and last day)
            Timestamp("2024-02-29 00:00:00"),  # floor to previous month end
            Timestamp("2024-03-15 00:00:00"),  # ceil to 15th
        ),
        (
            "hourly",
            Timestamp("2024-03-15 14:37:23"),  # timestamp with minutes and seconds
            "h",  # hourly
            Timestamp("2024-03-15 14:00:00"),  # floor to hour
            Timestamp("2024-03-15 15:00:00"),  # ceil to next hour
        ),
        (
            "hourly_exact",
            Timestamp("2024-03-15 01:00:00"),  # timestamp with minutes and seconds
            "h",  # hourly
            Timestamp("2024-03-15 01:00:00"),  # floor to hour
            Timestamp("2024-03-15 02:00:00"),  # ceil to next hour
        ),
        (
            "daily",
            Timestamp("2024-03-15 14:37:23"),  # timestamp with time
            "D",  # daily
            Timestamp("2024-03-15 00:00:00"),  # floor to day
            Timestamp("2024-03-16 00:00:00"),  # ceil to next day
        ),
        (
            "daily_exact",
            Timestamp("2024-03-15 00:00:00"),  # timestamp with time
            "D",  # daily
            Timestamp("2024-03-15 00:00:00"),  # floor to day
            Timestamp("2024-03-16 00:00:00"),  # ceil to next day
        ),
    ],
)
def test_floor_ceil_timestamp(
    test_id: str,
    ts: Timestamp,
    freq: str,
    expected_floor: Timestamp,
    expected_ceil: Timestamp,
) -> None:
    """
    Test floor and ceil functions with various frequency strings.

    Frequency aliases:
    https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    ts : Timestamp
        Input timestamp to floor/ceil.
    freq : str
        Frequency string ('MS', 'ME', 'SMS', 'SME', 'H', 'D').
    expected_floor : Timestamp
        Expected result of floor operation.
    expected_ceil : Timestamp
        Expected result of ceil operation.

    """
    assert floor_ts(ts, freq) == expected_floor
    assert ceil_ts(ts, freq) == expected_ceil


def test_pandas_bug_floor_ceil_with_non_fixed_frequency():
    # If fixed in pandas, correct in 'oups.store.utils.ceil_ts()' and
    # 'oups.store.utils.floor_ts()'.
    # 'floor' & 'ceil' do not accept non-fixed freqstr.
    ts = Timestamp("2024-03-15 00:00:00")
    with pytest.raises(ValueError, match=".*is a non-fixed frequency$"):
        ts.floor("MS")
    with pytest.raises(ValueError, match=".*is a non-fixed frequency$"):
        ts.ceil("MS")


def test_pandas_bug_timestamp_on_offset_and_ceil():
    # If fixed in pandas, correct in 'oups.store.utils.ceil_ts()'
    # 'ceil' in pandas and a Timestamp on offset returns its floor,
    # not its ceil.
    ts = Timestamp("2024-03-15 00:00:00")
    # Expected: 2024-03-16 00:00:00
    assert ts.ceil("D") == ts
    # Same with 'h'our...
    assert ts.ceil("h") == Timestamp("2024-03-15 00:00:00")
