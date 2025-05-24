#!/usr/bin/env python3
"""
Created on Sat May 24 18:35:00 2025.

@author: yoh

"""
from pandas import Timedelta
from pandas import Timestamp
from pandas import date_range


def floor_ts(ts: Timestamp, freq: str) -> Timestamp:
    """
    Floor a timestamp even if using non-fixed frequency.

    Parameters
    ----------
    ts : Timestamp
        Timestamp to floor.
    freq : str
        Frequency string.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """
    try:
        return ts.floor(freq)
    except ValueError:
        return date_range(end=ts.normalize(), periods=1, freq=freq)[0]


def ceil_ts(ts: Timestamp, freq: str) -> Timestamp:
    """
    Ceil a timestamp even if using non-fixed frequency.

    Parameters
    ----------
    ts : Timestamp
        Timestamp to ceil.
    freq : str
        Frequency string.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    """
    try:
        # Can't use 'ceil' from pandas because it does not work on 'D'aily
        # frequency.
        # return ts.ceil(freq)
        return (ts + Timedelta(1, unit=freq)).floor(freq)
    except ValueError:
        return date_range(start=floor_ts(ts, freq), periods=2, freq=freq)[1]
