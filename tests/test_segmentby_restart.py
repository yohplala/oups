#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import array
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_INT64
from oups.segmentby import KEY_LAST_BIN_LABEL
from oups.segmentby import KEY_RESTART_KEY
from oups.segmentby import LEFT
from oups.segmentby import by_scale
from oups.segmentby import by_x_rows


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "by, closed, end_indices, chunk_labels_refs, next_chunk_starts_refs, n_null_chunks_refs, chunk_ends_refs, buffer_refs",
    [
        (
            Grouper(freq="5T", label="left", closed="left"),
            None,
            [3, 6, 9],
            [
                DatetimeIndex(
                    ["2020-01-01 08:00:00", "2020-01-01 08:05:00", "2020-01-01 08:10:00"],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
                DatetimeIndex(
                    ["2020-01-01 08:10:00", "2020-01-01 08:15:00", "2020-01-01 08:20:00"],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
                DatetimeIndex(
                    [
                        "2020-01-01 08:20:00",
                        "2020-01-01 08:25:00",
                        "2020-01-01 08:30:00",
                        "2020-01-01 08:35:00",
                        "2020-01-01 08:40:00",
                        "2020-01-01 08:45:00",
                        "2020-01-01 08:50:00",
                    ],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
            ],
            [
                array([2, 2, 3], dtype=DTYPE_INT64),
                array([0, 2, 3], dtype=DTYPE_INT64),
                array([0, 0, 0, 0, 2, 2, 3], dtype=DTYPE_INT64),
            ],
            [1, 1, 5],
            [
                DatetimeIndex(
                    ["2020-01-01 08:05:00", "2020-01-01 08:10:00", "2020-01-01 08:15:00"],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
                DatetimeIndex(
                    ["2020-01-01 08:15:00", "2020-01-01 08:20:00", "2020-01-01 08:25:00"],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
                DatetimeIndex(
                    [
                        "2020-01-01 08:25:00",
                        "2020-01-01 08:30:00",
                        "2020-01-01 08:35:00",
                        "2020-01-01 08:40:00",
                        "2020-01-01 08:45:00",
                        "2020-01-01 08:50:00",
                        "2020-01-01 08:55:00",
                    ],
                    dtype=DTYPE_DATETIME64,
                    freq="5T",
                ),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:20:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00")},
            ],
        ),
    ],
)
def test_by_scale(
    by,
    closed,
    end_indices,
    chunk_labels_refs,
    next_chunk_starts_refs,
    n_null_chunks_refs,
    chunk_ends_refs,
    buffer_refs,
):
    on = Series(
        [
            pTimestamp("2020/01/01 08:00"),  # 0
            pTimestamp("2020/01/01 08:03"),
            pTimestamp("2020/01/01 08:12"),  # 2
            pTimestamp("2020/01/01 08:15"),  # 3 # 0
            pTimestamp("2020/01/01 08:16"),
            pTimestamp("2020/01/01 08:21"),  # 5
            pTimestamp("2020/01/01 08:40"),  # 6 # 0
            pTimestamp("2020/01/01 08:41"),
            pTimestamp("2020/01/01 08:50"),  # 8
        ]
    )
    start_idx = 0
    buffer = {}
    for i, end_idx in enumerate(end_indices):
        (
            next_chunk_starts,
            chunk_labels,
            n_null_chunks,
            by_closed,
            chunk_ends,
        ) = by_scale(on[start_idx:end_idx], by, closed=closed, buffer=buffer)
        assert nall(chunk_labels == chunk_labels_refs[i])
        assert nall(chunk_ends == chunk_ends_refs[i])
        assert nall(next_chunk_starts == next_chunk_starts_refs[i])
        assert n_null_chunks == n_null_chunks_refs[i]
        # 'freq' attribute is gonna be deprecated for pandas Timestamp,
        # but is present in resulting Timestamp in 'buffer'. Hence comparing
        # numpy Timestamp instead.
        assert buffer[KEY_RESTART_KEY].to_numpy() == buffer_refs[i][KEY_RESTART_KEY].to_numpy()
        start_idx = end_idx
    assert by_closed == by.closed


@pytest.mark.parametrize(
    "by, closed, end_indices, chunk_labels_refs, next_chunk_starts_refs, chunk_ends_refs, buffer_refs",
    [
        (
            4,
            LEFT,
            [3, 6, 9],
            [
                DatetimeIndex(["2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64),
                DatetimeIndex(
                    ["2020-01-01 08:00:00", "2020-01-01 08:16:00"], dtype=DTYPE_DATETIME64
                ),
                DatetimeIndex(
                    ["2020-01-01 08:16:00", "2020-01-01 08:50:00"], dtype=DTYPE_DATETIME64
                ),
            ],
            [
                array([3], dtype=DTYPE_INT64),
                array([1, 3], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [
                DatetimeIndex(["2020-01-01 08:12"], dtype=DTYPE_DATETIME64),
                DatetimeIndex(["2020-01-01 08:16", "2020-01-01 08:21:00"], dtype=DTYPE_DATETIME64),
                DatetimeIndex(
                    ["2020-01-01 08:50:00", "2020-01-01 08:50:00"], dtype=DTYPE_DATETIME64
                ),
            ],
            [
                {KEY_RESTART_KEY: 3, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:00")},
                {KEY_RESTART_KEY: 2, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:16")},
                {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:50")},
            ],
        )
    ],
)
def test_by_x_rows(
    by,
    closed,
    end_indices,
    chunk_labels_refs,
    next_chunk_starts_refs,
    chunk_ends_refs,
    buffer_refs,
):
    on = Series(
        [
            pTimestamp("2020/01/01 08:00"),  # 0
            pTimestamp("2020/01/01 08:03"),
            pTimestamp("2020/01/01 08:12"),  # 2
            pTimestamp("2020/01/01 08:15"),  # 3 # 0
            pTimestamp("2020/01/01 08:16"),
            pTimestamp("2020/01/01 08:21"),  # 5
            pTimestamp("2020/01/01 08:40"),  # 6 # 0
            pTimestamp("2020/01/01 08:41"),
            pTimestamp("2020/01/01 08:50"),  # 8
        ]
    )
    start_idx = 0
    buffer = {}
    for i, end_idx in enumerate(end_indices):
        (
            next_chunk_starts,
            chunk_labels,
            n_null_chunks,
            by_closed,
            chunk_ends,
        ) = by_x_rows(on[start_idx:end_idx], by, buffer=buffer)
        assert nall(chunk_labels == chunk_labels_refs[i])
        assert nall(chunk_ends == chunk_ends_refs[i])
        assert nall(next_chunk_starts == next_chunk_starts_refs[i])
        assert not n_null_chunks
        assert buffer[KEY_RESTART_KEY] == buffer_refs[i][KEY_RESTART_KEY]
        assert buffer[KEY_LAST_BIN_LABEL] == buffer_refs[i][KEY_LAST_BIN_LABEL]
        start_idx = end_idx
    assert by_closed == closed


# by_scale: with a 1st chunk having a single incomplete bin to check it is correctly managed.
# normally, buffer should not be started?
# by_x_rows: restart with end_indices falling exactly on bin ends
# by_scale: restart with end_indices falling exactly on bin ends
# in by_scale, test 'closed' (important to confirm restart_key is ok, with way is generated 'first' for a date range)
# by_scale: test with 'by' as a Series
# in by_x_rows, test 'closed'
# Test with 'by_x_rows' ending exactly on the right number of rows to check it is ok
# Test exception if 'by' when a Series, does not restart with correct value for safe restart
# Test case when there is a single point (only if using by as a Series - with Grouper, there is necessarily two point: start and ends.)
# Do a restart with several empty chunk
# Do a restart with several values before the end of the 1st chunk
# Test a Grouper, closed right/left, how key_last_key is taken into account?
# Move restart test case for 'by_x_rows' into this file

# check in segmentby(): check no empty bins after running biny from user
# raise error if it is detected.
