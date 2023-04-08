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
from pandas import date_range

from oups.cumsegagg import DTYPE_INT64
from oups.segmentby import KEY_LAST_BIN_LABEL
from oups.segmentby import KEY_LAST_ON_VALUE
from oups.segmentby import KEY_RESTART_KEY
from oups.segmentby import LEFT
from oups.segmentby import RIGHT
from oups.segmentby import by_scale
from oups.segmentby import by_x_rows


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "by, closed, end_indices, chunk_labels_refs, next_chunk_starts_refs, n_null_chunks_refs, chunk_ends_refs, buffer_refs",
    [
        (  # 0
            # Check with a 3rd chunk starting with several empty bins.
            Grouper(freq="5T", label="left", closed="left"),
            None,
            [3, 6, 9],
            [
                date_range(start="2020-01-01 08:00:00", end="2020-01-01 08:10:00", freq="5T"),
                date_range(start="2020-01-01 08:10:00", end="2020-01-01 08:20:00", freq="5T"),
                date_range(start="2020-01-01 08:20:00", end="2020-01-01 08:50:00", freq="5T"),
            ],
            [
                array([2, 2, 3], dtype=DTYPE_INT64),
                array([0, 2, 3], dtype=DTYPE_INT64),
                array([0, 0, 0, 0, 2, 2, 3], dtype=DTYPE_INT64),
            ],
            [1, 1, 5],
            [
                date_range(start="2020-01-01 08:05:00", end="2020-01-01 08:15:00", freq="5T"),
                date_range(start="2020-01-01 08:15:00", end="2020-01-01 08:25:00", freq="5T"),
                date_range(start="2020-01-01 08:25:00", end="2020-01-01 08:55:00", freq="5T"),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:20:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00")},
            ],
        ),
        (  # 1
            # Check specific restart key when chunk has a single incomplete bin.
            Grouper(freq="5T", label="left", closed="left"),
            None,
            [2, 7, 9],
            [
                DatetimeIndex(
                    ["2020-01-01 08:00:00"],
                    freq="5T",
                ),
                date_range(start="2020-01-01 08:00:00", end="2020-01-01 08:40:00", freq="5T"),
                date_range(start="2020-01-01 08:40:00", end="2020-01-01 08:50:00", freq="5T"),
            ],
            [
                array([2], dtype=DTYPE_INT64),
                array([0, 0, 1, 3, 4, 4, 4, 4, 5], dtype=DTYPE_INT64),
                array([1, 1, 2], dtype=DTYPE_INT64),
            ],
            [0, 5, 1],
            [
                DatetimeIndex(
                    ["2020-01-01 08:05:00"],
                    freq="5T",
                ),
                date_range(start="2020-01-01 08:05:00", end="2020-01-01 08:45:00", freq="5T"),
                date_range(start="2020-01-01 08:45:00", end="2020-01-01 08:55:00", freq="5T"),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:00:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:40:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00")},
            ],
        ),
        (  # 2
            # Check 'closed' parameter overrides 'by_closed' parameter.
            # Check specific restart key when chunk has a single incomplete bin.
            Grouper(freq="5T", label="left", closed="left"),
            RIGHT,
            [1, 7, 9],
            [
                DatetimeIndex(
                    ["2020-01-01 07:55:00"],
                    freq="5T",
                ),
                date_range(start="2020-01-01 07:55:00", end="2020-01-01 08:35:00", freq="5T"),
                date_range(start="2020-01-01 08:35:00", end="2020-01-01 08:45:00", freq="5T"),
            ],
            [
                array([1], dtype=DTYPE_INT64),
                array([0, 1, 1, 3, 4, 5, 5, 5, 6], dtype=DTYPE_INT64),
                array([0, 1, 2], dtype=DTYPE_INT64),
            ],
            [0, 4, 1],
            [
                DatetimeIndex(
                    ["2020-01-01 08:00:00"],
                    freq="5T",
                ),
                date_range(start="2020-01-01 08:00:00", end="2020-01-01 08:40:00", freq="5T"),
                date_range(start="2020-01-01 08:40:00", end="2020-01-01 08:50:00", freq="5T"),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:00:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:40:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00")},
            ],
        ),
        (  # 3
            # Check with a 3rd chunk starting with several empty bins.
            Grouper(freq="5T", label="left", closed="right"),
            None,
            [3, 6, 9],
            [
                date_range(start="2020-01-01 07:55:00", end="2020-01-01 08:10:00", freq="5T"),
                date_range(start="2020-01-01 08:10:00", end="2020-01-01 08:20:00", freq="5T"),
                date_range(start="2020-01-01 08:20:00", end="2020-01-01 08:45:00", freq="5T"),
            ],
            [
                array([1, 2, 2, 3], dtype=DTYPE_INT64),
                array([1, 2, 3], dtype=DTYPE_INT64),
                array([0, 0, 0, 1, 2, 3], dtype=DTYPE_INT64),
            ],
            [1, 0, 3],
            [
                date_range(start="2020-01-01 08:00:00", end="2020-01-01 08:15:00", freq="5T"),
                date_range(start="2020-01-01 08:15:00", end="2020-01-01 08:25:00", freq="5T"),
                date_range(start="2020-01-01 08:25:00", end="2020-01-01 08:50:00", freq="5T"),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:15:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:25:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00")},
            ],
        ),
        (  # 4
            # Check with a Series.
            # Specific case, 1st & 2nd Series end before end of data.
            # and 2nd & 3rd Series restart after last data from previous iteration.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            # 'left' means end is excluded.
            LEFT,
            [3, 6, 9],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            [
                array([0, 2], dtype=DTYPE_INT64),
                array([0, 2], dtype=DTYPE_INT64),
                array([0, 0, 2], dtype=DTYPE_INT64),
            ],
            [1, 1, 2],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            [
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:06:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:12:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:18:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:21:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:50:00"),
                },
            ],
        ),
        (  # 5
            # Check with a Series.
            # Series end after data, and even after start of data at next iter.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            # 'left' means end is excluded.
            LEFT,
            [3, 6, 9],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            [
                array([0, 3], dtype=DTYPE_INT64),
                array([1, 3], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [1, 0, 0],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:16:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:42:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:52:00")},
            ],
        ),
        (  # 6
            # Check with a Series, ends are included by use of 'right'.
            # Specific case, 1st & 2nd Series end before end of data.
            # and 2nd & 3rd Series restart after last data from previous
            # iteration.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            # 'right' means end is included.
            RIGHT,
            [3, 6, 9],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            [
                array([1, 2], dtype=DTYPE_INT64),
                array([0, 2], dtype=DTYPE_INT64),
                array([0, 0, 3], dtype=DTYPE_INT64),
            ],
            [0, 1, 2],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:18")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:18"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            [
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:06:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:12:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:18:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:21:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:50:00"),
                },
            ],
        ),
        (  # 7
            # Check with a Series, ends are included by use of 'right'.
            # Series end after data, and even after start of data at next iter.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            # 'right' means end is excluded.
            RIGHT,
            [3, 6, 9],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            [
                array([1, 3], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [0, 0, 0],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:16")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:16"), pTimestamp("2020/01/01 08:42")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:42"), pTimestamp("2020/01/01 08:52")]),
            ],
            [
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:16:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:42:00")},
                {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:52:00")},
            ],
        ),
        (  # 8
            # Check with a Series, with a single point per chunk.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:03"),
                        pTimestamp("2020/01/01 08:36"),
                    ]
                ),
            ],
            # 'left' means end is excluded.
            LEFT,
            [3, 6, 9],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03"), pTimestamp("2020/01/01 08:36")]),
            ],
            [
                array([1], dtype=DTYPE_INT64),
                array([0], dtype=DTYPE_INT64),
                array([0, 0], dtype=DTYPE_INT64),
            ],
            [0, 1, 2],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03"), pTimestamp("2020/01/01 08:36")]),
            ],
            [
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:03:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:12:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:03:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:21:00"),
                },
                {
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:36:00"),
                    KEY_LAST_ON_VALUE: pTimestamp("2020-01-01 08:50:00"),
                },
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
    if isinstance(by, Grouper):
        if closed is None:
            closed = by.closed
        by = [by] * 3
    for i, end_idx in enumerate(end_indices):
        (
            next_chunk_starts,
            chunk_labels,
            n_null_chunks,
            by_closed,
            chunk_ends,
            unknown_chunk_end,
        ) = by_scale(on[start_idx:end_idx], by[i], closed=closed, buffer=buffer)
        assert nall(chunk_labels == chunk_labels_refs[i])
        assert nall(chunk_ends == chunk_ends_refs[i])
        assert nall(next_chunk_starts == next_chunk_starts_refs[i])
        assert n_null_chunks == n_null_chunks_refs[i]
        # 'freq' attribute is gonna be deprecated for pandas Timestamp,
        # but is present in resulting Timestamp in 'buffer'. Hence comparing
        # numpy Timestamp instead.
        assert buffer[KEY_RESTART_KEY].to_numpy() == buffer_refs[i][KEY_RESTART_KEY].to_numpy()
        start_idx = end_idx
    assert not unknown_chunk_end
    assert by_closed == closed


@pytest.mark.parametrize(
    "by, closed, end_indices, bin_labels_refs, next_bin_starts_refs, bin_ends_refs, unknown_bin_end_refs, buffer_refs",
    [
        (
            4,
            LEFT,
            [3, 6, 9],
            [
                DatetimeIndex(["2020-01-01 08:00:00"]),
                DatetimeIndex(["2020-01-01 08:00:00", "2020-01-01 08:16:00"]),
                DatetimeIndex(["2020-01-01 08:16:00", "2020-01-01 08:50:00"]),
            ],
            [
                array([3], dtype=DTYPE_INT64),
                array([1, 3], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [
                DatetimeIndex(["2020-01-01 08:12"]),
                DatetimeIndex(["2020-01-01 08:16", "2020-01-01 08:21:00"]),
                DatetimeIndex(["2020-01-01 08:50:00", "2020-01-01 08:50:00"]),
            ],
            [True, True, True],
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
    bin_labels_refs,
    next_bin_starts_refs,
    bin_ends_refs,
    unknown_bin_end_refs,
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
            next_bin_starts,
            bin_labels,
            n_null_bins,
            by_closed,
            bin_ends,
            unknown_bin_end,
        ) = by_x_rows(on[start_idx:end_idx], by, buffer=buffer)
        assert nall(bin_labels == bin_labels_refs[i])
        assert nall(bin_ends == bin_ends_refs[i])
        assert nall(next_bin_starts == next_bin_starts_refs[i])
        assert not n_null_bins
        assert unknown_bin_end == unknown_bin_end_refs[i]
        assert buffer[KEY_RESTART_KEY] == buffer_refs[i][KEY_RESTART_KEY]
        assert buffer[KEY_LAST_BIN_LABEL] == buffer_refs[i][KEY_LAST_BIN_LABEL]
        start_idx = end_idx
    assert by_closed == closed


# by_scale: test exception if 'by' is a Series:
#             raise ValueError(
#                f"first value expected in 'by' to restart correctly is {buffer[KEY_RESTART_KEY]}"
# by_scale: test this exception
#                 raise ValueError(
#                    f"2nd chunk end in 'by' has to be larger than value {buffer[KEY_LAST_ON_VALUE]} to restart correctly."
#                )
# by_x_rows: restart with end_indices falling exactly on bin ends
# in by_x_rows, test 'closed'
# Test with 'by_x_rows' ending exactly on the right number of rows to check it is ok
# Test case when there is a single point (only if using by as a Series - with Grouper, there is necessarily two point: start and ends.)
# Move restart test case for 'by_x_rows' into this file
# check in segmentby(): check no empty bins after running biny from user
# raise error if it is detected.
