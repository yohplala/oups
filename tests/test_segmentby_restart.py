#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import all as nall
from numpy import arange
from numpy import array
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import date_range

from oups.segmentby import DTYPE_DATETIME64
from oups.segmentby import DTYPE_INT64
from oups.segmentby import KEY_BIN
from oups.segmentby import KEY_LAST_BIN_LABEL
from oups.segmentby import KEY_LAST_ON_VALUE
from oups.segmentby import KEY_RESTART_KEY
from oups.segmentby import KEY_SNAP
from oups.segmentby import LEFT
from oups.segmentby import NULL_INT64_1D_ARRAY
from oups.segmentby import RIGHT
from oups.segmentby import by_scale
from oups.segmentby import by_x_rows
from oups.segmentby import segmentby
from oups.segmentby import setup_segmentby


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
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:12")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:12"),
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
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:12")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:12"),
                        pTimestamp("2020/01/01 08:36"),
                        pTimestamp("2020/01/01 08:50"),
                    ]
                ),
            ],
            [
                array([1, 2], dtype=DTYPE_INT64),
                array([0, 0], dtype=DTYPE_INT64),
                array([0, 0, 3], dtype=DTYPE_INT64),
            ],
            [0, 2, 2],
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:00"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:06"), pTimestamp("2020/01/01 08:12")]),
                DatetimeIndex(
                    [
                        pTimestamp("2020/01/01 08:12"),
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
                    KEY_RESTART_KEY: pTimestamp("2020-01-01 08:12:00"),
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
        assert buffer == buffer_refs[i]
        start_idx = end_idx
    assert not unknown_chunk_end
    assert by_closed == closed


@pytest.mark.parametrize(
    "by, closed, end_indices, exception_mess",
    [
        (  # 0
            # 'by' as a Series, 1st value in 2nd chuunk of 'by' is not
            # the same as last value from previous iteration.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03"), pTimestamp("2020/01/01 08:06")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:07")]),
            ],
            LEFT,
            (3, 6),
            "^first value expected in 'by'",
        ),
        (  # 1
            # 'by' as a Series, 'closed' is 'left'.
            # Last value in 'on' at 1st iter. is after 2nd value of 'by' in 2nd
            # chunk.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03"), pTimestamp("2020/01/01 08:12")]),
            ],
            LEFT,
            (3, 6),
            "^2nd chunk end in 'by'",
        ),
        (  # 2
            # 'by' as a Series, 'closed' is 'right'.
            # Last value in 'on' at 1st iter. is after 2nd value of 'by' in 2nd
            # chunk.
            [
                DatetimeIndex([pTimestamp("2020/01/01 08:03")]),
                DatetimeIndex([pTimestamp("2020/01/01 08:03"), pTimestamp("2020/01/01 08:11")]),
            ],
            RIGHT,
            (3, 6),
            "^2nd chunk end in 'by'",
        ),
    ],
)
def test_by_scale_exceptions(by, closed, end_indices, exception_mess):
    on = Series(
        [
            pTimestamp("2020/01/01 08:00"),  # 0
            pTimestamp("2020/01/01 08:03"),
            pTimestamp("2020/01/01 08:12"),  # 2
            pTimestamp("2020/01/01 08:15"),  # 3
            pTimestamp("2020/01/01 08:16"),
            pTimestamp("2020/01/01 08:21"),  # 5
        ]
    )
    buffer = {}
    end1, end2 = end_indices
    # 1st run, ok
    by_scale(on[0:end1], by[0], closed=closed, buffer=buffer)
    with pytest.raises(ValueError, match=exception_mess):
        by_scale(on[end1:end2], by[1], closed=closed, buffer=buffer)


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
        ),
        (  # 1
            # Bin end falling exactly on chunk end.
            4,
            RIGHT,
            [4, 6, 9],
            [
                DatetimeIndex(["2020-01-01 08:00:00"]),
                DatetimeIndex(["2020-01-01 08:16:00"]),
                DatetimeIndex(["2020-01-01 08:16:00", "2020-01-01 08:50:00"]),
            ],
            [
                array([4], dtype=DTYPE_INT64),
                array([2], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [
                DatetimeIndex(["2020-01-01 08:15:00"]),
                DatetimeIndex(["2020-01-01 08:21:00"]),
                DatetimeIndex(["2020-01-01 08:41:00", "2020-01-01 08:50:00"]),
            ],
            [False, True, True],
            [
                {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:00")},
                {KEY_RESTART_KEY: 2, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:16")},
                {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:50")},
            ],
        ),
        (  # 2
            # Bin end falling exactly on chunk end.
            4,
            LEFT,
            [4, 6, 9],
            [
                DatetimeIndex(["2020-01-01 08:00:00"]),
                DatetimeIndex(["2020-01-01 08:16:00"]),
                DatetimeIndex(["2020-01-01 08:16:00", "2020-01-01 08:50:00"]),
            ],
            [
                array([4], dtype=DTYPE_INT64),
                array([2], dtype=DTYPE_INT64),
                array([2, 3], dtype=DTYPE_INT64),
            ],
            [
                DatetimeIndex(["2020-01-01 08:15:00"]),
                DatetimeIndex(["2020-01-01 08:21:00"]),
                DatetimeIndex(["2020-01-01 08:50:00", "2020-01-01 08:50:00"]),
            ],
            [True, True, True],
            [
                {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:00")},
                {KEY_RESTART_KEY: 2, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:16")},
                {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2020/01/01 08:50")},
            ],
        ),
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
        ) = by_x_rows(on[start_idx:end_idx], by, closed=closed, buffer=buffer)
        assert nall(bin_labels == bin_labels_refs[i])
        assert nall(bin_ends == bin_ends_refs[i])
        assert nall(next_bin_starts == next_bin_starts_refs[i])
        assert not n_null_bins
        assert unknown_bin_end == unknown_bin_end_refs[i]
        assert buffer[KEY_RESTART_KEY] == buffer_refs[i][KEY_RESTART_KEY]
        assert buffer[KEY_LAST_BIN_LABEL] == buffer_refs[i][KEY_LAST_BIN_LABEL]
        start_idx = end_idx
    assert by_closed == closed


@pytest.mark.parametrize(
    "len_data, x_rows, closed, buffer_in, buffer_out, chunk_starts_ref,"
    "next_chunk_starts_ref, unknown_last_bin_end_ref",
    [
        (
            3,
            4,
            LEFT,
            {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            array([0]),
            array([3]),
            True,
        ),
        (
            7,
            4,
            LEFT,
            {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 11:00")},
            array([0, 3]),
            array([3, 7]),
            True,
        ),
        (
            7,
            4,
            LEFT,
            {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            {KEY_RESTART_KEY: 3, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 12:00")},
            array([0, 4]),
            array([4, 7]),
            True,
        ),
        (
            8,
            4,
            LEFT,
            {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 15:00")},
            array([0, 3, 7]),
            array([3, 7, 8]),
            True,
        ),
        (
            8,
            4,
            LEFT,
            {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 07:50")},
            {KEY_RESTART_KEY: 4, KEY_LAST_BIN_LABEL: pTimestamp("2022/01/01 12:00")},
            array([0, 4]),
            array([4, 8]),
            True,
        ),
    ],
)
def test_by_x_rows_single_shots(
    len_data,
    x_rows,
    closed,
    buffer_in,
    buffer_out,
    chunk_starts_ref,
    next_chunk_starts_ref,
    unknown_last_bin_end_ref,
):
    start = pTimestamp("2022/01/01 08:00")
    dummy_data = arange(len_data)
    data = pDataFrame(
        {"dummy_data": dummy_data, "dti": date_range(start, periods=len_data, freq="1H")}
    )
    chunk_labels_ref = data.iloc[chunk_starts_ref, -1].reset_index(drop=True)
    chunk_ends_idx = next_chunk_starts_ref.copy()
    if closed == LEFT:
        chunk_ends_idx[-1] = len_data - 1
        chunk_ends_ref = data.iloc[chunk_ends_idx, -1].reset_index(drop=True)
    else:
        chunk_ends_idx -= 1
        chunk_ends_ref = data.iloc[chunk_ends_idx, -1].reset_index(drop=True)
    if buffer_in[KEY_RESTART_KEY] != x_rows:
        chunk_labels_ref.iloc[0] = buffer_in[KEY_LAST_BIN_LABEL]
    (
        next_chunk_starts,
        chunk_labels,
        n_null_chunks,
        chunk_closed,
        chunk_ends,
        unknown_last_bin_end,
    ) = by_x_rows(data, x_rows, closed=closed, buffer=buffer_in)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert nall(chunk_labels == chunk_labels_ref)
    assert not n_null_chunks
    assert chunk_closed == closed
    # 'chunk_ends' is expected to be the same than 'chunk_labels'.
    assert nall(chunk_ends == chunk_ends_ref)
    assert buffer_in == buffer_out
    assert unknown_last_bin_end == unknown_last_bin_end_ref


def test_segmentby_exception_trailing_empty_bin():
    bin_on = "dti"
    dti = date_range("2020/01/01 08:04", periods=4, freq="3T")
    data = pDataFrame({bin_on: dti, "ordered_on": range(len(dti))})
    len_data = len(data)

    def by_empty_trailing_bin(on, buffer=None):
        # 'next_chunk_starts' ends with an empty bin.
        return (
            array([1, len_data, len_data]),
            Series(["a", "o", "u"]),
            1,
            LEFT,
            dti[[1, len_data - 1, len_data - 1]],
            False,
        )

    with pytest.raises(ValueError, match="^there is at least one empty trailing bin."):
        segmentby(data=data, bin_by=by_empty_trailing_bin, bin_on=bin_on, buffer={})


@pytest.mark.parametrize(
    "bin_by, bin_on, ordered_on, snap_by, end_indices, "
    "next_chunk_starts_refs, bin_indices_refs, bin_labels_refs, n_null_bins_refs, "
    "snap_labels_refs, n_max_null_snaps_refs, buffer_refs",
    [
        (
            # 0/ 'bin_by' only, as a Callable.
            #  'data'
            #   dti  odr  slices  bins
            #  8:10    0             1
            #  8:10    1
            #  8:12    2
            #  8:17    3       0
            #  8:19    4             2
            #  8:20    5
            #  9:00    6       0
            #  9:10    7
            #  9:30    8             3
            by_x_rows,
            "ordered_on",
            None,
            None,
            [3, 6, 9],
            [[3], [1, 3], [2, 3]],
            [NULL_INT64_1D_ARRAY] * 3,
            [[0], [0, 4], [4, 8]],
            [0] * 3,
            [None] * 3,
            [0] * 3,
            [
                {KEY_RESTART_KEY: 3, KEY_LAST_BIN_LABEL: 0},
                {KEY_RESTART_KEY: 2, KEY_LAST_BIN_LABEL: 4},
                {KEY_RESTART_KEY: 1, KEY_LAST_BIN_LABEL: 8},
            ],
        ),
        (
            # 1/ 'bin_by' and 'snap_by' as Grouper.
            #    'bin_by', 20T, is a multiple of 'snap_by', 10T
            #   dti  odr  slices  bins     snaps
            #                      1-8:00
            #  8:10    0
            #  8:10    1
            #  8:12    2
            #  8:17    3       0
            #  8:19    4
            #  8:20    5           2-8:20   1-8:20 (excl.)
            #                               2-8:30
            #                      3-8:40   3-8:40
            #                               4-8:50
            #  9:00    6       0   4-9:00   5-9:00 (excl.)
            #  9:10    7                    6-9:10 (excl.)
            #                      5-9:20   7-9:20
            #  9:30    8                    8-9:30 (excl.)
            #                               9-9:40
            Grouper(freq="20T", label="left", closed="left", key="dti"),
            None,
            None,
            Grouper(freq="10T", label="right", closed="left", key="dti"),
            [3, 6, 9],
            # next_chunk_starts
            [
                #         b
                array([3, 3]),
                #         b     b
                array([2, 2, 3, 3]),
                #      s  s  b  s  s  b  s  s  b  s  s  b
                array([0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3]),
            ],
            # bin_indices
            [array([1]), array([1, 3]), array([2, 5, 8, 11])],
            # bin_labels
            [
                DatetimeIndex(["2020-01-01 08:00"], freq="20T"),
                DatetimeIndex(["2020-01-01 08:00", "2020-01-01 08:20"], freq="20T"),
                DatetimeIndex(
                    [
                        "2020-01-01 08:20",
                        "2020-01-01 08:40",
                        "2020-01-01 09:00",
                        "2020-01-01 09:20",
                    ],
                    freq="20T",
                ),
            ],
            # n_null_bins
            [0, 0, 2],
            # snap_labels
            [
                DatetimeIndex(["2020-01-01 08:20"], freq="10T"),
                DatetimeIndex(["2020-01-01 08:20", "2020-01-01 08:30"], freq="10T"),
                DatetimeIndex(
                    [
                        "2020-01-01 08:30",
                        "2020-01-01 08:40",
                        "2020-01-01 08:50",
                        "2020-01-01 09:00",
                        "2020-01-01 09:10",
                        "2020-01-01 09:20",
                        "2020-01-01 09:30",
                        "2020-01-01 09:40",
                    ],
                    freq="10T",
                ),
            ],
            # n_max_null_snaps
            [0, 0, 7],
            [
                # First 'restart_key' is 1st value in data in this case because
                # there is a single bin. This applies to snapshot as well.
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10")},
                },
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:20")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:20")},
                },
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 09:20")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 09:30")},
                },
            ],
        ),
        (
            # 2/ 'bin_by' as Grouper, 'snap_by' as Series.
            #    'bin_by', 20T
            #   dti  odr  slices  bins     snaps
            #                      1-8:00
            #  8:10    0
            #  8:10    1
            #  8:12    2                    1-8:12 (excl.)
            #                               2-8:15
            #  8:17    3       0
            #  8:19    4
            #  8:20    5           2-8:20   3-8:20 (excl.)
            #                               4-8:35
            #                      3-8:40   5-8:40
            #
            #  9:00    6       0   4-9:00   6-9:00 (excl.)
            #  9:10    7
            #                               7-9:11 (excl.)
            #                      5-9:20   8-9:20
            #  9:30    8
            #                               9-9:40
            #                              10-9:44 (will be ignored)
            Grouper(freq="20T", label="left", closed="left", key="dti"),
            None,
            "dti",
            [
                DatetimeIndex(["2020-01-01 08:12", "2020-01-01 08:15"]),
                DatetimeIndex(["2020-01-01 08:15", "2020-01-01 08:20", "2020-01-01 08:35"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:35",
                        "2020-01-01 08:40",
                        "2020-01-01 09:00",
                        "2020-01-01 09:11",
                        "2020-01-01 09:20",
                        "2020-01-01 09:40",
                        "2020-01-01 09:44",
                    ]
                ),
            ],
            [3, 6, 9],
            # next_chunk_starts
            [
                #            b
                array([2, 3, 3]),
                #            b     b
                array([0, 2, 2, 3, 3]),
                #      s  s  b  s  b  s  s  b  s  b
                array([0, 0, 0, 0, 0, 2, 2, 2, 3, 3]),
            ],
            # bin_indices
            [array([2]), array([2, 4]), array([2, 4, 7, 9])],
            # bin_labels
            [
                DatetimeIndex(["2020-01-01 08:00"], freq="20T"),
                DatetimeIndex(["2020-01-01 08:00", "2020-01-01 08:20"], freq="20T"),
                DatetimeIndex(
                    [
                        "2020-01-01 08:20",
                        "2020-01-01 08:40",
                        "2020-01-01 09:00",
                        "2020-01-01 09:20",
                    ],
                    freq="20T",
                ),
            ],
            # n_null_bins
            [0, 0, 2],
            # snap_labels
            [
                DatetimeIndex(["2020-01-01 08:12", "2020-01-01 08:15"]),
                DatetimeIndex(["2020-01-01 08:15", "2020-01-01 08:20", "2020-01-01 08:35"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:35",
                        "2020-01-01 08:40",
                        "2020-01-01 09:00",
                        "2020-01-01 09:11",
                        "2020-01-01 09:20",
                        "2020-01-01 09:40",
                    ]
                ),
            ],
            # n_max_null_snaps
            [0, 1, 5],
            [
                # First 'restart_key' is 1st value in data in this case because
                # there is a single bin.
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:15")},
                },
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:20")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:35")},
                },
                {
                    KEY_BIN: {KEY_RESTART_KEY: pTimestamp("2020-01-01 09:20")},
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 09:40")},
                },
            ],
        ),
        (
            # 3/ 'bin_by' using 'by_x_rows', 'snap_by' as '15T' Grouper.
            #   dti  odr  slices  bins     snaps
            #  8:10    0           1-8:10
            #  8:10    1
            #  8:12    2
            #                               1-8:15
            #  8:17    3       0
            #  8:19    4           2-8:19
            #  8:20    5
            #                               2-8:30
            #                               3-8:45
            #                               4-9:00
            #  9:00    6       0
            #  9:10    7
            #                               5-9:15
            #                               6-9:30
            #  9:30    8           3-9:30
            #                               7-9:45
            by_x_rows,
            None,
            "dti",
            Grouper(freq="15T", label="right", closed="left", key="dti"),
            [3, 6, 9],
            # next_chunk_starts
            [
                #         b
                array([3, 3]),
                #         b     b
                array([0, 1, 3, 3]),
                #      s  s  s  s  s  b  s  b
                array([0, 0, 0, 2, 2, 2, 3, 3]),
            ],
            # bin_indices
            [array([1]), array([1, 3]), array([5, 7])],
            # bin_labels
            [
                DatetimeIndex(["2020-01-01 08:10"]),
                DatetimeIndex(["2020-01-01 08:10", "2020-01-01 08:19"]),
                DatetimeIndex(
                    ["2020-01-01 08:19", "2020-01-01 09:30"],
                ),
            ],
            # n_null_bins
            [0, 0, 0],
            # snap_labels
            [
                DatetimeIndex(["2020-01-01 08:15"]),
                DatetimeIndex(["2020-01-01 08:15", "2020-01-01 08:30"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:30",
                        "2020-01-01 08:45",
                        "2020-01-01 09:00",
                        "2020-01-01 09:15",
                        "2020-01-01 09:30",
                        "2020-01-01 09:45",
                    ]
                ),
            ],
            # n_max_null_snaps
            [0, 1, 4],
            [
                # First 'restart_key' is 1st value in data in this case because
                # there is a single snap.
                {
                    KEY_BIN: {
                        KEY_RESTART_KEY: 3,
                        KEY_LAST_BIN_LABEL: pTimestamp("2020-01-01 08:10"),
                    },
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:10")},
                },
                {
                    KEY_BIN: {
                        KEY_RESTART_KEY: 2,
                        KEY_LAST_BIN_LABEL: pTimestamp("2020-01-01 08:19"),
                    },
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 08:15")},
                },
                {
                    KEY_BIN: {
                        KEY_RESTART_KEY: 1,
                        KEY_LAST_BIN_LABEL: pTimestamp("2020-01-01 09:30"),
                    },
                    KEY_SNAP: {KEY_RESTART_KEY: pTimestamp("2020-01-01 09:30")},
                },
            ],
        ),
    ],
)
def test_segmentby(
    bin_by,
    bin_on,
    ordered_on,
    snap_by,
    end_indices,
    next_chunk_starts_refs,
    bin_indices_refs,
    bin_labels_refs,
    n_null_bins_refs,
    snap_labels_refs,
    n_max_null_snaps_refs,
    buffer_refs,
):
    dti = array(
        [
            "2020-01-01T08:10",
            "2020-01-01T08:10",
            "2020-01-01T08:12",
            "2020-01-01T08:17",
            "2020-01-01T08:19",
            "2020-01-01T08:20",
            "2020-01-01T09:00",
            "2020-01-01T09:10",
            "2020-01-01T09:30",
        ],
        dtype=DTYPE_DATETIME64,
    )
    data = pDataFrame({"dti": dti, "ordered_on": range(len(dti))})
    start_idx = 0
    buffer = {}
    bin_by = setup_segmentby(bin_by, bin_on, ordered_on, snap_by)
    for i, end_idx in enumerate(end_indices):
        (
            next_chunk_starts,
            bin_indices,
            bin_labels,
            n_null_bins,
            snap_labels,
            n_max_null_snaps,
        ) = segmentby(
            data[start_idx:end_idx],
            bin_by,
            snap_by=snap_by[i] if isinstance(snap_by, list) else snap_by,
            buffer=buffer,
        )
        assert nall(next_chunk_starts == next_chunk_starts_refs[i])
        assert nall(bin_indices == bin_indices_refs[i])
        assert nall(bin_labels == bin_labels_refs[i])
        assert n_null_bins == n_null_bins_refs[i]
        assert nall(snap_labels == snap_labels_refs[i])
        assert n_max_null_snaps == n_max_null_snaps_refs[i]
        assert buffer == buffer_refs[i]
        start_idx = end_idx


# /!\ WiP start
# test restart with bin AND snap to make sure it makes sense.
# test cases restart with both cases: one with unknown_bin_end True, another with False, then restart.
# /!\ WiP end
