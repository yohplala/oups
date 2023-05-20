#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from functools import partial

import pytest
from numpy import array
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_FLOAT64
from oups.cumsegagg import DTYPE_INT64
from oups.cumsegagg import DTYPE_NULLABLE_INT64
from oups.cumsegagg import KEY_LAST_CHUNK_RES
from oups.cumsegagg import cumsegagg
from oups.cumsegagg import setup_cumsegagg
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import LAST
from oups.jcumsegagg import MIN
from oups.jcumsegagg import SUM
from oups.segmentby import by_x_rows
from oups.segmentby import setup_segmentby


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "end_indices, bin_by, ordered_on, last_chunk_res_ref, " "indices_of_null_res, bin_res_ref",
    [
        # 0/ 15mn bin left closed; right label
        # 2nd iter. starting on empty bin b3.
        # This validates removale of '-1' values in 'null_bin_indices' (in
        # 'cumsegagg()' function).
        #  'data'
        #  datetime value  qty  bins     row_idx (group)
        #      8:10   4.0    4  b1-8:15  0 (0)
        #      8:10   4.2    3  b1
        #      8:12   3.9    1  b1
        #      8:17   5.6    7  b2-8:30
        #      8:19   6.0    2  b2
        #      8:20   9.8    6  b2       5
        #                       b3-8:45  6 (1)
        #      9:00   4.5    2  b4-9:00
        #      9:10   1.1    8  b4
        #                       b5-9:15
        #      9:30   3.2    1  b6-9:30
        (
            [6, 11],
            Grouper(freq="15T", key="dti", closed="left", label="right"),
            None,
            [pDataFrame({FIRST: [5.6], SUM: [15]}), pDataFrame({FIRST: [3.2], SUM: [1]})],
            [
                pTimestamp("2020-01-01 08:45:00"),
                pTimestamp("2020-01-01 09:00:00"),
                pTimestamp("2020-01-01 09:30:00"),
            ],
            None,
        ),
        # 1/ 15mn bin left closed; right label
        #  'data'
        #  datetime value  qty  bins     row_idx (group)
        #      8:10   4.0    4  b1-8:15  0 (0)
        #      8:10   4.2    3  b1
        #      8:12   3.9    1  b1
        #      8:17   5.6    7  b2-8:30
        #      8:19   6.0    2  b2       4
        #      8:20   9.8    6  b2       1
        #                       b3-8:45  2 (1)
        #      9:00   4.5    2  b4-9:00
        #      9:10   1.1    8  b4
        #                       b5-9:15
        #      9:30   3.2    1  b6-9:30
        (
            [5, 11],
            Grouper(freq="15T", key="dti", closed="left", label="right"),
            None,
            [pDataFrame({FIRST: [5.6], SUM: [9]}), pDataFrame({FIRST: [3.2], SUM: [1]})],
            [
                pTimestamp("2020-01-01 08:45:00"),
                pTimestamp("2020-01-01 09:00:00"),
                pTimestamp("2020-01-01 09:30:00"),
            ],
            None,
        ),
        # 2/ 6-rows bin left closed
        # Testing with new bin starting at iter. 2.
        #  'data'
        #  datetime value  qty  bins     row_idx (group)
        #      8:10   4.0    4  b1-8:10  0 (0)
        #      8:10   4.2    3  b1
        #      8:12   3.9    1  b1
        #      8:17   5.6    7  b1
        #      8:19   6.0    2  b1
        #      8:20   9.8    6  b1       5
        #      9:00   4.5    2  b2-9:00  1
        #      9:10   1.1    8  b2
        #      9:30   3.2    1  b2
        (
            [6, 11],
            partial(by_x_rows, by=6, closed="left"),
            "dti",
            [pDataFrame({FIRST: [4.0], SUM: [23]}), pDataFrame({FIRST: [4.5], SUM: [11]})],
            [],
            pDataFrame(
                {FIRST: [4.0, 4.5], SUM: [23, 11]},
                index=[pTimestamp("2020-01-01 08:10:00"), pTimestamp("2020-01-01 09:00:00")],
            ),
        ),
        # 4/ 6-rows bin left closed
        #  'data'
        #  datetime value  qty  bins     row_idx (group)
        #      8:10   4.0    4  b1-8:10  0 (0)
        #      8:10   4.2    3  b1
        #      8:12   3.9    1  b1
        #      8:17   5.6    7  b1
        #      8:19   6.0    2  b1       5
        #      8:20   9.8    6  b1       1
        #      9:00   4.5    2  b2-9:00
        #      9:10   1.1    8  b2
        #      9:30   3.2    1  b2
        (
            [5, 11],
            partial(by_x_rows, by=6, closed="left"),
            "dti",
            [pDataFrame({FIRST: [4.0], SUM: [17]}), pDataFrame({FIRST: [4.5], SUM: [11]})],
            [],
            pDataFrame(
                {FIRST: [4.0, 4.5], SUM: [23, 11]},
                index=[pTimestamp("2020-01-01 08:10:00"), pTimestamp("2020-01-01 09:00:00")],
            ),
        ),
    ],
)
def test_cumsegagg_bin_only(
    end_indices, bin_by, ordered_on, last_chunk_res_ref, indices_of_null_res, bin_res_ref
):
    # Test binning with null chunks.
    # 'data' as follow
    values = array([4.0, 4.2, 3.9, 5.6, 6.0, 9.8, 4.5, 1.1, 3.2], dtype=DTYPE_FLOAT64)
    qties = array([4, 3, 1, 7, 2, 6, 2, 8, 1], dtype=DTYPE_INT64)
    dtidx = array(
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
    value = "value"
    qty = "qty"
    dti = "dti"
    data = pDataFrame({value: values, qty: qties, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        SUM: (qty, SUM),
    }
    if isinstance(bin_by, Grouper):
        # Reference results from pandas Grouper.
        bin_res_ref = data.groupby(bin_by).agg(**agg)
    else:
        # Last cosmetic changes on 'bin_res_ref' if a DataFrame.
        bin_res_ref.index.name = dti
    if indices_of_null_res:
        # Update null int values.
        bin_res_ref[SUM] = bin_res_ref[SUM].astype(DTYPE_NULLABLE_INT64)
        bin_res_ref.loc[indices_of_null_res, SUM] = pNA
    # Initialize.
    agg = setup_cumsegagg(agg, data.dtypes.to_dict())
    bin_by = setup_segmentby(bin_by, ordered_on=ordered_on)
    start_idx = 0
    buffer = {}
    bin_res_to_concatenate = []
    # Run in loop.
    for i, end_idx in enumerate(end_indices):
        bin_res = cumsegagg(
            data=data.iloc[start_idx:end_idx], agg=agg, bin_by=bin_by, buffer=buffer
        )
        assert buffer[KEY_LAST_CHUNK_RES].equals(last_chunk_res_ref[i])
        bin_res_to_concatenate.append(bin_res)
        start_idx = end_idx
    bin_res = pconcat(bin_res_to_concatenate)
    bin_res = bin_res[~bin_res.index.duplicated(keep="last")]
    assert bin_res.equals(bin_res_ref)


@pytest.mark.parametrize(
    "bin_by, ordered_on, snap_by, end_indices, " "last_chunk_res_ref, bin_ref, snap_ref",
    [
        (
            # 0/ 'bin_by' using 'by_x_rows', 'snap_by' as Series.
            #    In this test case, 'unknown_bin_end' is False after 1st data
            #    chunk.
            #    Last rows in 'data' are not accounted for in 1st series of
            #    snapshots (8:20 snasphot not present), but is in 2nd series.
            #   dti  odr  val  slices   bins     snaps
            #  8:10    0    1         1-8:10
            #  8:10    1    4
            #                                   1-8:12 (excl.)
            #  8:12    2    6
            #                                   2-8:15
            #  8:17    3    2
            #  8:19    4    3   0     2-8:19
            #                                   3-8:20 (excl.)
            #  8:20    5    1
            #                                   4-8:35
            #                                   5-8:40
            #                                   6-9:00 (excl.)
            #  9:00    6    9
            #  9:10    7    3
            #                                   7-9:11
            #                                   8-9:20
            #  9:30    8    2         3-9:30
            #                                   9-9:40
            #                                  10-9:44
            by_x_rows,
            "dti",
            [
                DatetimeIndex(["2020-01-01 08:12", "2020-01-01 08:15"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:15",
                        "2020-01-01 08:20",
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
            [4, 9],
            # last_chunk_res_ref
            [pDataFrame({MIN: [1], LAST: [2]}), pDataFrame({MIN: [2], LAST: [2]})],
            # bin_ref
            pDataFrame(
                {MIN: [1, 1, 2], LAST: [2, 3, 2]},
                index=DatetimeIndex(["2020-01-01 08:10", "2020-01-01 08:19", "2020-01-01 09:30"]),
            ),
            # snap_ref
            pDataFrame(
                {MIN: [1, 1, 3, 1, 1, 1, 1, 1, 2], LAST: [4, 6, 3, 1, 1, 1, 3, 3, 2]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 08:15",
                        "2020-01-01 08:20",
                        "2020-01-01 08:35",
                        "2020-01-01 08:40",
                        "2020-01-01 09:00",
                        "2020-01-01 09:11",
                        "2020-01-01 09:20",
                        "2020-01-01 09:40",
                    ]
                ),
            ),
        ),
        (
            # 1/ 'bin_by' as Grouper and 'snap_by' as Series.
            #    In this test case, 2nd chunk is traversed without new snap,
            #    and without new bin.
            #   dti  odr  val  slices   bins     snaps
            #  8:10    0    1         1-8:00
            #  8:10    1    4
            #                                   1-8:12 (excl.)
            #  8:12    2    6
            #  8:17    3    2   0
            #  8:19    4    3
            #  8:20    5    1
            #  9:00    6    9   0     2-9:00
            #                                   2-9:10 (excl.)
            #  9:10    7    3
            #  9:30    8    2
            # bin_by
            Grouper(freq="1H", label="left", closed="left", key="dti"),
            # ordered_on
            None,
            # snap_by
            [
                DatetimeIndex(["2020-01-01 08:12"]),
                DatetimeIndex(["2020-01-01 08:12"]),
                DatetimeIndex(["2020-01-01 08:12", "2020-01-01 09:10"]),
            ],
            # end_indices
            [3, 6, 9],
            # last_chunk_res_ref
            [
                pDataFrame({MIN: [1], LAST: [6]}),
                pDataFrame({MIN: [1], LAST: [1]}),
                pDataFrame({MIN: [2], LAST: [2]}),
            ],
            # bin_ref
            pDataFrame(
                {MIN: [1, 2], LAST: [1, 2]},
                index=DatetimeIndex(["2020-01-01 08:00", "2020-01-01 09:00"]),
            ),
            # snap_ref
            pDataFrame(
                {MIN: [1, 9], LAST: [4, 9]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 09:10",
                    ]
                ),
            ),
        ),
    ],
)
def test_cumsegagg_bin_snap(
    bin_by, ordered_on, snap_by, end_indices, last_chunk_res_ref, bin_ref, snap_ref
):
    qties = array([1, 4, 6, 2, 3, 1, 9, 3, 2], dtype=DTYPE_INT64)
    dtidx = array(
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
    qty = "qty"
    dti = "dti"
    data = pDataFrame({qty: qties, dti: dtidx})
    agg = {
        MIN: (qty, MIN),
        LAST: (qty, LAST),
    }
    start_idx = 0
    buffer = {}
    # Initialize.
    agg = setup_cumsegagg(agg, data.dtypes.to_dict())
    bin_by = setup_segmentby(bin_by, ordered_on=ordered_on)
    start_idx = 0
    buffer = {}
    bin_res_to_concatenate = []
    snap_res_to_concatenate = []
    # Run in loop.
    for i, end_idx in enumerate(end_indices):
        bin_res, snap_res = cumsegagg(
            data=data.iloc[start_idx:end_idx],
            agg=agg,
            bin_by=bin_by,
            buffer=buffer,
            ordered_on=ordered_on,
            snap_by=snap_by[i] if isinstance(snap_by, list) else snap_by,
        )
        assert buffer[KEY_LAST_CHUNK_RES].equals(last_chunk_res_ref[i])
        bin_res_to_concatenate.append(bin_res)
        snap_res_to_concatenate.append(snap_res)
        start_idx = end_idx
    bin_ref.index.name = dti
    bin_res = pconcat(bin_res_to_concatenate)
    bin_res = bin_res[~bin_res.index.duplicated(keep="last")]
    assert bin_res.equals(bin_ref)
    snap_ref.index.name = dti
    snap_res = pconcat(snap_res_to_concatenate)
    snap_res = snap_res[~snap_res.index.duplicated(keep="last")]
    assert snap_res.equals(snap_ref)


# Questions:
#  - in cumsegagg, when restarting with 3 empty chunks, assuming the
#    there are 2 'snaps', and the 3rd is a 'bin' (which was in progress at
#    prev iter.)
#    With proposed methodo, it is not enough to simply let the 'in-progress data'
#    from previous iteration. the new intermediate chunk needs to be created.
#
# Start with no snapshot in 1st iteration (using Series): does it still produce
# an empty snapshot dataframe?
#
#
# Make test
#   - with empty snap at prev iter, then new empty snaps at next iter then end of bin.
#   - with non empty snap at prev iter, then new empty one (res forwarded?) at next iter then end of bin
#   - is it possible to have 1st an empty snapshot then a not empty new bin
#     (similar to test case 0 above, but 2nd snap at 8:18)
#     -> should modify 'by_x_rows': first bin (of next ier.) can never be new when end of bin (at prev. iter.)
#        is unknown. Because at next iter, we need to be able to have clear end oflast on-going bin to position
#        first snapshots
#     -> if bin are left-opened, check if this works the same.
#     Write this reasoning clearly.
#     Raise an error when detecting 'closed' = left (right open) and 'first_bin_is_new' then this case can be
#     not correctly supported.
