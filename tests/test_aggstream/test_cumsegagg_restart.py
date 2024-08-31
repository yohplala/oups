#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

"""
from functools import partial

import pytest
from numpy import array
from numpy import random as nrandom
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Timedelta
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat
from pandas.core.resample import TimeGrouper

from oups.aggstream.cumsegagg import DTYPE_DATETIME64
from oups.aggstream.cumsegagg import DTYPE_FLOAT64
from oups.aggstream.cumsegagg import DTYPE_INT64
from oups.aggstream.cumsegagg import DTYPE_NULLABLE_INT64
from oups.aggstream.cumsegagg import KEY_LAST_CHUNK_RES
from oups.aggstream.cumsegagg import cumsegagg
from oups.aggstream.cumsegagg import setup_cumsegagg
from oups.aggstream.jcumsegagg import FIRST
from oups.aggstream.jcumsegagg import LAST
from oups.aggstream.jcumsegagg import MIN
from oups.aggstream.jcumsegagg import SUM
from oups.aggstream.segmentby import by_x_rows
from oups.aggstream.segmentby import setup_segmentby


# from pandas.testing import assert_frame_equal


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
            TimeGrouper(freq="15T", key="dti", closed="left", label="right"),
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
            TimeGrouper(freq="15T", key="dti", closed="left", label="right"),
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
    end_indices,
    bin_by,
    ordered_on,
    last_chunk_res_ref,
    indices_of_null_res,
    bin_res_ref,
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
    if isinstance(bin_by, TimeGrouper):
        # Reference results from pandas TimeGrouper.
        bin_res_ref = data.groupby(bin_by).agg(**agg)
    else:
        # Last cosmetic changes on 'bin_res_ref' if a DataFrame.
        bin_res_ref.index.name = dti
    bin_res_ref[SUM] = bin_res_ref[SUM].astype(DTYPE_NULLABLE_INT64)
    if indices_of_null_res:
        # Update null int values.
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
            data=data.iloc[start_idx:end_idx],
            agg=agg,
            bin_by=bin_by,
            buffer=buffer,
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
            # snap_by
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
                    ],
                ),
            ],
            [4, 9],
            # last_chunk_res_ref
            [pDataFrame({MIN: [1], LAST: [2]}), pDataFrame({MIN: [2], LAST: [2]})],
            # bin_ref
            pDataFrame(
                {MIN: [1, 1, 2], LAST: [2, 3, 2]},
                index=DatetimeIndex(["2020-01-01 08:10", "2020-01-01 08:19", "2020-01-01 09:30"]),
                dtype=DTYPE_NULLABLE_INT64,
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
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
        ),
        (
            # 1/ 'bin_by' as TimeGrouper and 'snap_by' as Series.
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
            TimeGrouper(freq="1H", label="left", closed="left", key="dti"),
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
                dtype=DTYPE_NULLABLE_INT64,
            ),
            # snap_ref
            pDataFrame(
                {MIN: [1, 9], LAST: [4, 9]},
                index=DatetimeIndex(["2020-01-01 08:12", "2020-01-01 09:10"]),
                dtype=DTYPE_NULLABLE_INT64,
            ),
        ),
        (
            # 2/ 'bin_by' as TimeGrouper and 'snap_by' as Series.
            #    In this test case, 2nd iteration contains null bins/snaps.
            #    Testing concatenation of int64 / Int64 (pandas nullable int)
            #    pandas DataFrame.
            #   dti  odr  val  slices   bins     snaps
            #  8:10    0    1         1-8:00
            #  8:10    1    4
            #                                   1-8:12 (excl.)
            #  8:12    2    6
            #  8:17    3    2
            #  8:19    4    3
            #  8:20    5    1
            #                         2-8:30                    empty bin
            #                                   2-8:33 (excl.)
            #                                   3-9:00 (excl.)
            #  9:00    6    9   0     3-9:00
            #                                   4-9:10 (excl.)
            #  9:10    7    3
            #  9:30    8    2         4-9:30
            # bin_by
            TimeGrouper(freq="30T", label="left", closed="left", key="dti"),
            # ordered_on
            None,
            # snap_by
            [
                DatetimeIndex(["2020-01-01 08:12"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 08:33",
                        "2020-01-01 09:00",
                        "2020-01-01 09:10",
                    ],
                ),
            ],
            # end_indices
            [6, 9],
            # last_chunk_res_ref
            [pDataFrame({MIN: [1], LAST: [1]}), pDataFrame({MIN: [2], LAST: [2]})],
            # bin_ref
            pDataFrame(
                {MIN: [1, pNA, 3, 2], LAST: [1, pNA, 3, 2]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:00",
                        "2020-01-01 08:30",
                        "2020-01-01 09:00",
                        "2020-01-01 09:30",
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
            # snap_ref
            pDataFrame(
                {MIN: [1, pNA, pNA, 9], LAST: [4, pNA, pNA, 9]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 08:33",
                        "2020-01-01 09:00",
                        "2020-01-01 09:10",
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
        ),
        (
            # 3/ 'bin_by' as TimeGrouper and 'snap_by' as Series.
            #    In this test case, 2nd iteration starts with 2 empty snaps,
            #    then a null bin.
            #   dti  odr  val  slices   bins     snaps
            #  8:10    0    1         1-8:00
            #  8:10    1    4
            #                                   1-8:12 (excl.)
            #  8:12    2    6
            #  8:17    3    2
            #  8:19    4    3
            #  8:20    5    1
            #                                   2-8:22 (excl.)
            #                                   3-9:23 (excl.)
            #                         2-8:30                    empty bin
            #  9:00    6    9   0     3-9:00
            #                                   4-9:10 (excl.)
            #  9:10    7    3
            #  9:30    8    2         4-9:30
            # bin_by
            TimeGrouper(freq="30T", label="left", closed="left", key="dti"),
            # ordered_on
            None,
            # snap_by
            [
                DatetimeIndex(["2020-01-01 08:12"]),
                DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 08:22",
                        "2020-01-01 08:23",
                        "2020-01-01 09:10",
                    ],
                ),
            ],
            # end_indices
            [6, 9],
            # last_chunk_res_ref
            [pDataFrame({MIN: [1], LAST: [1]}), pDataFrame({MIN: [2], LAST: [2]})],
            # bin_ref
            pDataFrame(
                {MIN: [1, pNA, 3, 2], LAST: [1, pNA, 3, 2]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:00",
                        "2020-01-01 08:30",
                        "2020-01-01 09:00",
                        "2020-01-01 09:30",
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
            # snap_ref
            pDataFrame(
                {MIN: [1, 1, 1, 9], LAST: [4, 1, 1, 9]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:12",
                        "2020-01-01 08:22",
                        "2020-01-01 08:23",
                        "2020-01-01 09:10",
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
        ),
        (
            # 4/ 'bin_by' as TimeGrouper and 'snap_by' as Series.
            #    In this test case, 1st iteration is started without snapshot
            #    (empty array), as not known yet!
            #   dti  odr  val  slices   bins     snaps
            #  8:10    0    1         1-8:00
            #  8:10    1    4
            #  8:12    2    6
            #  8:17    3    2
            #  8:19    4    3
            #  8:20    5    1
            #                         2-8:30                    empty bin
            #                                   1-8:33 (excl.)
            #                                   2-9:00 (excl.)
            #  9:00    6    9   0     3-9:00
            #                                   3-9:10 (excl.)
            #  9:10    7    3
            #  9:30    8    2         4-9:30
            # bin_by
            TimeGrouper(freq="30T", label="left", closed="left", key="dti"),
            # ordered_on
            None,
            # snap_by
            [
                DatetimeIndex([]),
                DatetimeIndex(["2020-01-01 08:33", "2020-01-01 09:00", "2020-01-01 09:10"]),
            ],
            # end_indices
            [6, 9],
            # last_chunk_res_ref
            [pDataFrame({MIN: [1], LAST: [1]}), pDataFrame({MIN: [2], LAST: [2]})],
            # bin_ref
            pDataFrame(
                {MIN: [1, pNA, 3, 2], LAST: [1, pNA, 3, 2]},
                index=DatetimeIndex(
                    [
                        "2020-01-01 08:00",
                        "2020-01-01 08:30",
                        "2020-01-01 09:00",
                        "2020-01-01 09:30",
                    ],
                ),
                dtype=DTYPE_NULLABLE_INT64,
            ),
            # snap_ref
            pDataFrame(
                {MIN: [pNA, pNA, 9], LAST: [pNA, pNA, 9]},
                index=DatetimeIndex(["2020-01-01 08:33", "2020-01-01 09:00", "2020-01-01 09:10"]),
                dtype=DTYPE_NULLABLE_INT64,
            ),
        ),
    ],
)
def test_cumsegagg_bin_snap(
    bin_by,
    ordered_on,
    snap_by,
    end_indices,
    last_chunk_res_ref,
    bin_ref,
    snap_ref,
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


def test_cumsegagg_bin_snap_time_grouper():
    # Testing bins and snapshots, both as time grouper.
    # This test case has shown trouble that transforming input data within
    # 'setup_segmentby()' may lead to. Transformed values 'snap_by',
    # and 'bin_on' were then not retrieved in 'cumsegagg()', leading to
    # segmentation fault.
    ordered_on = "ts"
    val = "val"
    bin_by = TimeGrouper(key=ordered_on, freq="10T", closed="left", label="left")
    agg = {FIRST: (val, FIRST), LAST: (val, LAST)}
    snap_by = TimeGrouper(key=ordered_on, freq="5T", closed="left", label="left")
    # Seed data.
    start = pTimestamp("2020/01/01")
    rr = nrandom.default_rng(1)
    N = 50
    rand_ints = rr.integers(120, size=N)
    rand_ints.sort()
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed = pDataFrame({ordered_on: ts, val: rand_ints})
    # Setup for restart
    agg_as_list = setup_cumsegagg(agg, seed.dtypes.to_dict())
    bin_by_as_dict = setup_segmentby(bin_by=bin_by, ordered_on=ordered_on, snap_by=snap_by)
    seed1 = seed[:28]
    seed2 = seed[28:]
    buffer = {}
    # Aggregation
    bin_res1, snap_res1 = cumsegagg(
        data=seed1,
        agg=agg_as_list,
        bin_by=bin_by_as_dict,
        buffer=buffer,
    )
    bin_res2, snap_res2 = cumsegagg(
        data=seed2,
        agg=agg_as_list,
        bin_by=bin_by_as_dict,
        buffer=buffer,
    )
    bin_res = pconcat([bin_res1, bin_res2])
    bin_res = bin_res[~bin_res.index.duplicated(keep="last")]
    snap_res = pconcat([snap_res1, snap_res2])
    snap_res = snap_res[~snap_res.index.duplicated(keep="last")]
    # Reference results obtained by a straight execution.
    bin_res_ref, snap_res_ref = cumsegagg(
        data=seed,
        agg=agg,
        bin_by=bin_by,
        ordered_on=ordered_on,
        snap_by=snap_by,
    )
    assert bin_res.equals(bin_res_ref)
    assert snap_res.equals(snap_res_ref)
