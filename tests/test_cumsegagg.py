#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

"""
from functools import partial

import pytest
from numpy import NaN as nNaN
from numpy import all as nall
from numpy import array
from numpy import ndarray
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import NaT as pNaT
from pandas import Timestamp as pTimestamp
from pandas import date_range

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_FLOAT64
from oups.cumsegagg import DTYPE_INT64
from oups.cumsegagg import DTYPE_NULLABLE_INT64
from oups.cumsegagg import cumsegagg
from oups.cumsegagg import setup_cumsegagg
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import LAST
from oups.jcumsegagg import MAX
from oups.jcumsegagg import MIN
from oups.jcumsegagg import SUM
from oups.jcumsegagg import jfirst
from oups.jcumsegagg import jlast
from oups.jcumsegagg import jmax
from oups.jcumsegagg import jmin
from oups.jcumsegagg import jsum
from oups.segmentby import by_x_rows


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


def test_setup_csagg():
    # Test config generation for aggregation step in cumsegagg.
    # ``{dtype: List[str], 'cols_name_in_data'
    #                      column name in input data, with this dtype,
    #           List[str], 'cols_name_in_agg_res'
    #                      expected column names in aggregation result,
    #           ndarray[int64], 'cols_idx'
    #                           2d-array,
    #                           Per aggregation function (row index is agg func
    #                           id), column indices in input data, and results.
    #           ndarray[int64], 'n_cols'
    #                           1d-array, number of input columns in data,
    #                           to which apply this aggregation function,
    # Setup.
    df = pDataFrame(
        {
            "val1_float": [1.1, 2.1, 3.1],
            "val2_float": [4.1, 5.1, 6.1],
            "val3_int": [1, 2, 3],
            "val4_datetime": [
                pTimestamp("2022/01/01 08:00"),
                pTimestamp("2022/01/01 09:00"),
                pTimestamp("2022/01/01 08:00"),
            ],
        },
    )
    agg_cfg = {
        "val1_first": ("val1_float", FIRST),
        "val2_first": ("val2_float", FIRST),
        "val2_sum": ("val2_float", SUM),
        "val4_first": ("val4_datetime", FIRST),
        "val3_last": ("val3_int", LAST),
        "val3_min": ("val3_int", MIN),
        "val3_max": ("val3_int", MAX),
    }
    csagg_cfg_res = setup_cumsegagg(agg_cfg, df.dtypes.to_dict())
    # In reference results, the 3rd iterable is voluntarily a tuple, to "flag"
    # it and unpack it when checking it ('assert' below).
    csagg_cfg_ref = {
        DTYPE_FLOAT64: [
            ["val1_float", "val2_float"],
            ["val1_first", "val2_first", "val2_sum"],
            (
                (jfirst, array([0, 1], dtype=DTYPE_INT64), array([0, 1], dtype=DTYPE_INT64)),
                (
                    jsum,
                    array(
                        [
                            1,
                        ],
                        dtype=DTYPE_INT64,
                    ),
                    array([2], dtype=DTYPE_INT64),
                ),
            ),
            3,
        ],
        DTYPE_DATETIME64: [
            ["val4_datetime"],
            ["val4_first"],
            ((jfirst, array([0], dtype=DTYPE_INT64), array([0], dtype=DTYPE_INT64)),),
            1,
        ],
        DTYPE_INT64: [
            ["val3_int"],
            ["val3_last", "val3_min", "val3_max"],
            (
                (jlast, array([0], dtype=DTYPE_INT64), array([0], dtype=DTYPE_INT64)),
                (jmin, array([0], dtype=DTYPE_INT64), array([1], dtype=DTYPE_INT64)),
                (jmax, array([0], dtype=DTYPE_INT64), array([2], dtype=DTYPE_INT64)),
            ),
            3,
        ],
    }
    for val_res, val_ref in zip(csagg_cfg_res.values(), csagg_cfg_ref.values()):
        for it_res, it_ref in zip(val_res, val_ref):
            if isinstance(it_ref, tuple):
                for sub_it_res, sub_it_ref in zip(it_res, it_ref):
                    for sub_sub_it_res, sub_sub_it_ref in zip(sub_it_res, sub_it_ref):
                        if isinstance(sub_sub_it_ref, ndarray):
                            nall(sub_sub_it_res == sub_sub_it_ref)
                        else:
                            assert sub_sub_it_res == sub_sub_it_ref
            else:
                assert it_res == it_ref


@pytest.mark.parametrize(
    "ndata, cols",
    [
        (
            array(
                [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]],
                dtype=DTYPE_FLOAT64,
            ),
            True,
        ),
        (array([[2, 20], [4, 40], [5, 50], [8, 80], [9, 90]], dtype=DTYPE_INT64), True),
        (
            array(
                [
                    ["2020-01-01T09:00", "2020-01-02T10:00"],
                    ["2020-01-01T09:05", "2020-01-02T10:05"],
                    ["2020-01-01T09:08", "2020-01-02T10:08"],
                    ["2020-01-01T09:12", "2020-01-02T10:12"],
                    ["2020-01-01T09:14", "2020-01-02T14:00"],
                ],
                dtype=DTYPE_DATETIME64,
            ),
            True,
        ),
        (array([2.0, 4.0, 5.0, 8.0, 9.0], dtype=DTYPE_FLOAT64), False),
        (array([20, 40, 50, 80, 90], dtype=DTYPE_INT64), False),
        (
            array(
                [
                    "2020-01-01T09:00",
                    "2020-01-01T09:05",
                    "2020-01-01T09:08",
                    "2020-01-01T09:12",
                    "2020-01-01T09:14",
                ],
                dtype=DTYPE_DATETIME64,
            ),
            False,
        ),
    ],
)
def test_cumsegagg_bin_single_dtype(ndata, cols):
    # Test binning aggregation for a single dtype.
    ndata_dti = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:05",
            "2020-01-01T08:08",
            "2020-01-01T08:12",
            "2020-01-01T08:14",
        ],
        dtype=DTYPE_DATETIME64,
    )
    time_idx = "datetime_idx"
    if cols:
        # 2-column data
        data = pDataFrame(
            {
                "col1": ndata[:, 0],
                "col2": ndata[:, 1],
                time_idx: ndata_dti,
            },
        )
        agg = {
            "res_first": ("col1", "first"),
            "res_last_col1": ("col1", "last"),
            "res_last_col2": ("col2", "last"),
        }
    else:
        # 1-column data
        data = pDataFrame(
            {
                "col1_f": ndata,
                time_idx: ndata_dti,
            },
        )
        agg = {"res_first": ("col1_f", "first"), "res_last": ("col1_f", "last")}
    by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    bin_res = cumsegagg(data=data, agg=agg, bin_by=by)
    bin_res_ref = data.groupby(by).agg(**agg)
    assert bin_res.equals(bin_res_ref)


def test_cumsegagg_bin_mixed_dtype():
    # Test binning aggregation for a mixed dtype.
    ar_float = array(
        [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]],
        dtype=DTYPE_FLOAT64,
    )
    ar_int = array([[1, 10], [3, 30], [6, 60], [7, 70], [9, 90]], dtype=DTYPE_INT64)
    ar_dti = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:05",
            "2020-01-01T08:08",
            "2020-01-01T08:12",
            "2020-01-01T08:14",
        ],
        dtype=DTYPE_DATETIME64,
    )
    time_idx = "datetime_idx"
    data = pDataFrame(
        {
            "col1_f": ar_float[:, 0],
            "col2_f": ar_float[:, 1],
            "col3_i": ar_int[:, 0],
            "col4_i": ar_int[:, 1],
            time_idx: ar_dti,
        },
    )
    agg = {
        "res_first_f": ("col1_f", "first"),
        "res_sum_f": ("col1_f", "sum"),
        "res_last_f": ("col2_f", "last"),
        "res_min_f": ("col3_i", "min"),
        "res_max_f": ("col4_i", "max"),
        "res_first_d": ("datetime_idx", "first"),
    }
    bin_by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    agg_res = cumsegagg(data=data, agg=agg, bin_by=bin_by)
    agg_res_ref = data.groupby(bin_by).agg(**agg)
    assert agg_res.equals(agg_res_ref)


@pytest.mark.parametrize(
    "b_by_closed, b_by_label, s_first_val, s_max_val, s_min_val, s_last_val, s_sum_qty, s_first_ts, null_b_dti, start_s_dti, null_s_dti",
    [
        # 1/ bin left closed; right label, point of observations excluded
        #  'data'
        #  datetime value  qty      snaps     bins   to check
        #                                             no 5mn snapshots empty
        #      8:10   4.0    4    s1-8:15  b1-8:30   aggregation over 2 values
        #      8:10   4.2    3    s1       b1
        #      8:12   3.9    1    s1       b1
        #      8:17   5.6    7    s2-8:20  b1        aggregation over 2 values
        #      8:19   6.0    2    s2       b1
        #      8:20   9.8    6    s3-8:25  b1        observation point is excluded
        #                         s4-8:30  b1        1 constant snapshot
        #                         s5-8:35  b2-9:00   6 empty snapshots
        #                               |  b2        1 empty bin
        #                        s10-9:00  b2
        #      9:00   4.5    2   s11-9:05  b3-9:30
        #                        s12-9:10  b3        1 constant snapshot
        #      9:10   1.1    8   s13-9:15  b3
        #                        s14-9:20  b3        4 constant snapshots
        #                               |  b3
        #                        s16-9:30  b3
        #      9:30   3.2    1   s17-9:35  b4-10:00
        (
            "left",
            "right",
            # s1->s4    s5->s10      s11->s16    s17                           (first)
            [4.0] * 4 + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1->s4               s5->s10      s11->s16    s17                (max)
            [4.2, 6.0, 9.8, 9.8] + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1->s4    s5->s10      s11, s12     s13->s16    s17              (min)
            [3.9] * 4 + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4               s5->s10      s11, s12     s13->s16    s17   (last)
            [3.9, 6.0, 9.8, 9.8] + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4           s5->s10  s11, s12 s13->s16   s17                (sum)
            [8, 17, 23, 23] + [0] * 6 + [2, 2] + [10] * 4 + [1],
            [pTimestamp("2020-01-01 08:10:00")] * 4
            + [pNaT] * 6
            + [pTimestamp("2020-01-01 09:00:00")] * 6
            + [pTimestamp("2020-01-01 09:30:00")],
            # null_bins_dti
            [pTimestamp("2020-01-01 09:00:00")],
            # start_s_dti
            "2020-01-01 08:15:00",
            # null_snaps_dti
            date_range("2020-01-01 08:35:00", periods=6, freq="5T"),
        ),
        # 2/ bin right closed; left label, point of observations included
        #  'data'
        #  datetime value  qty      snaps      bins   to check
        #                                             no 5mn snapshots empty
        #      8:10   4.0    4    s1-8:10   b1-8:00   aggregation over 2 values
        #      8:10   4.2    3    s1        b1
        #      8:12   3.9    1    s2-8:15   b1
        #      8:17   5.6    7    s3-8:20   b1        aggregation over 2 values
        #      8:19   6.0    2    s3        b1
        #      8:20   9.8    6    s3        b1        observation point is included
        #                         s4-8:25   b1        2 constant snapshots
        #                         s5-8:30   b1
        #                         s6-8:35   b2-8:30   5 empty snapshots
        #                               |   b2
        #                        s10-8:55   b2
        #      9:00   4.5    2   s11:9:00   b2
        #                        s12-9:05   b3-9:00   1 empty snapshot
        #      9:10   1.1    8   s13-9:10   b3        4 constant snapshots
        #                        s14-9:15   b3
        #                               |   b3
        #                        s16-9:25   b3
        #      9:30   3.2    1   s17-9:30   b3
        (
            "right",
            "left",
            # s1->s5    s6->s10      s11, s12       s13->s17                    (first)
            [4.0] * 5 + [nNaN] * 5 + [4.5, nNaN] + [1.1] * 5,
            # s1, s2     s3->s5      s6->s10      s11, s12      s13->s16    s17 (max)
            [4.2, 4.2] + [9.8] * 3 + [nNaN] * 5 + [4.5, nNaN] + [1.1] * 4 + [3.2],
            # s1    s2->s5      s6->s10      s11, s12     s13->s16              (min)
            [4.0] + [3.9] * 4 + [nNaN] * 5 + [4.5, nNaN] + [1.1] * 5,
            # s1, s2     s3->s5      s6->s10      s11, s12      s13->s16    s17 (last)
            [4.2, 3.9] + [9.8] * 3 + [nNaN] * 5 + [4.5, nNaN] + [1.1] * 4 + [3.2],
            # s1, s2 s3->s5     s6->s10   s11, s12 s13->s16  s17                (sum)
            [7, 8] + [23] * 3 + [0] * 5 + [2, 0] + [8] * 4 + [9],
            [pTimestamp("2020-01-01 08:10:00")] * 5
            + [pNaT] * 5
            + [pTimestamp("2020-01-01 09:00:00")]
            + [pNaT]
            + [pTimestamp("2020-01-01 09:10:00")] * 5,
            # null_bins_dti
            [],
            # start_s_dti
            "2020-01-01 08:10:00",
            # null_snaps_dti
            date_range("2020-01-01 08:35:00", periods=5, freq="5T").to_list()
            + [pTimestamp("2020-01-01 09:05:00")],
        ),
    ],
)
def test_cumsegagg_bin_snap_with_null_chunks(
    b_by_closed,
    b_by_label,
    s_first_val,
    s_max_val,
    s_min_val,
    s_last_val,
    s_sum_qty,
    s_first_ts,
    null_b_dti,
    start_s_dti,
    null_s_dti,
):
    # Test binning and snapshotting aggregation with null chunks.
    # - 5 minutes snapshots
    # - 30 minutes bins
    # - 'data' as follow below
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
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
        SUM: (qty, SUM),
        "ts_first": (dti, FIRST),
    }
    bin_by = Grouper(freq="30T", closed=b_by_closed, label=b_by_label, key=dti)
    snap_by = Grouper(freq="5T", closed=b_by_closed, key=dti, label="right")
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    if null_b_dti:
        bins_ref[SUM] = bins_ref[SUM].astype(DTYPE_NULLABLE_INT64)
        bins_ref.loc[null_b_dti, SUM] = pNA
    assert bins_res.equals(bins_ref)
    snaps_dti = date_range(start_s_dti, periods=len(s_first_val), freq="5T")
    snaps_ref = pDataFrame(
        {
            FIRST: s_first_val,
            MAX: s_max_val,
            MIN: s_min_val,
            LAST: s_last_val,
            SUM: s_sum_qty,
            "ts_first": s_first_ts,
        },
        index=snaps_dti,
    )
    snaps_ref.index.name = dti
    snaps_ref[SUM] = snaps_ref[SUM].astype(DTYPE_NULLABLE_INT64)
    snaps_ref.loc[null_s_dti, SUM] = pNA
    assert snaps_res.equals(snaps_ref)


@pytest.mark.parametrize(
    "b_by_closed, b_by_label, s_first_val, s_max_val, s_min_val, s_last_val, s_sum_qty, s_first_ts, null_b_dti, start_s_dti, null_s_dti",
    [
        # 1/ bin left closed; right label, point of observations excluded
        #  'data'
        #  datetime value  qty      snaps     bins   to check
        #      8:00   4.0    4    s1-8:05  b1-8:30
        #                         s2-8:10  b1
        #      8:10   4.2    3    s3-8:15  b1
        #      8:12   3.9    1    s3       b1
        #      8:17   5.6    7    s4-8:20  b1
        #      8:19   6.0    2    s4       b1
        #      8:20   9.8    6    s5-8:25  b1
        #                         s6-8:30  b2-9:00
        #                         s7-8:35  b2-9:00
        #                               |  b2
        #                        s12-9:00  b3-9:30
        #                        s13-9:05  b3
        #                        s14-9:10  b3
        #      9:10   1.1    8   s15-9:15  b3
        (
            "left",
            "right",
            # s1->s6    s7->s14      s15                                (first)
            [4.0] * 6 + [nNaN] * 8 + [1.1],
            # s1, s2    s3->s6                 s7->s14      s15         (max)
            [4.0] * 2 + [4.2, 6.0, 9.8, 9.8] + [nNaN] * 8 + [1.1],
            # s1, s2    s3->s6      s7->s14      s15                    (min)
            [4.0] * 2 + [3.9] * 4 + [nNaN] * 8 + [1.1],
            # s1, s2    s3->s6                 s7->s14      s15         (last)
            [4.0] * 2 + [3.9, 6.0, 9.8, 9.8] + [nNaN] * 8 + [1.1],
            # s1, s2  s3->s6            s7->s14   s15                   (sum)
            [4] * 2 + [8, 17, 23, 23] + [0] * 8 + [8],
            # first timestamp in bin
            [pTimestamp("2020-01-01 08:00:00")] * 6
            + [pNaT] * 8
            + [pTimestamp("2020-01-01 09:10:00")],
            # null_bins_dti (label)
            [pTimestamp("2020-01-01 09:00:00")],
            # start_s_dti
            "2020-01-01 08:05:00",
            # null_snaps_dti
            date_range("2020-01-01 08:35:00", periods=8, freq="5T"),
        ),
        # 2/ bin right closed; left label, point of observations included
        #  'data'
        #  datetime value  qty      snaps     bins   to check
        #      8:00   4.0    4    s1-8:00  b1-7:30
        #                         s2-8:05  b2-8:00
        #      8:10   4.2    3    s3-8:10  b2
        #      8:12   3.9    1    s4-8:15  b2
        #      8:17   5.6    7    s5-8:20  b2
        #      8:19   6.0    2    s5       b2
        #      8:20   9.8    6    s5       b2
        #                         s6-8:25  b2
        #                         s7-8:30  b2
        #                         s8-8:35  b3-8:30
        #                               |  b3
        #                        s13-9:00  b3
        #                        s14-9:05  b4-9:00
        #      9:10   1.1    8   s15-9:10  b4
        (
            "right",
            "left",
            # s1, s2      s3->s7      s8->s14      s15                  (first)
            [4.0, nNaN] + [4.2] * 5 + [nNaN] * 7 + [1.1],
            # s1, s2      s3, s4      s5->s7      s8->s14      s15      (max)
            [4.0, nNaN] + [4.2] * 2 + [9.8] * 3 + [nNaN] * 7 + [1.1],
            # s1->s3           s4->s7      s8->s14      s15             (min)
            [4.0, nNaN, 4.2] + [3.9] * 4 + [nNaN] * 7 + [1.1],
            # s1->s4                s5->s7      s8->s14      s15        (last)
            [4.0, nNaN, 4.2, 3.9] + [9.8] * 3 + [nNaN] * 7 + [1.1],
            # s1->s4       s5->s7     s8->s14   s15                    (sum)
            [4, 0, 3, 4] + [19] * 3 + [0] * 7 + [8],
            # first timestamp in bin
            [pTimestamp("2020-01-01 08:00:00")]
            + [pNaT]
            + [pTimestamp("2020-01-01 08:10:00")] * 5
            + [pNaT] * 7
            + [pTimestamp("2020-01-01 09:10:00")],
            # null_bins_dti (label)
            [pTimestamp("2020-01-01 08:30:00")],
            # start_s_dti
            "2020-01-01 08:00:00",
            # null_snaps_dti
            [pTimestamp("2020-01-01 08:05:00")]
            + date_range("2020-01-01 08:35:00", periods=7, freq="5T").to_list(),
        ),
    ],
)
def test_cumsegagg_bin_snap_with_null_chunks_other(
    b_by_closed,
    b_by_label,
    s_first_val,
    s_max_val,
    s_min_val,
    s_last_val,
    s_sum_qty,
    s_first_ts,
    null_b_dti,
    start_s_dti,
    null_s_dti,
):
    # Test binning and snapshotting aggregation with null chunks.
    # - 5 minutes snapshots
    # - 30 minutes bins
    # - 'data' as follow below
    values = array([4.0, 4.2, 3.9, 5.6, 6.0, 9.8, 1.1], dtype=DTYPE_FLOAT64)
    qties = array([4, 3, 1, 7, 2, 6, 8], dtype=DTYPE_INT64)
    dtidx = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:10",
            "2020-01-01T08:12",
            "2020-01-01T08:17",
            "2020-01-01T08:19",
            "2020-01-01T08:20",
            "2020-01-01T09:10",
        ],
        dtype=DTYPE_DATETIME64,
    )
    value = "value"
    qty = "qty"
    dti = "dti"
    data = pDataFrame({value: values, qty: qties, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
        SUM: (qty, SUM),
        "ts_first": (dti, FIRST),
    }
    bin_by = Grouper(freq="30T", closed=b_by_closed, label=b_by_label, key=dti)
    snap_by = Grouper(freq="5T", closed=b_by_closed, key=dti, label="right")
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    if null_b_dti:
        bins_ref[SUM] = bins_ref[SUM].astype(DTYPE_NULLABLE_INT64)
        bins_ref.loc[null_b_dti, SUM] = pNA
    assert bins_res.equals(bins_ref)
    snaps_dti = date_range(start_s_dti, periods=len(s_first_val), freq="5T")
    snaps_ref = pDataFrame(
        {
            FIRST: s_first_val,
            MAX: s_max_val,
            MIN: s_min_val,
            LAST: s_last_val,
            SUM: s_sum_qty,
            "ts_first": s_first_ts,
        },
        index=snaps_dti,
    )
    snaps_ref.index.name = dti
    snaps_ref[SUM] = snaps_ref[SUM].astype(DTYPE_NULLABLE_INT64)
    snaps_ref.loc[null_s_dti, SUM] = pNA
    assert snaps_res.equals(snaps_ref)


@pytest.mark.parametrize(
    "b_by_closed, b_by_label, s_first_val, s_max_val, s_min_val, s_last_val, start_s_dti",
    [
        # 1/ bin left closed; left label, point of observations excluded
        #  'data'
        #  datetime value           snaps     bins
        #      8:02   4.0         s1-8:05  b1-8:00
        #                         s2-8:10  b1
        #      8:10   4.2         s3-8:15  b1
        #      8:12   3.9         s3-8:15  b1
        #      8:17   5.6         s4-8:20  b1
        #      8:19   6.0         s4       b1
        #      8:20   9.8         s5-8:25  b1
        (
            "left",
            "left",
            # s1->s5                                (first)
            [4.0] * 5,
            # s1, s2    s3, s4, s5                  (max)
            [4.0] * 2 + [4.2, 6.0, 9.8],
            # s1, s2    s3->s5                      (min)
            [4.0] * 2 + [3.9] * 3,
            # s1, s2    s3, s4, s5                  (last)
            [4.0] * 2 + [3.9, 6.0, 9.8],
            # start_s_dti
            "2020-01-01 08:05:00",
        ),
        # 2/ bin right closed; right label, point of observations included
        #  'data'
        #  datetime value           snaps     bins
        #      8:02   4.0         s1-8:05  b1-8:30
        #      8:10   4.2         s2-8:10  b1
        #      8:12   3.9         s3-8:15  b1
        #      8:17   5.6         s4-8:20  b1
        #      8:19   6.0         s4       b1
        #      8:20   9.8         s4       b1
        (
            "right",
            "right",
            # s1->s4                          (first)
            [4.0] * 4,
            # s1->s4                          (max)
            [4.0, 4.2, 4.2, 9.8],
            # s1, s2    s3, s4                (min)
            [4.0] * 2 + [3.9] * 2,
            # s1->s4                          (last)
            [4.0, 4.2, 3.9, 9.8],
            # start_s_dti
            "2020-01-01 08:05:00",
        ),
    ],
)
def test_cumsegagg_single_bin_several_snaps(
    b_by_closed,
    b_by_label,
    s_first_val,
    s_max_val,
    s_min_val,
    s_last_val,
    start_s_dti,
):
    # Test binning and snapshotting aggregation.
    # - 5 minutes snapshots
    # - 30 minutes bin (a single one)
    # - 'data' as follow below
    values = array([4.0, 4.2, 3.9, 5.6, 6.0, 9.8], dtype=DTYPE_FLOAT64)
    dtidx = array(
        [
            "2020-01-01T08:02",
            "2020-01-01T08:10",
            "2020-01-01T08:12",
            "2020-01-01T08:17",
            "2020-01-01T08:19",
            "2020-01-01T08:20",
        ],
        dtype=DTYPE_DATETIME64,
    )
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
    }
    bin_by = Grouper(freq="30T", closed=b_by_closed, label=b_by_label, key=dti)
    snap_by = Grouper(freq="5T", closed=b_by_closed, key=dti, label="right")
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    assert bins_res.equals(bins_ref)
    snaps_dti = date_range(start_s_dti, periods=len(s_first_val), freq="5T")
    snaps_ref = pDataFrame(
        {
            FIRST: s_first_val,
            MAX: s_max_val,
            MIN: s_min_val,
            LAST: s_last_val,
        },
        index=snaps_dti,
    )
    snaps_ref.index.name = dti
    assert snaps_res.equals(snaps_ref)


@pytest.mark.parametrize(
    "s_by_closed, s_res",
    [
        # 1/ bins left-closed, point of observations excluded
        #  'data'
        #  datetime value  snaps   snaps_id       bins   to check
        #                   8:00    s1-8:00    b1-8:30
        #                   8:05    s2-8:05    b1
        #                   8:06    s3-8:06    b1
        #                           s4-8:10    b1
        #      8:10   4.2                      b1
        #                   8:44    s5-8:44    b2-9:00
        #                   8:48    s6-8:48    b2
        #                   8:52    s7-8:52    b2
        #                           s8-8:55    b2
        #      8:55   3.2                      b2
        #                   9:06    s9-9:06    b3-9:30   (not existing in 'bin_res')
        #                   9:09   s10-9:09    b3        (after s9 excluded,
        #                   9:16   s11-9:16    b3         remaining snapshots are trimmed)
        (
            "left",
            # s1->s9                          (first, max, min, last)
            [nNaN] * 9,
        ),
        # 2/ bins right-closed, point of observations included
        #  'data'
        #  datetime value  snaps   snaps_id       bins   to check
        #                   8:00    s1-8:00    b1-8:00
        #                   8:05    s2-8:05    b2-8:30
        #                   8:06    s3-8:06    b2
        #      8:10   4.2           s4-8:10    b2
        #                   8:44    s5-8:44    b3-9:00
        #                   8:48    s6-8:48    b3
        #                   8:52    s7-8:52    b3
        #      8:55   3.2           s8-8:55    b3
        #                   9:06    s9-9:06    b4-9:30   (not existing in 'bin_res')
        #                   9:09   s10-9:09    b4        (after s8 excluded,
        #                   9:16   s11-9:16    b4         remaining snapshots are trimmed)
        (
            "right",
            # s1->s3     s4      s5->s7       s8
            [nNaN] * 3 + [4.2] + [nNaN] * 3 + [3.2],
        ),
    ],
)
def test_cumsegagg_several_null_snaps_at_start_of_bins(s_by_closed, s_res):
    # Test binning and snapshotting aggregation with null chunks.
    # - 30 minutes bins
    # - snapshots as an interval index
    # - 'data' as follow below
    # these tests actually make clever use of 'unknown_last_bin_end' parameter.
    values = array([4.2, 3.2], dtype=DTYPE_FLOAT64)
    dtidx = array(
        [
            "2020-01-01T08:10",
            "2020-01-01T08:55",
        ],
        dtype=DTYPE_DATETIME64,
    )
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
    }
    bin_by = Grouper(freq="30T", closed=s_by_closed, label="right", key=dti)
    snap_by = DatetimeIndex(
        [
            pTimestamp("2020-01-01 08:00:00"),
            pTimestamp("2020-01-01 08:05:00"),
            pTimestamp("2020-01-01 08:06:00"),
            pTimestamp("2020-01-01 08:10:00"),
            pTimestamp("2020-01-01 08:44:00"),
            pTimestamp("2020-01-01 08:48:00"),
            pTimestamp("2020-01-01 08:52:00"),
            pTimestamp("2020-01-01 08:55:00"),
            pTimestamp("2020-01-01 09:06:00"),
            pTimestamp("2020-01-01 09:09:00"),
            pTimestamp("2020-01-01 09:16:00"),
        ],
    )
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    assert bins_res.equals(bins_ref)
    snaps_ref = pDataFrame(
        {
            FIRST: s_res,
            MAX: s_res,
            MIN: s_res,
            LAST: s_res,
        },
        index=snap_by[: len(s_res)],
    )
    snaps_ref.index.name = dti
    assert snaps_res.equals(snaps_ref)


@pytest.mark.parametrize(
    "b_by_closed, b_by_label, s_first_val, s_max_val, s_min_val, s_last_val",
    [
        # 1/ bin left closed; right label, point of observations excluded
        #  'data'
        #  datetime value        snaps    snaps_id       bins   to check
        #                         7:58     s1-7:58
        #                         8:00     s2-8:00
        #      8:00   4.0                  s3-8:13    b1-8:30
        #      8:10   4.2                  s3         b1
        #      8:12   3.9                  s3         b1
        #                         8:13     s3         b1
        #      8:17   5.6                  s4-8:20    b1
        #      8:19   6.0                  s4         b1
        #                         8:20     s4         b1
        #      8:20   9.8                             b1
        #                                  s5-8:50    b2
        #                         8:50     s5         b2-9:00
        #                         9:06     s6-9:06    b3-9:30
        #      9:10   1.1                             b3
        (
            "left",
            "right",
            # s1, s2     s3, s4      s5, s6                           (first)
            [nNaN] * 2 + [4.0] * 2 + [nNaN] * 2,
            # s1, s2     s3, s4       s5, s6                          (max)
            [nNaN] * 2 + [4.2, 6.0] + [nNaN] * 2,
            # s1, s2     s3, s4      s5, s6                           (min)
            [nNaN] * 2 + [3.9] * 2 + [nNaN] * 2,
            # s1, s2     s3, s4       s5, s6                          (last)
            [nNaN] * 2 + [3.9, 6.0] + [nNaN] * 2,
        ),
        # 2/ bin right closed; left label, point of observations included
        #  'data'
        #  datetime value        snaps    snaps_id       bins   to check
        #                         7:58     s1-7:58    b1
        #      8:00   4.0                  s2-8:00    b1-7:30
        #      8:10   4.2                  s3-8:13    b2-8:00
        #      8:12   3.9                  s3         b2
        #                         8:13     s3         b2
        #      8:17   5.6                  s4-8:20    b2
        #      8:19   6.0                  s4         b2
        #      8:20   9.8                  s4         b2
        #                         8:50     s5-8:50    b3-8:30
        #                         9:06     s6-9:06    b4-9:00
        #      9:10   1.1                             b4
        (
            "right",
            "left",
            # s1     s2->s4            s5, s6                     (first)
            [nNaN] + [4.0, 4.2, 4.2] + [nNaN] * 2,
            # s2     s2->s4            s5, s6                     (max)
            [nNaN] + [4.0, 4.2, 9.8] + [nNaN] * 2,
            # s1     s2->s4            s5, s6                     (min)
            [nNaN] + [4.0, 3.9, 3.9] + [nNaN] * 2,
            # s1     s2->s4            s5, s6                     (last)
            [nNaN] + [4.0, 3.9, 9.8] + [nNaN] * 2,
        ),
    ],
)
def test_cumsegagg_binby_grouper_snapby_intervalindex(
    b_by_closed,
    b_by_label,
    s_first_val,
    s_max_val,
    s_min_val,
    s_last_val,
):
    # Test binning and snapshotting aggregation with null chunks.
    # - 30 minutes bins
    # - snapshots as an interval index
    # - 'data' as follow below
    values = array([4.0, 4.2, 3.9, 5.6, 6.0, 9.8, 1.1], dtype=DTYPE_FLOAT64)
    dtidx = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:10",
            "2020-01-01T08:12",
            "2020-01-01T08:17",
            "2020-01-01T08:19",
            "2020-01-01T08:20",
            "2020-01-01T09:10",
        ],
        dtype=DTYPE_DATETIME64,
    )
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
    }
    bin_by = Grouper(freq="30T", closed=b_by_closed, label=b_by_label, key=dti)
    snap_by = DatetimeIndex(
        [
            pTimestamp("2020-01-01 07:58:00"),
            pTimestamp("2020-01-01 08:00:00"),
            pTimestamp("2020-01-01 08:13:00"),
            pTimestamp("2020-01-01 08:20:00"),
            pTimestamp("2020-01-01 08:50:00"),
            pTimestamp("2020-01-01 09:06:00"),
        ],
    )
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    assert bins_res.equals(bins_ref)
    snaps_ref = pDataFrame(
        {
            FIRST: s_first_val,
            MAX: s_max_val,
            MIN: s_min_val,
            LAST: s_last_val,
        },
        index=snap_by,
    )
    snaps_ref.index.name = dti
    assert snaps_res.equals(snaps_ref)


@pytest.mark.parametrize(
    "b_by_closed, s_first_val, s_max_val, s_min_val, s_last_val",
    [
        # 1/ bin left closed, by 4 rows,
        #    point of observations excluded.
        #  'data'
        #  datetime value        snaps    snaps_id       bins   to check
        #                         7:58     s1-7:58
        #                         8:00     s2-8:00
        #      8:00   4.0                  s3-8:13    b1-8:00
        #      8:10   4.2                  s3         b1
        #      8:12   3.9                  s3         b1
        #                         8:13     s3         b1
        #      8:17   5.6                             b1
        #      8:19   6.0                  s4-8:20    b2-8:19
        #                         8:20     s4         b2
        #      8:20   9.8                             b2
        #                                  s5-8:50    b2
        #                         8:50     s5         b2
        #                         9:06     s6-9:06    b2
        #      9:10   1.1                             b2
        (
            "left",
            # s1, s2      s3,    s4->s6                         (first)
            [nNaN] * 2 + [4.0] + [6.0] * 3,
            # s1, s2     s3, s4       s5, s6                    (max)
            [nNaN] * 2 + [4.2, 6.0] + [9.8] * 2,
            # s1, s2     s3,     s4->s6                         (min)
            [nNaN] * 2 + [3.9] + [6.0] * 3,
            # s1, s2     s3,     s4->s6                         (last)
            [nNaN] * 2 + [3.9] + [6.0, 9.8, 9.8],
        ),
        # 2/ bin right closed, by 4 rows,
        #    point of observations included
        #  'data'
        #  datetime value        snaps    snaps_id       bins   to check
        #                         7:58     s1-7:58
        #      8:00   4.0                  s2-8:00    b1-8:00
        #      8:10   4.2                  s3-8:13    b1
        #      8:12   3.9                  s3         b1
        #                         8:13     s3         b1
        #      8:17   5.6                             b1
        #      8:19   6.0                  s4-8:20    b2-8:19
        #      8:20   9.8                  s4         b2
        #                         8:50     s5-8:50    b2
        #                         9:06     s6-9:06    b2
        #      9:10   1.1                             b2
        (
            "right",
            # s1     s2, s3      s4->s6                     (first)
            [nNaN] + [4.0] * 2 + [6.0] * 3,
            # s1     s2, s3       s4->s6                    (max)
            [nNaN] + [4.0, 4.2] + [9.8] * 3,
            # s1     s2, s3       s4->s6                    (min)
            [nNaN] + [4.0, 3.9] + [6.0] * 3,
            # s1     s2, s3       s4->s6                     (last)
            [nNaN] + [4.0, 3.9] + [9.8] * 3,
        ),
    ],
)
def test_cumsegagg_binby_callable_snapby_intervalindex(
    b_by_closed,
    s_first_val,
    s_max_val,
    s_min_val,
    s_last_val,
):
    # Test binning and snapshotting aggregation with null chunks.
    # - 4-row bins
    # - snapshots as an interval index
    # - 'data' as follow below
    values = array([4.0, 4.2, 3.9, 5.6, 6.0, 9.8, 1.1], dtype=DTYPE_FLOAT64)
    dtidx = array(
        [
            "2020-01-01T08:00",
            "2020-01-01T08:10",
            "2020-01-01T08:12",
            "2020-01-01T08:17",
            "2020-01-01T08:19",
            "2020-01-01T08:20",
            "2020-01-01T09:10",
        ],
        dtype=DTYPE_DATETIME64,
    )
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {
        FIRST: (value, FIRST),
        MAX: (value, MAX),
        MIN: (value, MIN),
        LAST: (value, LAST),
    }
    snap_by = DatetimeIndex(
        [
            pTimestamp("2020-01-01 07:58:00"),
            pTimestamp("2020-01-01 08:00:00"),
            pTimestamp("2020-01-01 08:13:00"),
            pTimestamp("2020-01-01 08:20:00"),
            pTimestamp("2020-01-01 08:50:00"),
            pTimestamp("2020-01-01 09:06:00"),
        ],
    )
    bin_by = partial(by_x_rows, closed=b_by_closed)
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        bin_on=dti,
        ordered_on=dti,
        snap_by=snap_by,
    )
    bins_ref = pDataFrame(
        {"first": [4.0, 6.0], "max": [5.6, 9.8], "min": [3.9, 1.1], "last": [5.6, 1.1]},
        index=DatetimeIndex(["2020-01-01 08:00:00", "2020-01-01 08:19:00"]),
    )
    assert bins_res.equals(bins_ref)
    snaps_ref = pDataFrame(
        {
            FIRST: s_first_val,
            MAX: s_max_val,
            MIN: s_min_val,
            LAST: s_last_val,
        },
        index=snap_by,
    )
    snaps_ref.index.name = dti
    assert snaps_res.equals(snaps_ref)


def test_exit_on_null_data():
    # Test exception if a col in 'agg' does not exist in data.
    data = pDataFrame({"val": []})
    agg = {FIRST: ("val", FIRST)}
    bin_by = Grouper(freq="10T", closed="left", label="right", key="val")
    snap_by = Grouper(freq="5T", closed="left", key="val")
    none_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        snap_by=snap_by,
    )
    assert none_res is None


def test_exception_col_not_existing():
    # Test exception if a col in 'agg' does not exist in data.
    values = array([1.0], dtype=DTYPE_FLOAT64)
    dtidx = array(["2020-01-01T08:00"], dtype=DTYPE_DATETIME64)
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {FIRST: (f"{value}_", FIRST)}
    bin_by = Grouper(freq="10T", closed="left", label="right", key=dti)
    snap_by = Grouper(freq="5T", closed="left", key=dti)
    with pytest.raises(ValueError, match="^column 'value_' does not"):
        bins_res, snaps_res = cumsegagg(
            data=data,
            agg=agg,
            bin_by=bin_by,
            ordered_on=dti,
            snap_by=snap_by,
        )


def test_exception_error_on_0():
    # Test exception if there are 0 in aggregation results.
    values = array([0.0], dtype=DTYPE_FLOAT64)
    dtidx = array(["2020-01-01T08:00"], dtype=DTYPE_DATETIME64)
    value = "value"
    dti = "dti"
    data = pDataFrame({value: values, dti: dtidx})
    agg = {FIRST: (value, FIRST)}
    bin_by = Grouper(freq="10T", closed="left", label="right", key=dti)
    snap_by = Grouper(freq="5T", closed="left", key=dti, label="right")
    with pytest.raises(ValueError, match="^at least one null value exists in 'snap_res'"):
        bins_res, snaps_res = cumsegagg(
            data=data,
            agg=agg,
            bin_by=bin_by,
            ordered_on=dti,
            snap_by=snap_by,
        )
