#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
import pytest
from numpy import NaN as nNaN
from numpy import all as nall
from numpy import array
from numpy import ndarray
from numpy import zeros
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import NaT as pNaT
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import date_range

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_FLOAT64
from oups.cumsegagg import DTYPE_INT64
from oups.cumsegagg import DTYPE_NULLABLE_INT64
from oups.cumsegagg import _next_chunk_starts
from oups.cumsegagg import bin_by_time
from oups.cumsegagg import cumsegagg
from oups.cumsegagg import setup_cgb_agg
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import LAST
from oups.jcumsegagg import MAX
from oups.jcumsegagg import MIN
from oups.jcumsegagg import SUM


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


def test_setup_cgb_agg():
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
        }
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
    cgb_agg_cfg_res = setup_cgb_agg(agg_cfg, df.dtypes.to_dict())
    cgb_agg_cfg_ref = {
        DTYPE_FLOAT64: [
            ["val1_float", "val2_float"],
            ["val1_first", "val2_first", "val2_sum"],
            array(
                [
                    [[0, 0], [1, 1]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[1, 2], [0, 0]],
                ],
                dtype=DTYPE_INT64,
            ),
            array([2, 0, 0, 0, 1], dtype=DTYPE_INT64),
        ],
        DTYPE_DATETIME64: [
            ["val4_datetime"],
            ["val4_first"],
            array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]], dtype=DTYPE_INT64),
            array([1, 0, 0, 0, 0], dtype=DTYPE_INT64),
        ],
        DTYPE_INT64: [
            ["val3_int"],
            ["val3_last", "val3_min", "val3_max"],
            array([[[0, 0]], [[0, 0]], [[0, 1]], [[0, 2]], [[0, 0]]], dtype=DTYPE_INT64),
            array([0, 1, 1, 1, 0], dtype=DTYPE_INT64),
        ],
    }
    for val_res, val_ref in zip(cgb_agg_cfg_res.values(), cgb_agg_cfg_ref.values()):
        for it_res, it_ref in zip(val_res, val_ref):
            if isinstance(it_res, ndarray):
                assert nall(it_res == it_ref)
            else:
                assert it_res == it_ref


@pytest.mark.parametrize(
    "data, right_edges, right, ref, n_null_chunks_ref",
    [
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      2  4  4  5  6
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            True,
            array([2, 4, 4, 5, 6], dtype=DTYPE_INT64),
            1,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      1  4  4  4  5
            array([2, 5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([1, 4, 4, 4, 5], dtype=DTYPE_INT64),
            2,
        ),
        (
            #      0  1  2  3  4  5  6
            array([1, 2, 3, 4, 7, 8, 9], dtype=DTYPE_INT64),
            #      7   7   7   7
            array([50, 60, 70, 88], dtype=DTYPE_INT64),
            False,
            array([7, 7, 7, 7], dtype=DTYPE_INT64),
            3,
        ),
        (
            #       0   1   2
            array([10, 22, 32], dtype=DTYPE_INT64),
            #      0  0  0  0
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 0, 0, 0], dtype=DTYPE_INT64),
            4,
        ),
        (
            #      0  1  2
            array([5, 5, 6], dtype=DTYPE_INT64),
            #      0  2  3  3
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 3, 3], dtype=DTYPE_INT64),
            2,
        ),
        (
            #      0  1  2  3  4
            array([5, 5, 7, 7, 7], dtype=DTYPE_INT64),
            #      0  2  2  5
            array([5, 6, 7, 8], dtype=DTYPE_INT64),
            False,
            array([0, 2, 2, 5], dtype=DTYPE_INT64),
            2,
        ),
    ],
)
def test_next_chunk_starts(data, right_edges, right, ref, n_null_chunks_ref):
    next_chunk_starts = zeros(len(right_edges), dtype=DTYPE_INT64)
    n_null_chunks = zeros(1, dtype=DTYPE_INT64)
    _next_chunk_starts(data, right_edges, right, next_chunk_starts, n_null_chunks)
    assert nall(ref == next_chunk_starts)
    assert n_null_chunks[0] == n_null_chunks_ref


@pytest.mark.parametrize(
    "bin_on, by, bin_labels_ref, next_chunk_starts_ref, n_null_bins_ref",
    [
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(["2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="left"),
            DatetimeIndex(
                ["2020-01-01 08:05:00", "2020-01-01 08:10:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([2, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="left"),
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(
                ["2020-01-01 07:55:00", "2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="left", closed="right"),
            DatetimeIndex(["2020-01-01 08:00:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:00"),
                    pTimestamp("2020/01/01 08:04"),
                    pTimestamp("2020/01/01 08:05"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="right"),
            DatetimeIndex(
                ["2020-01-01 08:00:00", "2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"
            ),
            array([1, 3], dtype=DTYPE_INT64),
            0,
        ),
        (
            Series(
                [
                    pTimestamp("2020/01/01 08:01"),
                    pTimestamp("2020/01/01 08:03"),
                    pTimestamp("2020/01/01 08:04"),
                ]
            ),
            Grouper(freq="5T", label="right", closed="right"),
            DatetimeIndex(["2020-01-01 08:05:00"], dtype=DTYPE_DATETIME64, freq="5T"),
            array([3], dtype=DTYPE_INT64),
            0,
        ),
    ],
)
def test_bin_by_time(bin_on, by, bin_labels_ref, next_chunk_starts_ref, n_null_bins_ref):
    bins, next_chunk_starts, n_null_bins = bin_by_time(bin_on, by)
    bin_labels = bins.left if by.label == "left" else bins.right
    assert nall(bin_labels == bin_labels_ref)
    assert nall(next_chunk_starts == next_chunk_starts_ref)
    assert n_null_bins == n_null_bins_ref


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
            }
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
            }
        )
        agg = {"res_first": ("col1_f", "first"), "res_last": ("col1_f", "last")}
    by = Grouper(freq="5T", closed="left", label="left", key=time_idx)
    bin_res = cumsegagg(data=data, agg=agg, bin_by=by)
    bin_res_ref = data.groupby(by).agg(**agg)
    assert bin_res.equals(bin_res_ref)


def test_cumsegagg_bin_mixed_dtype():
    # Test binning aggregation for a mixed dtype.
    ar_float = array(
        [[2.0, 20.0], [4.0, 40.0], [5.0, 50.0], [8.0, 80.0], [9.0, 90.0]], dtype=DTYPE_FLOAT64
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
        }
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
    "b_by_closed, b_by_label, s_by_closed, s_first_val, s_max_val, s_min_val, s_last_val, s_sum_qty, s_first_ts, null_b_dti, start_s_dti, null_s_dti",
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
            "left",
            # s1->s4    s5->s10      s11->s16    s17                           (first)
            [4.0] * 4 + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1->s4               s5->s10      s11->s16    s17                (max)
            [4.2, 6.0, 9.8, 9.8] + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1->s4    s5->s10      s11, s12     s13->s16    s17              (min)
            [3.9] * 4 + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4               s5->s10      s11, s12     s13->s16    s17   (last)
            [3.9, 6.0, 9.8, 9.8] + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4           s5->s10  s11,s12  s13->s16   s17                (sum)
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
        # 2/ bin right closed; right label, point of observations excluded
        #  This combination does not make that much sense.
        #  'data'
        #  datetime value  qty      snaps      bins   to check
        #                                             no 5mn snapshots empty
        #      8:10   4.0    4    s1-8:15   b1-8:30   aggregation over 2 values
        #      8:10   4.2    3    s1        b1
        #      8:12   3.9    1    s1        b1
        #      8:17   5.6    7    s2-8:20   b1        aggregation over 2 values
        #      8:19   6.0    2    s2        b1
        #      8:20   9.8    6    s3-8:25   b1        observation point is excluded
        #                         s4-8:30   b1        1 constant snapshot
        #                         s5-8:35   b2-9:00   6 empty snapshots
        #                               |   b2
        #                        s10-9:00   b2
        #      9:00   4.5    2              b2        1 filled bin
        #                        s11-9:05   b3-9:30   2 empty snapshots
        #                        s12-9:10   b3
        #      9:10   1.1    8   s13-9:15   b3
        #                        s14-9:20   b3        4 constant snapshots
        #                               |   b3
        #                        s16-9:30   b3
        #      9:30   3.2    1              b3
        #                        s17-9:35   (b4)      1 empty snapshot
        # Note on s17: snapshot 17 is created as per date 'range_logic'.
        # This logic is based on interval logic; The interval if left-closed.
        # '9:30' point is then part of this interval.
        # Because it is however also part of 'b3', then 'b4' is empty and 's17'
        # as well.
        (
            "right",
            "right",
            "left",
            # s1->s4    s5->s12      s13->s16    s17                           (first)
            [4.0] * 4 + [nNaN] * 8 + [1.1] * 4 + [nNaN],
            # s1->s4               s5->s12      s13->s16    s17                (max)
            [4.2, 6.0, 9.8, 9.8] + [nNaN] * 8 + [1.1] * 4 + [nNaN],
            # s1->s4    s5->s12      s13->s16    s17                           (min)
            [3.9] * 4 + [nNaN] * 8 + [1.1] * 4 + [nNaN],
            # s1->s4               s5->s12      s13->s16    s17                (last)
            [3.9, 6.0, 9.8, 9.8] + [nNaN] * 8 + [1.1] * 4 + [nNaN],
            # s1->s4           s5->s12  s13->s16  s17                          (sum)
            [8, 17, 23, 23] + [0] * 8 + [8] * 4 + [0],
            [pTimestamp("2020-01-01 08:10:00")] * 4
            + [pNaT] * 8
            + [pTimestamp("2020-01-01 09:10:00")] * 4
            + [pNaT],
            # null_bins_dti
            [],
            # start_s_dti
            "2020-01-01 08:15:00",
            # null_snaps_dti
            date_range("2020-01-01 08:35:00", periods=8, freq="5T").to_list()
            + [pTimestamp("2020-01-01 09:35:00")],
        ),
        # 3/ bin left closed; left label, point of observations included
        #  datetime value  qty      snaps      bins   to check
        #                                             no 5mn snapshots empty
        #      8:10   4.0    4    s1-8:10   b1-8:00   aggregation over 2 values
        #      8:10   4.2    3    s1        b1
        #      8:12   3.9    1    s2-8:15   b1
        #      8:17   5.6    7    s3-8:20   b1        aggregation over 2 values
        #      8:19   6.0    2    s3        b1
        #      8:20   9.8    6    s3        b1        observation point is included
        #                         s4-8:25   b1        1 constant snapshot
        #                         s5-8:30   b2-8:30   6 empty snapshots
        #                               |   b2
        #                        s10-8:55   b2
        #      9:00   4.5    2   s11:9:00   b3-9:00   2 constant snapshots
        #                        s12-9:05   b3
        #      9:10   1.1    8   s13-9:10   b3        4 constant snapshots
        #                        s14-9:15   b3
        #                               |   b3
        #                        s16-9:25   b3
        #      9:30   3.2    1   s17-9:30   b4-9:30
        (
            "left",
            "left",
            "right",
            # s1->s4    s5->s10      s11->s16    s17                           (first)
            [4.0] * 4 + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1->s4               s5->s10      s11->s16    s17                (max)
            [4.2, 4.2, 9.8, 9.8] + [nNaN] * 6 + [4.5] * 6 + [3.2],
            # s1    s2->s4      s5->s10      s11, s12     s13->s16    s17      (min)
            [4.0] + [3.9] * 3 + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4               s5->s10      s11, s12     s13->s16    s17   (last)
            [4.2, 3.9, 9.8, 9.8] + [nNaN] * 6 + [4.5, 4.5] + [1.1] * 4 + [3.2],
            # s1->s4         s5->s10   s11, s12 s13->s16   s17                 (sum)
            [7, 8, 23, 23] + [0] * 6 + [2, 2] + [10] * 4 + [1],
            [pTimestamp("2020-01-01 08:10:00")] * 4
            + [pNaT] * 6
            + [pTimestamp("2020-01-01 09:00:00")] * 6
            + [pTimestamp("2020-01-01 09:30:00")],
            # null_bins_dti
            [pTimestamp("2020-01-01 08:30:00")],
            # start_s_dti
            "2020-01-01 08:10:00",
            # null_snaps_dti
            date_range("2020-01-01 08:30:00", periods=6, freq="5T"),
        ),
        # 4/ bin right closed; left label, point of observations included
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
            "right",
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
def test_cumsegagg_bin_snaps_with_null_chunks(
    b_by_closed,
    b_by_label,
    s_by_closed,
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
    snap_by = Grouper(freq="5T", closed=s_by_closed, key=dti)
    bins_res, snaps_res = cumsegagg(
        data=data,
        agg=agg,
        bin_by=bin_by,
        ordered_on=dti,
        snap_by=snap_by,
        allow_bins_snaps_disalignment=True,
    )
    bins_ref = data.groupby(bin_by).agg(**agg)
    if null_b_dti:
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


# WiP
# create a test case testing snapshot and bin, similar to the one above, but this
# time starting right on the start of a bin, for instance with a point at 8:00
# and check all logics are ok.
#
# test exception: when bin are left closed or right closed, if observation point should be
# included or not.
#
# test bin & snapshot with empty bins / snaps
# test with a single bin and several snapshots
# test snapshot: as Grouper, as IntervalIndex
# segment(): test error message: 'by.closed' should either be 'right' or 'left', nothing else.
# Test with null values in agg_res (or modify test case above)
# Test error message if 'bin_on' is None in 'cumsegagg'.
# test error message input column in 'agg' not in input dataframe.
# test snapshot:
# with left-closed bin and snapshot included: snapshot come after bin
# with left-closed bin and snapsht excluded: snapshot come after bin
# Test error snap_by.key or snap_by.name not set
# WiP test case in case snap_by is IntevalIndex: its name has to be set after a column in data.
# WiP: test cases for 'bin_by_time'
# Wip test error: if bin_by and snap_by both a pd Grouper, check that both key parameter
# point on same column, otherwise, not possible possibly to compare
# Test case snapshot with a snapshot ending exactly on a bin end,
# and another not ending on a bin end.
# Snapshot test case with null rows (empty snapshots)
# Have a snapshot ending exactly on a bin end, then a next snapshot bigger than next bin

# Test exception when bin_on and bin_by.key not the same value
# Test sxeception
#   ordered_on not set but snap_by set
#   ordered_on different than snap_by.key
#   ordered_on not ordered
#   ordered_on different than bin_by.key when bin_by is a Grouper
# Test name of 'bin_res' index: ordered_on if ordered_on is set, else bin_on


# Test boh snapshot + bin, but in data there are
#  - only snapshots
#  - only bins
#  - a mix of them
# Make a test with empty bins
# Make a test with empty snapshots (to check all -1 in n_max_null_snap_indices are correctly removed)

# Todo Cumsegagg: remove "if" from the loop
