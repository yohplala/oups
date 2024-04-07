#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

Test utils.
- Check pandas DataFrame equality:
from pandas.testing import assert_frame_equal
- Run pytest in iPython:
run -m pytest /home/yoh/Documents/code/oups/tests/test_aggstream/test_aggstream_advanced.py
- Initialize store object & seed path:
tmp_path = os_path.expanduser('~/Documents/code/data/oups')
store = ParquetSet(os_path.join(tmp_path, "store"), Indexer)
seed_path = os_path.join(tmp_path, "seed")

"""
from copy import deepcopy
from os import path as os_path

import numpy as np
import pytest
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from pandas import DataFrame as pDataFrame
from pandas import NaT as pNaT
from pandas import Series as pSeries
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat
from pandas import date_range
from pandas import merge_ordered
from pandas.core.resample import TimeGrouper

from oups import AggStream
from oups import ParquetSet
from oups import toplevel
from oups.aggstream.aggstream import KEY_AGGSTREAM
from oups.aggstream.aggstream import KEY_RESTART_INDEX
from oups.aggstream.cumsegagg import DTYPE_NULLABLE_INT64
from oups.aggstream.cumsegagg import cumsegagg
from oups.aggstream.jcumsegagg import FIRST
from oups.aggstream.jcumsegagg import LAST
from oups.aggstream.jcumsegagg import MAX
from oups.aggstream.jcumsegagg import MIN
from oups.aggstream.segmentby import by_x_rows


@toplevel
class Indexer:
    dataset_ref: str


ordered_on = "ts"


@pytest.fixture
def store(tmp_path):
    # Reuse pre-defined Indexer.
    return ParquetSet(os_path.join(tmp_path, "store"), Indexer)


@pytest.fixture
def seed_path(tmp_path):
    return os_path.join(tmp_path, "seed")


def test_3_keys_only_bins(store, seed_path):
    # Test with 3 keys, no snapshots, parallel iterations.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    # - key 3: 'by' as callable, every 4 rows, agg 'min', 'max',
    #
    # Setup streamed aggregation.
    key1 = Indexer("agg_2T")
    key2 = Indexer("agg_13T")
    key3 = Indexer("agg_4rows")
    key1_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {FIRST: ("val", FIRST), LAST: ("val", LAST)},
    }
    key2_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {FIRST: ("val", FIRST), MAX: ("val", MAX)},
    }
    key3_cf = {
        "bin_by": by_x_rows,
        "agg": {MIN: ("val", MIN), MAX: ("val", MAX)},
    }
    max_row_group_size = 6
    key_configs = {
        key1: deepcopy(key1_cf),
        key2: deepcopy(key2_cf),
        key3: deepcopy(key3_cf),
    }
    as_ = AggStream(
        ordered_on=ordered_on,
        store=store,
        keys=key_configs,
        max_row_group_size=max_row_group_size,
        parallel=True,
    )
    # Seed data.
    start = Timestamp("2020/01/01")
    rr = np.random.default_rng(1)
    N = 24
    rand_ints = rr.integers(100, size=N)
    rand_ints.sort()
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    N_third = int(N / 3)
    bin_val = np.array(
        [1] * (N_third - 2) + [2] * (N_third - 4) + [3] * (N - 2 * N_third) + [4] * 6,
    )
    bin_on = "direct_bin"
    seed_df = pDataFrame({ordered_on: ts, "val": rand_ints, bin_on: bin_val})
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path).iter_row_groups()
    # Streamed aggregation.
    as_.agg(
        seed=seed,
        discard_last=True,
    )

    def get_ref_results(seed_df):
        # Get results
        key_configs = {
            key1: deepcopy(key1_cf),
            key2: deepcopy(key2_cf),
            key3: deepcopy(key3_cf),
        }
        k1_res = (
            seed_df.groupby(key_configs[key1]["bin_by"])
            .agg(**key_configs[key1]["agg"])
            .reset_index()
        )
        k1_res[FIRST] = k1_res[FIRST].astype(DTYPE_NULLABLE_INT64)
        k1_res[LAST] = k1_res[LAST].astype(DTYPE_NULLABLE_INT64)
        k2_res = (
            seed_df.groupby(key_configs[key2]["bin_by"])
            .agg(**key_configs[key2]["agg"])
            .reset_index()
        )
        key3_bins = by_x_rows(on=seed_df[ordered_on], buffer={})
        key3_bins = pSeries(pNaT, index=np.arange(len(seed_df)))
        key3_bin_starts = np.arange(0, len(seed_df), 4)
        key3_bins.iloc[key3_bin_starts] = seed_df.iloc[key3_bin_starts].loc[:, ordered_on]
        key3_bins.ffill(inplace=True)
        k3_res = seed_df.groupby(key3_bins).agg(**key_configs[key3]["agg"])
        k3_res.index.name = ordered_on
        k3_res.reset_index(inplace=True)
        return k1_res, k2_res, k3_res

    # Remove last 'group' as per 'ordered_on' in 'seed_df'.
    seed_df_trim = seed_df[seed_df[ordered_on] < seed_df[ordered_on].iloc[-1]]
    k1_res, k2_res, k3_res = get_ref_results(seed_df_trim)
    ref_res = {
        key1: k1_res,
        key2: k2_res,
        key3: k3_res,
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100, bin_on: bin_val + 10})
    fp_write(
        seed_path,
        seed_df2,
        row_group_offsets=max_row_group_size,
        file_scheme="hive",
        append=True,
    )
    seed = ParquetFile(seed_path).iter_row_groups()
    # Streamed aggregation.
    as_.agg(
        seed=seed,
        discard_last=True,
    )
    # Test results
    seed_df2_trim = seed_df2[seed_df2[ordered_on] < seed_df2[ordered_on].iloc[-1]]
    seed_df2_ref = pconcat([seed_df, seed_df2_trim], ignore_index=True)
    k1_res, k2_res, k3_res = get_ref_results(seed_df2_ref)
    ref_res = {
        key1: k1_res,
        key2: k2_res,
        key3: k3_res,
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)
    # 2nd append of new data.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400, bin_on: bin_val + 40})
    fp_write(
        seed_path,
        seed_df3,
        row_group_offsets=max_row_group_size,
        file_scheme="hive",
        append=True,
    )
    seed = ParquetFile(seed_path).iter_row_groups()
    # Streamed aggregation.
    as_.agg(
        seed=seed,
        discard_last=True,
    )
    # Test results
    seed_df3_trim = seed_df3[seed_df3[ordered_on] < seed_df3[ordered_on].iloc[-1]]
    seed_df3_ref = pconcat([seed_df, seed_df2, seed_df3_trim], ignore_index=True)
    k1_res, k2_res, k3_res = get_ref_results(seed_df3_ref)
    ref_res = {
        key1: k1_res,
        key2: k2_res,
        key3: k3_res,
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)


def test_exception_different_indexes_at_restart(store, seed_path):
    # Test exception at restart with 2 different 'seed_index_restart' for 2
    # different keys.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    #
    # Setup a 1st separate streamed aggregations (awkward...).
    key1 = Indexer("agg_2T")
    key1_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {FIRST: ("val", FIRST), LAST: ("val", LAST)},
    }
    max_row_group_size = 6
    as1 = AggStream(
        ordered_on=ordered_on,
        store=store,
        keys={key1: deepcopy(key1_cf)},
        max_row_group_size=max_row_group_size,
    )
    # Seed data.
    start = Timestamp("2020/01/01")
    rr = np.random.default_rng(1)
    N = 10
    rand_ints = rr.integers(100, size=N)
    rand_ints.sort()
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df = pDataFrame({ordered_on: ts, "val": rand_ints})
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path).iter_row_groups()
    # Streamed aggregation for 'key1'.
    as1.agg(
        seed=seed,
        discard_last=True,
    )
    # Setup a 2nd separate streamed aggregation.
    key2 = Indexer("agg_13T")
    key2_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {FIRST: ("val", FIRST), MAX: ("val", MAX)},
    }
    as2 = AggStream(
        ordered_on=ordered_on,
        store=store,
        keys={key2: deepcopy(key2_cf)},
        max_row_group_size=max_row_group_size,
    )
    # Extend seed.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100})
    fp_write(
        seed_path,
        seed_df2,
        row_group_offsets=max_row_group_size,
        file_scheme="hive",
        append=True,
    )
    seed = ParquetFile(seed_path).iter_row_groups()
    # Streamagg for 'key2'.
    as2.agg(
        seed=seed,
        discard_last=True,
    )
    # Now a streamed aggregation for both keys.
    with pytest.raises(
        ValueError,
        match="^not possible to aggregate on multiple keys with existing",
    ):
        AggStream(
            ordered_on=ordered_on,
            store=store,
            keys={key1: deepcopy(key1_cf), key2: deepcopy(key2_cf)},
            max_row_group_size=max_row_group_size,
        )


def test_3_keys_bins_snaps_filters(store, seed_path):
    # Test with 3 keys, bins and snapshots, filters and parallel iterations.
    # - filter 'True' : key 1: time grouper '10T', agg 'first' & 'last'.
    # - filter 'False' : key 2: time grouper '20T', agg 'first' & 'last'.
    # - filter 'True' : key 3: time grouper '2T', agg 'first' & 'last'.
    # - snap '5T'
    # No head or tail trimming.
    #
    # Setup 'post'.
    def post(buffer: dict, bin_res: pDataFrame, snap_res: pDataFrame):
        """
        Aggregate previous and current bin aggregation results.

        Keep per row 'first' value of previous and current bin, and 'last'
        value from current snapshot.

        Notes
        -----
          - This function has been defined by using 'label' parameter for
            'bin_by' and 'snap_by' set to 'right'.
          - This function has been decently tested and optimized by this very
            test case...

        """
        bin_res_last_ts = bin_res.loc[:, ordered_on].iloc[-1]
        snap_res_last_ts = snap_res.loc[:, ordered_on].iloc[-1]
        if buffer:
            # Retrieve previous results, concat to new, and shift.
            if (
                buffer["prev_bin_res"].loc[:, ordered_on].iloc[-1]
                == bin_res.loc[:, ordered_on].iloc[0]
            ):
                prev_bin_res = buffer["prev_bin_res"].iloc[:-1]
            else:
                prev_bin_res = buffer["prev_bin_res"].iloc[1:]
            shifted_bin_res = pconcat([prev_bin_res, bin_res], ignore_index=True)
            # Forced to keep one row more in case the last row is a duplicate
            # index with 1st row of next iteration.
            buffer["prev_bin_res"] = shifted_bin_res.iloc[-2:]
            # Remove row added by past concat.
            shifted_bin_res = shifted_bin_res.iloc[:-1]
        else:
            # 1st iteration.
            # Forced to keep one row more in case the last row is a duplicate
            # index with 1st row of next iteration.
            buffer["prev_bin_res"] = bin_res.iloc[-2:]
            shifted_bin_res = bin_res.shift(1)
        # Align 'bin_res' on 2 columns.
        # Keep track of existing NA by filling with '-1' as others will be
        # created by 'merge_ordered' which have to be managed differently.
        shifted_bin_res = (
            pconcat(
                [
                    shifted_bin_res.drop(ordered_on, axis=1).rename(
                        columns={FIRST: "prev_first", LAST: "prev_last"},
                    ),
                    bin_res,
                ],
                axis=1,
            )
            .rename(
                columns={FIRST: "current_first", LAST: "current_last"},
            )
            .fillna(-1)
        )
        if bin_res_last_ts < snap_res_last_ts:
            # Shift horizontally last row and keep it apart for reuse later.
            hshifted_bin_res_last_row = (
                shifted_bin_res.set_index(ordered_on)
                .loc[bin_res_last_ts]
                .reindex(
                    [
                        "prev_first",
                        "current_first",
                        "prev_last",
                        "current_last",
                    ],
                )
                .shift(-1)
                .loc[["prev_first", "prev_last"]]
            )
        merged_res = merge_ordered(
            shifted_bin_res.loc[:, [ordered_on, "prev_first", "prev_last"]],
            snap_res.fillna(-1),
            on=ordered_on,
        )
        merged_res = merged_res.astype(
            {
                "prev_first": DTYPE_NULLABLE_INT64,
                FIRST: DTYPE_NULLABLE_INT64,
                "prev_last": DTYPE_NULLABLE_INT64,
                LAST: DTYPE_NULLABLE_INT64,
            },
        )
        # This reindex is essentially for pretty printing when debugging.
        merged_res = merged_res.reindex(
            columns=[ordered_on, "prev_first", FIRST, "prev_last", LAST],
        )
        # 'bin_res' columns.
        bin_res_cols = ["prev_first", "prev_last"]
        if bin_res_last_ts < snap_res_last_ts:
            # In case 'snap_res' ends with several rows later than last row in
            # 'bin_res', last values have to be derived from 'bin_res' last row
            # that are then shifted by one horizontally, before being forward
            # filled.
            merged_res.loc[len(merged_res) - 1, bin_res_cols] = hshifted_bin_res_last_row
        # Fill empty rows in 'bin_res' columns.
        merged_res.loc[:, bin_res_cols] = merged_res.loc[:, bin_res_cols].bfill()
        # Remove possibly created rows in 'snap_res' columns.
        # be it because more bins than snaps, or snaps indexes with different
        # values than bin indexes.
        return merged_res.dropna(subset=FIRST, ignore_index=True)

    # Setup streamed aggregation.
    val = "val"
    max_row_group_size = 5
    snap_duration = "5T"
    common_key_params = {
        "snap_by": TimeGrouper(key=ordered_on, freq=snap_duration, closed="left", label="right"),
        "agg": {FIRST: (val, FIRST), LAST: (val, LAST)},
    }
    key1 = Indexer("agg_10T")
    key1_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="10T", closed="left", label="right"),
    }
    key2 = Indexer("agg_20T")
    key2_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="20T", closed="left", label="right"),
    }
    key3 = Indexer("agg_2T")
    key3_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="2T", closed="left", label="right"),
    }
    filter1 = "filter1"
    filter2 = "filter2"
    filter_on = "filter_on"
    as_ = AggStream(
        ordered_on=ordered_on,
        store=store,
        keys={
            filter1: {
                key1: deepcopy(key1_cf),
                key3: deepcopy(key3_cf),
            },
            filter2: {key2: deepcopy(key2_cf)},
        },
        filters={
            filter1: [(filter_on, "==", True)],
            filter2: [(filter_on, "==", False)],
        },
        max_row_group_size=max_row_group_size,
        **common_key_params,
        parallel=True,
        post=post,
    )
    # Seed data & streamed aggregation over a list of 2 seed chunks.
    #    rr = np.random.default_rng(1)
    #    N = 50
    #    rand_ints = rr.integers(120, size=N)
    #    rand_ints.sort()
    rand_ints = np.hstack(
        [
            np.array([2, 3, 4, 7, 10, 14, 14, 14, 16, 17]),
            np.array([24, 29, 30, 31, 32, 33, 36, 37, 39, 45]),
            np.array([48, 49, 50, 54, 54, 56, 58, 59, 60, 61]),
            np.array([64, 65, 77, 89, 90, 90, 90, 94, 98, 98]),
            np.array([99, 100, 103, 104, 108, 113, 114, 115, 117, 117]),
        ],
    )
    start = Timestamp("2020/01/01")
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    filter_val = np.ones(len(ts), dtype=bool)
    filter_val[::2] = False
    seed_df = pDataFrame({ordered_on: ts, val: rand_ints, filter_on: filter_val})
    #
    # filter = 'True'
    #     ts  val  filt row     2T_agg_res
    #                       pr1st   1st plast last
    #  00:03,   3,   T,   1
    # (00:05)                   3,   -1,    3,  -1
    #  00:07,   7,   T,   3
    # (00:10)                   7,   -1,    7,  -1
    #  00:14,  14,   T,   5
    #  00:14,  14,   T,   7
    # (00:15)                  -1,   14,   -1,  14
    #  00:17,  17,   T,   9
    # (00:20)                  17,   -1,   17,  -1
    # (00:25)                  -1,   -1,   -1,  -1
    #  00:29,  29,   T,  11
    # (00:30)                  -1,  29,    -1,  29
    #  00:31,  31,   T,  13
    #  00:33,  33,   T,  15
    # (00:35)                  33,  -1,    33,  -1
    #  00:37,  37,   T,  17
    # (00:40)                  37,  -1,    37,  -1
    # (00:45)                  -1,  -1,    -1,  -1
    #  00:45,  45,   T,  19
    #  00:49,  49,   T,  21
    # (00:50)                  -1,  49,    -1,  49
    #  00:54,  54,   T,  23
    # (00:55)                  -1,  54,    -1,  54
    #  00:56,  56,   T,  25 --
    #  00:59,  59,   T,  27
    # (01:00)                  56,  59,    56,  59
    #  01:01,  61,   T,  29
    # (01:05)                  -1,  -1,    -1,  -1
    #  01:05,  65,   T,  31
    # (01:10)                  -1,  -1,    -1,  -1
    # (01:15)                  -1,  -1,    -1,  -1
    # (01:20)                  -1,  -1,    -1,  -1
    # (01:25)                  -1,  -1,    -1,  -1
    #  01:29,  89,   T,  33
    # (01:30)                  -1,  89,    -1,  89
    #  01:30,  90,   T,  35
    #  01:34,  94,   T,  37
    # (01:35)                  -1,  94,    -1,  94
    #  01:38,  98,   T,  39
    # (01:40)                  -1,  98,    -1,  98
    #  01:40, 100,   T,  41
    #  01:44, 104,   T,  43
    # (01:45)                  -1, 104,    -1, 104
    # (01:50)                  -1,  -1,    -1,  -1
    #  01:53, 113,   T,  45
    # (01:55)                 113,  -1,   113,  -1
    #  01:55, 115,   T,  47
    #  01:57, 117,   T,  49
    # (02:00)                 117,  -1,   117,  -1
    #
    # filter = 'False'
    #     ts  val  filt row
    #                       pr1st  1st plast last
    # 00:02,   2,   F,   0
    # 00:04,   4,   F,   2
    # 00:10,  10,   F,   4
    # 00:14,  14,   F,   6
    # 00:16,  16,   F,   8
    # 00:24,  24,   F,  10
    # 00:30,  30,   F,  12
    # 00:32,  32,   F,  14
    # 00:36,  36,   F,  16
    # 00:39,  39,   F,  18
    # 00:48,  48,   F,  20
    # 00:50,  50,   F,  22
    # 00:54,  54,   F,  24
    # 00:58,  58,   F,  26 --
    # 01:00,  60,   F,  28
    # 01:04,  64,   F,  30
    # 01:17,  77,   F,  32
    # 01:30,  90,   F,  34
    # 01:30,  90,   F,  36
    # 01:38,  98,   F,  38
    # 01:39,  99,   F,  40
    # 01:43, 103,   F,  42
    # 01:48, 108,   F,  44
    # 01:54, 114,   F,  46
    # 01:57, 117,   F,  48
    seed_list = [seed_df.loc[:27], seed_df.loc[27:]]
    # --------------#
    # Data stream 1 #
    # --------------#
    as_.agg(
        seed=seed_list,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )
    # Ref. results for key3 hard recorded, in case of a bug in 'post'
    # propagating between streamed aggregation and continuous aggregation.
    snap_ts = date_range(
        start="2020-01-01 00:05:00",
        end="2020-01-01 02:00:00",
        freq="5T",
    )
    key3_data = np.array(
        [
            [3, -1, 3, -1],
            [7, -1, 7, -1],
            [-1, 14, -1, 14],
            [17, -1, 17, -1],
            [-1, -1, -1, -1],
            [-1, 29, -1, 29],
            [33, -1, 33, -1],
            [37, -1, 37, -1],
            [-1, -1, -1, -1],
            [-1, 49, -1, 49],
            [-1, 54, -1, 54],
            [56, 59, 56, 59],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, 89, -1, 89],
            [-1, 94, -1, 94],
            [-1, 98, -1, 98],
            [-1, 104, -1, 104],
            [-1, -1, -1, -1],
            [113, -1, 113, -1],
            [117, -1, 117, -1],
        ],
    )
    key3_res_ref = pDataFrame(key3_data, index=snap_ts).reset_index()
    key3_res_ref = key3_res_ref.rename(
        columns={
            "index": ordered_on,
            0: "prev_first",
            1: FIRST,
            2: "prev_last",
            3: LAST,
        },
    ).astype(
        {
            "prev_first": DTYPE_NULLABLE_INT64,
            FIRST: DTYPE_NULLABLE_INT64,
            "prev_last": DTYPE_NULLABLE_INT64,
            LAST: DTYPE_NULLABLE_INT64,
        },
    )

    # Ref. results by continuous aggregation for key1 and key2.
    def reference_results(seed: pDataFrame, key_conf: dict):
        """
        Get reference results from cumsegagg and post.
        """
        bin_res_ref, snap_res_ref = cumsegagg(
            data=seed,
            **key_conf,
            ordered_on=ordered_on,
        )
        return post({}, bin_res_ref.reset_index(), snap_res_ref.reset_index())

    key1_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key1_cf | common_key_params,
    )
    key2_res_ref = reference_results(
        seed_df.loc[~seed_df[filter_on], :],
        key2_cf | common_key_params,
    )
    # --------------#
    # Data stream 2 #
    # --------------#
    # Seed data & streamed aggregation, not writing final results, with a seed
    # data of two rows in 2 different snaps,
    # - one at same timestamp than last one.
    # - one at a new timestamp. This one will not be considered because when
    #   not writing final results, last row in agg res is set aside.
    seed_df = pDataFrame(
        {
            ordered_on: [ts[-1], ts[-1] + Timedelta(snap_duration)],
            val: [rand_ints[-1] + 1, rand_ints[-1] + 10],
            filter_on: [True] * 2,
        },
    )
    as_.agg(
        seed=seed_df,
        trim_start=False,
        discard_last=False,
        final_write=False,
    )
    # Check that results on disk have not changed.
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
    # --------------#
    # Data stream 3 #
    # --------------#
    # Write and check 'last_seed_index' has been correctly updated even for
    # 'key2'.
    as_.agg(
        seed=None,
        final_write=True,
    )
    # Reference results for 'key1' and 'key3'.
    seed_list.append(seed_df)
    seed_df = pconcat(seed_list)
    key1_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key1_cf | common_key_params,
    )
    key3_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key3_cf | common_key_params,
    )
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
    # For 'key2', results have not changed, as seed data has been filtered out.
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    # Even if no new seed data for key2, check that "last_seed_index" has been
    # updated.
    assert (
        store[key2]._oups_metadata[KEY_AGGSTREAM][KEY_RESTART_INDEX] == seed_df[ordered_on].iloc[-1]
    )
    # --------------#
    # Data stream 4 #
    # --------------#
    # Last data appending considering a single row in seed with same timestamp
    # and 'final_write' as last concatenation check with snapshots.
    seed_df = pDataFrame(
        {
            ordered_on: [seed_df.loc[:, ordered_on].iloc[-1]],
            val: [rand_ints[-1] + 50],
            filter_on: [True],
        },
    )
    as_.agg(
        seed=seed_df,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )
    # Reference results.
    seed_list.append(seed_df)
    seed_df = pconcat(seed_list)
    key1_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key1_cf | common_key_params,
    )
    key3_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key3_cf | common_key_params,
    )
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
    # --------------#
    # Data stream 5 #
    # --------------#
    # Empty snapshots are generated between 2 row groups in key2.
    seed_df = pDataFrame(
        {
            ordered_on: [
                seed_df.loc[:, ordered_on].iloc[-1],
                seed_df.loc[:, ordered_on].iloc[-1] + 10 * Timedelta(snap_duration),
            ],
            val: [rand_ints[-1] + 50, rand_ints[-1] + 100],
            filter_on: [False] * 2,
        },
    )
    seed = [seed_df.iloc[:1], seed_df.iloc[1:]]
    as_.agg(
        seed=seed,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )
    # Reference results.
    seed_list.extend(seed)
    seed_df = pconcat(seed_list)
    key2_res_ref = reference_results(
        seed_df.loc[~seed_df[filter_on], :],
        key2_cf | common_key_params,
    )
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
    # Even if no new seed data for key1 & key3, check that "last_seed_index"
    # has been updated.
    assert (
        store[key1]._oups_metadata[KEY_AGGSTREAM][KEY_RESTART_INDEX] == seed_df[ordered_on].iloc[-1]
    )
    assert (
        store[key3]._oups_metadata[KEY_AGGSTREAM][KEY_RESTART_INDEX] == seed_df[ordered_on].iloc[-1]
    )
    # --------------#
    # Data stream 6 #
    # --------------#
    # Several seed chunks where neither bins, nor snaps end.
    seed_df = pDataFrame(
        {
            ordered_on: [
                seed_df.loc[:, ordered_on].iloc[-1],
                seed_df.loc[:, ordered_on].iloc[-1],
                seed_df.loc[:, ordered_on].iloc[-1],
            ],
            val: [
                seed_df.loc[:, val].iloc[-1] + 10,
                seed_df.loc[:, val].iloc[-1] + 20,
                seed_df.loc[:, val].iloc[-1] + 30,
            ],
            filter_on: [True] * 3,
        },
    )
    seed = [seed_df.iloc[:1], seed_df.iloc[1:2], seed_df.iloc[2:]]
    as_.agg(
        seed=seed,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )
    # Reference results.
    seed_list.extend(seed)
    seed_df = pconcat(seed_list)
    key1_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key1_cf | common_key_params,
    )
    key3_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key3_cf | common_key_params,
    )
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
    # --------------#
    # Data stream 7 #
    # --------------#
    # 2 1st bins (10T) and some snapshot (5T) start right on a new segment.
    rand_ints = np.array([1, 3, 4, 7, 10, 14, 14, 14, 16, 17])
    start = Timestamp("2020/01/01 03:00:00")
    ts = [start] + [start + Timedelta(f"{mn}T") for mn in rand_ints[1:]]
    seed = pDataFrame({ordered_on: ts, val: rand_ints, filter_on: [True] * len(ts)})
    seed = [seed.iloc[:4], seed.iloc[4:]]
    as_.agg(
        seed=seed,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )
    seed_list.extend(seed)
    seed_df = pconcat(seed_list)
    key1_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key1_cf | common_key_params,
    )
    key3_res_ref = reference_results(
        seed_df.loc[seed_df[filter_on], :],
        key3_cf | common_key_params,
    )
    key1_res = store[key1].pdf
    assert key1_res.equals(key1_res_ref)
    key2_res = store[key2].pdf
    assert key2_res.equals(key2_res_ref)
    key3_res = store[key3].pdf
    assert key3_res.equals(key3_res_ref)
