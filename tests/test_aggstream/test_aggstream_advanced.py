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
from pandas import merge_ordered
from pandas.core.resample import TimeGrouper

from oups import AggStream
from oups import ParquetSet
from oups import toplevel
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


@pytest.fixture
def store(tmp_path):
    # Reuse pre-defined Indexer.
    return ParquetSet(os_path.join(tmp_path, "store"), Indexer)


@pytest.fixture
def seed_path(tmp_path):
    return os_path.join(tmp_path, "seed")


# Setup keys.
ordered_on = "ts"
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


def test_3_keys_only_bins(store, seed_path):
    # Test with 3 keys, no snapshots, parallel iterations.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    # - key 3: 'by' as callable, every 4 rows, agg 'min', 'max',
    #
    # Setup streamed aggregation.
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


def test_2_keys_bins_snaps_filters(store, seed_path):
    # Test with 2 keys, bins and snapshots, filters and parallel iterations.
    # - filter 'True' : key 1: time grouper '10T', agg 'first' & 'last'.
    # - filter 'False' : key 2: time grouper '20T', agg 'first' & 'last'.
    # - snap '5T'
    # No head or tail trimming.
    #
    # Setup streamed aggregation.
    # Setup 'post'.
    def post(buffer: dict, bin_res: pDataFrame, snap_res: pDataFrame):
        """
        Aggregate previous and current bin aggregation results.

        Keep per row 'first' value of previous and current bin, and 'last' value from
        current snapshot.

        """
        # Retrieve previous results, concat to new, and shift.
        bin_res_last_ts = bin_res.loc[:, ordered_on].iloc[-1]
        snap_res_last_ts = snap_res.loc[:, ordered_on].iloc[-1]
        if buffer:
            # Not first iteration.
            # Specific shift.
            shifted_bin_res = pconcat([buffer["prev_bin_res"], bin_res]).drop_duplicates(
                subset=ordered_on,
                keep="last",
                ignore_index=True,
            )
            if len(shifted_bin_res) > len(bin_res):
                # Remove first row if it has added a new row,
                # to keep dataframe of same length
                shifted_bin_res = shifted_bin_res.iloc[: len(bin_res)].reset_index(drop=True)
            shifted_bin_res = shifted_bin_res.drop(columns=ordered_on).rename(
                columns={FIRST: "prev_first", LAST: "prev_last"},
            )
        else:
            shifted_bin_res = (
                bin_res.drop(ordered_on, axis=1)
                .shift(1)
                .rename(columns={FIRST: "prev_first", LAST: "prev_last"})
            )
        buffer["prev_bin_res"] = bin_res.iloc[-1:]
        # Align 'bin_res' on 2 columns.
        shifted_bin_res = pconcat([shifted_bin_res, bin_res], axis=1).rename(
            columns={FIRST: "current_first", LAST: "current_last"},
        )
        # Keep track of existing NA by filling with '-1' as others will be
        # created by 'merge_ordered' which have to be filled differently.
        merged_res = merge_ordered(shifted_bin_res.fillna(-1), snap_res.fillna(-1), on=ordered_on)
        merged_res = merged_res.astype(
            {
                "prev_first": DTYPE_NULLABLE_INT64,
                "current_first": DTYPE_NULLABLE_INT64,
                FIRST: DTYPE_NULLABLE_INT64,
                "prev_last": DTYPE_NULLABLE_INT64,
                "current_last": DTYPE_NULLABLE_INT64,
                LAST: DTYPE_NULLABLE_INT64,
            },
        )
        merged_res = merged_res.reindex(
            columns=[
                ordered_on,
                "prev_first",
                "current_first",
                FIRST,
                "prev_last",
                "current_last",
                LAST,
            ],
        )
        # Remove possibly created rows in 'snap_res' columns.
        merged_res = merged_res.dropna(subset=FIRST)
        if bin_res_last_ts < snap_res_last_ts:
            # In case 'snap_res' ends with several rows later than last row in
            # 'bin_res', last row from 'bin_res' has to be shifted by one
            # horizontally, before being forward filled.
            shifted_last_row = merged_res.set_index(ordered_on).loc[bin_res_last_ts]
            shifted_last_row = shifted_last_row.loc[
                ["prev_first", "current_first", FIRST, "prev_last", "current_last", LAST]
            ].shift(-1)
            merged_res.loc[
                len(merged_res) - 1,
                ["prev_first", "current_first", "prev_last", "current_last"],
            ] = shifted_last_row.loc[["prev_first", "current_first", "prev_last", "current_last"]]
        # Fill empty rows in 'bin_res' columns.
        # 'bin_res' columns.
        bin_res_cols = [
            "prev_first",
            "current_first",
            "prev_last",
            "current_last",
        ]
        merged_res.loc[:, bin_res_cols] = merged_res.loc[:, bin_res_cols].bfill()
        # In case of more bins than snaps, only keep snaps.
        merged_res = merged_res.set_index(ordered_on).loc[snap_res.loc[:, ordered_on]].reset_index()
        return merged_res.drop(columns=["current_first", "current_last"])

    val = "val"
    max_row_group_size = 5
    common_key_params = {
        "snap_by": TimeGrouper(key=ordered_on, freq="5T", closed="left", label="right"),
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
    filter1 = "filter1"
    filter2 = "filter2"
    filter_on = "filter_on"
    as_ = AggStream(
        ordered_on=ordered_on,
        store=store,
        keys={
            filter1: {key1: deepcopy(key1_cf)},
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
    start = Timestamp("2020/01/01")
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
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    filter_val = np.ones(len(ts), dtype=bool)
    filter_val[::2] = False
    seed_df = pDataFrame({ordered_on: ts, val: rand_ints, filter_on: filter_val})
    #       ts  val  filt row
    # 00:02:00,   2,   F,   0
    # 00:03:00,   3,   T,   1
    # 00:04:00,   4,   F,   2
    # 00:07:00,   7,   T,   3
    # 00:10:00,  10,   F,   4
    # 00:14:00,  14,   T,   5
    # 00:14:00,  14,   F,   6
    # 00:14:00,  14,   T,   7
    # 00:16:00,  16,   F,   8
    # 00:17:00,  17,   T,   9
    # 00:24:00,  24,   F,  10
    # 00:29:00,  29,   T,  11
    # 00:30:00,  30,   F,  12
    # 00:31:00,  31,   T,  13
    # 00:32:00,  32,   F,  14
    # 00:33:00,  33,   T,  15
    # 00:36:00,  36,   F,  16
    # 00:37:00,  37,   T,  17
    # 00:39:00,  39,   F,  18
    # 00:45:00,  45,   T,  19
    # 00:48:00,  48,   F,  20
    # 00:49:00,  49,   T,  21
    # 00:50:00,  50,   F,  22
    # 00:54:00,  54,   T,  23
    # 00:54:00,  54,   F,  24
    # 00:56:00,  56,   T,  25
    # 00:58:00,  58,   F,  26 --
    # 00:59:00,  59,   T,  27
    # 01:00:00,  60,   F,  28
    # 01:01:00,  61,   T,
    # 01:04:00,  64,   F,
    # 01:05:00,  65,   T,
    # 01:17:00,  77,   F,
    # 01:29:00,  89,   T,
    # 01:30:00,  90,   F,
    # 01:30:00,  90,   T,
    # 01:30:00,  90,   F,
    # 01:34:00,  94,   T,
    # 01:38:00,  98,   F,
    # 01:38:00,  98,   T,
    # 01:39:00,  99,   F,
    # 01:40:00, 100,   T,
    # 01:43:00, 103,   F,
    # 01:44:00, 104,   T,
    # 01:48:00, 108,   F,
    # 01:53:00, 113,   T,
    # 01:54:00, 114,   F,
    # 01:55:00, 115,   T,
    # 01:57:00, 117,   F,
    # 01:57:00, 117,   T,
    seed_list = [seed_df.loc[:27], seed_df.loc[27:]]
    #    as_.agg(seed=seed_df,
    as_.agg(
        seed=seed_list,
        trim_start=False,
        discard_last=False,
        final_write=True,
    )

    # Ref. results.
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
    # Seed data & streamed aggregation with a seed data of a single row,
    # at same timestamp than last one, not writing final results.
    seed_df = pDataFrame(
        {
            ordered_on: [ts[-1]],
            val: [rand_ints[-1] + 1],
            filter_on: [filter_val[-1]],
        },
    )
    seed_list.append(seed_df)
    as_.agg(
        seed=seed_df,
        trim_start=False,
        discard_last=False,
        final_write=False,
    )
    # Check that results on disk have not changed.
    key1_res = store[key1].pdf
    key2_res = store[key2].pdf
    assert key1_res.equals(key1_res_ref)
    assert key2_res.equals(key2_res_ref)

    # Again a single row with same timestamp: check how snapshot concatenate,
    # -> is not possible, need 2 rows of agg res to be added (last row removed
    # when no in 'infal_write')
    # because 1st row does not seem to be repeated.

    # Add a 3rd key with lower time freq than snap.

    # Test Aggstream init que 'snap_by' peut être communalisé.

    # Seed data & streamed aggregation with a seed data of several row,
    # writing final results.
