#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

Test utils.
- Check pandas DataFrame equality:
from pandas.testing import assert_frame_equal
- Run pytest in iPython:
run -m pytest /home/yoh/Documents/code/oups/tests/test_aggstream/test_aggstream_simple.py
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
from pandas.core.resample import TimeGrouper

from oups import AggStream
from oups import ParquetSet
from oups import toplevel
from oups.aggstream.cumsegagg import DTYPE_NULLABLE_INT64
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
    # Test with 4 keys, no snapshots, parallel iterations.
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
        parallel=True,
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
        parallel=True,
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
        parallel=True,
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


def test_exception_different_indexes_at_restart(tmp_path):
    # Test exception at restart with 2 different 'seed_index_restart' for 2
    # different keys.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    #
    # Setup a 1st separate streamed aggregations (awkward...).
    max_row_group_size = 6
    ordered_on = "ts"
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
    seed_path = os_path.join(tmp_path, "seed")
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
