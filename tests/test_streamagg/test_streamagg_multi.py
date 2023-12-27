#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

"""
from os import path as os_path

import numpy as np
import pytest
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import NaT as pNaT
from pandas import Series as pSeries
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat
from pandas.core.resample import TimeGrouper

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.store.writer import MAX_ROW_GROUP_SIZE
from oups.streamagg.jcumsegagg import FIRST
from oups.streamagg.jcumsegagg import LAST
from oups.streamagg.segmentby import by_x_rows
from oups.streamagg.streamagg import _setup


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class Indexer:
    dataset_ref: str


def test_setup_4_keys_with_default_parameters_for_writing(tmp_path):
    # Test that config for each key and for seed is correctly consolidated,
    # with default values when parameters are not specified at key level.
    # 'max_row_group_size' and 'max_nirgs' are default parameters for writing.
    # 'post' also has a default value.
    # Key 1 has mostly parameters defined by default values.
    #       'by' is a pd.TimeGrouper.
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    ordered_on_dflt = "ts_dflt"
    bin_on_spec = "bin_on_spec"
    bin_out_spec = "bin_out_spec"
    in_spec = "in_spec"
    in_dflt = "in_dflt"
    key1 = Indexer("some_default")
    key2 = Indexer("only_specific")
    key3 = Indexer("only_default1")
    key4 = Indexer("only_default2")
    tgrouper = TimeGrouper(key=ordered_on_dflt, freq="1H")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    dummy_dti = DatetimeIndex(["01/01/2020 08:00", "01/01/2020 09:00"])
    in_spec_val = [1, 2]
    seed_df = pDataFrame(
        {
            ordered_on_dflt: dummy_dti,
            bin_on_spec: dummy_dti,
            in_spec: in_spec_val,
            in_dflt: in_spec_val,
        },
    )

    def dummy_by(**kwargs):
        # Dummy function for key 2.
        return True

    def dummy_post_spec(**kwargs):
        # Dummy function for key 1.
        return True

    def dummy_post_dflt(**kwargs):
        # Dummy function for key 1.
        return True

    keys_config = {
        key1: {
            "agg": {"out_spec": (in_spec, FIRST)},
            "bin_by": tgrouper,
            "post": dummy_post_spec,
        },
        key2: {
            "agg": {"out_spec": (in_spec, FIRST)},
            "bin_by": dummy_by,
            "post": None,
            "max_row_group_size": 3000,
        },
        key3: {"bin_by": dummy_by, "bin_on": (bin_on_spec, bin_out_spec)},
        key4: {"bin_by": tgrouper},
    }
    trim_start = True
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on_dflt,
        "trim_start": trim_start,
        "agg": {"out_dflt": (in_dflt, LAST)},
        "post": dummy_post_dflt,
        "max_row_group_size": 1000,
        "max_nirgs": 4,
        "seed_dtypes": seed_df.dtypes.to_dict(),
    }
    # Test.
    (
        all_cols_in_res,
        trim_start,
        seed_index_restart_set,
        keys_config_res,
    ) = _setup(**parameter_in)
    # Reference results.
    keys_config_ref = {
        str(key1): {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "dirpath": os_path.join(store.basepath, key1.to_path),
            "bin_on_out": ordered_on_dflt,
            "post": dummy_post_spec,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key2): {
            "dirpath": os_path.join(store.basepath, key2.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": None,
            "post": None,
            "write_config": {
                "max_row_group_size": 3000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key3): {
            "dirpath": os_path.join(store.basepath, key3.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": bin_out_spec,
            "post": dummy_post_dflt,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_out_spec,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key4): {
            "dirpath": os_path.join(store.basepath, key4.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": ordered_on_dflt,
            "post": dummy_post_dflt,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
    }
    # Check.
    # (do not check segmentation and aggregation configs)
    del keys_config_res[str(key1)]["seg_config"]
    del keys_config_res[str(key1)]["agg_config"]
    assert keys_config_ref[str(key1)] == keys_config_res[str(key1)]
    del keys_config_res[str(key2)]["seg_config"]
    del keys_config_res[str(key2)]["agg_config"]
    assert keys_config_ref[str(key2)] == keys_config_res[str(key2)]
    del keys_config_res[str(key3)]["seg_config"]
    del keys_config_res[str(key3)]["agg_config"]
    assert keys_config_ref[str(key3)] == keys_config_res[str(key3)]
    del keys_config_res[str(key4)]["seg_config"]
    del keys_config_res[str(key4)]["agg_config"]
    assert keys_config_ref[str(key4)] == keys_config_res[str(key4)]
    all_cols_in_ref = {"in_spec", ordered_on_dflt, bin_on_spec, "in_dflt"}
    assert set(all_cols_in_res) == all_cols_in_ref
    assert not trim_start
    assert not seed_index_restart_set


def test_setup_4_keys_wo_default_parameters_for_writing_nor_post(tmp_path):
    # Test config for each key and for seed is correctly consolidated,
    # with default values when parameters are not specified at key level.
    # 'max_row_group_size' and 'max_nirgs' are not provided for writing.
    # 'post' also has no default value either.
    # Key 1 has mostly parameters defined by default values.
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    ordered_on_dflt = "ts_dflt"
    bin_on_spec = "bin_on_spec"
    bin_out_spec = "bin_out_spec"
    in_spec = "in_spec"
    in_dflt = "in_dflt"
    key1 = Indexer("some_default")
    key2 = Indexer("only_specific")
    key3 = Indexer("only_default1")
    key4 = Indexer("only_default2")
    tgrouper = TimeGrouper(key=ordered_on_dflt, freq="1H")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    dummy_dti = DatetimeIndex(["01/01/2020 08:00", "01/01/2020 09:00"])
    in_spec_val = [1, 2]
    seed_df = pDataFrame(
        {
            ordered_on_dflt: dummy_dti,
            bin_on_spec: dummy_dti,
            in_spec: in_spec_val,
            in_dflt: in_spec_val,
        },
    )

    def dummy_by(**kwargs):
        # Dummy function for key 2.
        return True

    def dummy_post_spec(**kwargs):
        # Dummy function for key 1.
        return True

    def dummy_post_dflt(**kwargs):
        # Dummy function for key 1.
        return True

    keys_config = {
        key1: {
            "agg": {"out_spec": ("in_spec", "first")},
            "bin_by": tgrouper,
            "post": dummy_post_spec,
        },
        key2: {
            "agg": {"out_spec": ("in_spec", "first")},
            "bin_by": dummy_by,
            "max_row_group_size": 3000,
        },
        key3: {"bin_by": dummy_by, "bin_on": (bin_on_spec, bin_out_spec)},
        key4: {"bin_by": tgrouper},
    }
    trim_start = True
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on_dflt,
        "trim_start": trim_start,
        "agg": {"out_dflt": ("in_dflt", "last")},
        "post": None,
        "seed_dtypes": seed_df.dtypes.to_dict(),
    }
    # Test.
    (
        all_cols_in_res,
        trim_start,
        seed_index_restart_set,
        keys_config_res,
    ) = _setup(**parameter_in)
    # Reference results.
    keys_config_ref = {
        str(key1): {
            "dirpath": os_path.join(store.basepath, key1.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": ordered_on_dflt,
            "post": dummy_post_spec,
            "write_config": {
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
                "max_row_group_size": MAX_ROW_GROUP_SIZE,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key2): {
            "dirpath": os_path.join(store.basepath, key2.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": None,
            "post": None,
            "write_config": {
                "max_row_group_size": 3000,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key3): {
            "dirpath": os_path.join(store.basepath, key3.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": bin_out_spec,
            "post": None,
            "write_config": {
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_out_spec,
                "max_row_group_size": MAX_ROW_GROUP_SIZE,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
        str(key4): {
            "dirpath": os_path.join(store.basepath, key4.to_path),
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "bin_res": None,
            "bin_on_out": ordered_on_dflt,
            "post": None,
            "write_config": {
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
                "max_row_group_size": MAX_ROW_GROUP_SIZE,
            },
            "segagg_buffer": {},
            "post_buffer": {},
            "agg_res_buffer": [],
            "bin_res_buffer": [],
        },
    }
    # Check.
    del keys_config_res[str(key1)]["seg_config"]
    del keys_config_res[str(key1)]["agg_config"]
    assert keys_config_ref[str(key1)] == keys_config_res[str(key1)]
    del keys_config_res[str(key2)]["seg_config"]
    del keys_config_res[str(key2)]["agg_config"]
    assert keys_config_ref[str(key2)] == keys_config_res[str(key2)]
    del keys_config_res[str(key3)]["seg_config"]
    del keys_config_res[str(key3)]["agg_config"]
    assert keys_config_ref[str(key3)] == keys_config_res[str(key3)]
    del keys_config_res[str(key4)]["seg_config"]
    del keys_config_res[str(key4)]["agg_config"]
    assert keys_config_ref[str(key4)] == keys_config_res[str(key4)]
    all_cols_in_ref = {"in_spec", ordered_on_dflt, bin_on_spec, "in_dflt"}
    assert set(all_cols_in_res) == all_cols_in_ref
    assert not trim_start
    assert not seed_index_restart_set


def test_parquet_seed_3_keys(tmp_path):
    # Test with parquet seed, and 4 keys.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    # - key 3: 'by' as callable, every 4 rows, agg 'min', 'max',
    max_row_group_size = 6
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
    ordered_on = "ts"
    bin_on = "direct_bin"
    seed_df = pDataFrame({ordered_on: ts, "val": rand_ints, bin_on: bin_val})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key1 = Indexer("agg_2T")
    key2 = Indexer("agg_13T")
    key3 = Indexer("agg_4rows")

    # Setup aggregation.
    key1_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "last": ("val", "last")},
    }
    key2_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "max": ("val", "max")},
    }
    key3_cf = {
        "bin_by": by_x_rows,
        "agg": {"min": ("val", "min"), "max": ("val", "max")},
    }
    key_configs = {key1: key1_cf, key2: key2_cf, key3: key3_cf}
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        keys=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )

    def get_results(seed_df):
        # Get results
        k1_res = (
            seed_df.groupby(key_configs[key1]["bin_by"])
            .agg(**key_configs[key1]["agg"])
            .reset_index()
        )
        k1_res["first"] = k1_res["first"].astype("Int64")
        k1_res["last"] = k1_res["last"].astype("Int64")
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
    k1_res, k2_res, k3_res = get_results(seed_df_trim)
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
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        keys=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    seed_df2_trim = seed_df2[seed_df2[ordered_on] < seed_df2[ordered_on].iloc[-1]]
    seed_df2_ref = pconcat([seed_df, seed_df2_trim], ignore_index=True)
    k1_res, k2_res, k3_res = get_results(seed_df2_ref)
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
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        keys=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    seed_df3_trim = seed_df3[seed_df3[ordered_on] < seed_df3[ordered_on].iloc[-1]]
    seed_df3_ref = pconcat([seed_df, seed_df2, seed_df3_trim], ignore_index=True)
    k1_res, k2_res, k3_res = get_results(seed_df3_ref)
    ref_res = {
        key1: k1_res,
        key2: k2_res,
        key3: k3_res,
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)


def test_exception_setup_no_bin_by(tmp_path):
    ordered_on = "ts"
    key = Indexer("agg_res")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    keys_config = {key: {"agg": {"out_spec": ("in_spec", "first")}}}
    dummy_dti = DatetimeIndex(["01/01/2020 08:00", "01/01/2020 09:00"])
    in_spec_val = [1, 2]
    seed_df = pDataFrame(
        {
            ordered_on: dummy_dti,
            "in_spec": in_spec_val,
        },
    )
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on,
        "trim_start": True,
        "agg": None,
        "post": None,
        "seed_dtypes": seed_df.dtypes.to_dict(),
    }
    # Test.
    with pytest.raises(ValueError, match="^'bin_by' parameter is missing"):
        _setup(**parameter_in)


def test_exception_different_index_at_restart(tmp_path):
    # Test exception at restart with 2 different 'seed_index_restart' for 2
    # different keys.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    max_row_group_size = 6
    start = Timestamp("2020/01/01")
    rr = np.random.default_rng(1)
    N = 10
    rand_ints = rr.integers(100, size=N)
    rand_ints.sort()
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    ordered_on = "ts"
    seed_df = pDataFrame({ordered_on: ts, "val": rand_ints})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    # Streamagg with 'key1'.
    key1 = Indexer("agg_2T")
    key1_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "last": ("val", "last")},
    }
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        keys={key1: key1_cf},
        discard_last=True,
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
    seed = ParquetFile(seed_path)
    # Streamagg with 'key2'.
    key2 = Indexer("agg_13T")
    key2_cf = {
        "bin_by": TimeGrouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "max": ("val", "max")},
    }
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        keys={key2: key2_cf},
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Extend seed again.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400})
    fp_write(
        seed_path,
        seed_df3,
        row_group_offsets=max_row_group_size,
        file_scheme="hive",
        append=True,
    )
    seed = ParquetFile(seed_path)
    # Streamagg with 'key1' and 'key2'.
    key_configs = {key1: key1_cf, key2: key2_cf}
    # Test.
    with pytest.raises(
        ValueError,
        match="^not possible to aggregate on multiple keys with existing",
    ):
        streamagg(
            seed=seed,
            ordered_on=ordered_on,
            store=store,
            keys=key_configs,
            discard_last=True,
            max_row_group_size=max_row_group_size,
        )
