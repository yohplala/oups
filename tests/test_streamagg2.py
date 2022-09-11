#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from copy import copy
from os import path as os_path

import numpy as np
import pytest
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Series
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.streamagg import REDUCTION_BIN_COL_PREFIX
from oups.streamagg import _setup
from oups.writer import MAX_ROW_GROUP_SIZE


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class Indexer:
    dataset_ref: str


def test_setup_4_keys_with_default_parameters_for_writing(tmp_path):
    # Test config for each key and for seed is correctly consolidated,
    # with default values when parameters are not specified at key level.
    # 'max_row_group_size' and 'max_nirgs' are default parameters for writing.
    # 'post' also has a default value.
    # Key 1 has mostly parameters defined by default values.
    #       'by' is a pd.Grouper.
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    # 'reduction' is `False`.
    ordered_on_alt = "ts_alt"
    ordered_on_dflt = "ts_dflt"
    bin_on_spec = "bin_on_spec"
    bin_out_spec = "bin_out_spec"
    key1 = Indexer("some_default")
    key2 = Indexer("only_specific")
    key3 = Indexer("only_default1")
    key4 = Indexer("only_default2")
    tgrouper = Grouper(key=ordered_on_alt, freq="1H")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)

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
        key1: {"agg": {"out_spec": ("in_spec", "first")}, "by": tgrouper, "post": dummy_post_spec},
        key2: {
            "agg": {"out_spec": ("in_spec", "first")},
            "by": dummy_by,
            "post": None,
            "max_row_group_size": 3000,
        },
        key3: {"by": dummy_by, "bin_on": (bin_on_spec, bin_out_spec)},
        key4: {"bin_on": bin_on_spec},
    }
    trim_start = True
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on_dflt,
        "trim_start": trim_start,
        "agg": {"out_dflt": ("in_dflt", "last")},
        "post": dummy_post_dflt,
        "reduction": False,
        "max_row_group_size": 1000,
        "max_nirgs": 4,
    }
    # Test.
    (
        all_cols_in_res,
        trim_start,
        seed_index_restart_set,
        reduction_bin_cols_res,
        reduction_seed_chunk_cols_res,
        reduction_agg_res,
        keys_config_res,
    ) = _setup(**parameter_in)
    # Reference results.
    keys_config_ref = {
        key1: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": None,
            "reduction_bin_col": None,
            "bins": tgrouper,
            "bin_out_col": ordered_on_alt,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec", "first")},
            "post": dummy_post_spec,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_alt,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key2: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": None,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec", "first")},
            "post": None,
            "max_agg_row_group_size": 3000,
            "write_config": {
                "max_row_group_size": 3000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key3: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": [ordered_on_dflt, bin_on_spec],
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": bin_out_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_out_spec,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key4: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": None,
            "reduction_bin_col": None,
            "bins": bin_on_spec,
            "bin_out_col": bin_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_on_spec,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
    }
    # Check.
    key1_last_agg_row = keys_config_res[key1].pop("last_agg_row")
    assert key1_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key1] == keys_config_res[key1]
    key2_last_agg_row = keys_config_res[key2].pop("last_agg_row")
    assert key2_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key2] == keys_config_res[key2]
    key3_last_agg_row = keys_config_res[key3].pop("last_agg_row")
    assert key3_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key3] == keys_config_res[key3]
    key4_last_agg_row = keys_config_res[key4].pop("last_agg_row")
    assert key4_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key4] == keys_config_res[key4]
    #
    all_cols_in_ref = {"in_spec", ordered_on_dflt, bin_on_spec, ordered_on_alt, "in_dflt"}
    assert set(all_cols_in_res) == all_cols_in_ref
    assert not trim_start
    assert not seed_index_restart_set
    assert not reduction_bin_cols_res
    assert not reduction_seed_chunk_cols_res
    assert not reduction_agg_res


def test_setup_4_keys_wo_default_parameters_for_writing_nor_post(tmp_path):
    # Test config for each key and for seed is correctly consolidated,
    # with default values when parameters are not specified at key level.
    # 'max_row_group_size' and 'max_nirgs' are not provided for writing.
    # 'post' also has no default value either.
    # Key 1 has mostly parameters defined by default values.
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    ordered_on_alt = "ts_alt"
    ordered_on_dflt = "ts_dflt"
    bin_on_spec = "bin_on_spec"
    bin_out_spec = "bin_out_spec"
    key1 = Indexer("some_default")
    key2 = Indexer("only_specific")
    key3 = Indexer("only_default1")
    key4 = Indexer("only_default2")
    tgrouper = Grouper(key=ordered_on_alt, freq="1H")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)

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
        key1: {"agg": {"out_spec": ("in_spec", "first")}, "by": tgrouper, "post": dummy_post_spec},
        key2: {
            "agg": {"out_spec": ("in_spec", "first")},
            "by": dummy_by,
            "max_row_group_size": 3000,
        },
        key3: {"by": dummy_by, "bin_on": (bin_on_spec, bin_out_spec)},
        key4: {"bin_on": bin_on_spec},
    }
    trim_start = True
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on_dflt,
        "trim_start": trim_start,
        "agg": {"out_dflt": ("in_dflt", "last")},
        "reduction": False,
        "post": None,
    }
    # Test.
    (
        all_cols_in_res,
        trim_start,
        seed_index_restart_set,
        reduction_bin_cols_res,
        reduction_seed_chunk_cols_res,
        reduction_agg_res,
        keys_config_res,
    ) = _setup(**parameter_in)
    # Reference results.
    keys_config_ref = {
        key1: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": None,
            "reduction_bin_col": None,
            "bins": tgrouper,
            "bin_out_col": ordered_on_alt,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec", "first")},
            "post": dummy_post_spec,
            "max_agg_row_group_size": MAX_ROW_GROUP_SIZE,
            "write_config": {"ordered_on": ordered_on_dflt, "duplicates_on": ordered_on_alt},
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key2: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": None,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec", "first")},
            "post": None,
            "max_agg_row_group_size": 3000,
            "write_config": {
                "max_row_group_size": 3000,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key3: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": [ordered_on_dflt, bin_on_spec],
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": bin_out_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": None,
            "max_agg_row_group_size": MAX_ROW_GROUP_SIZE,
            "write_config": {"ordered_on": ordered_on_dflt, "duplicates_on": bin_out_spec},
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key4: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": None,
            "reduction_bin_col": None,
            "bins": bin_on_spec,
            "bin_out_col": bin_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": None,
            "max_agg_row_group_size": MAX_ROW_GROUP_SIZE,
            "write_config": {"ordered_on": ordered_on_dflt, "duplicates_on": bin_on_spec},
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
    }
    # Check.
    key1_last_agg_row = keys_config_res[key1].pop("last_agg_row")
    assert key1_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key1] == keys_config_res[key1]
    key2_last_agg_row = keys_config_res[key2].pop("last_agg_row")
    assert key2_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key2] == keys_config_res[key2]
    key3_last_agg_row = keys_config_res[key3].pop("last_agg_row")
    assert key3_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key3] == keys_config_res[key3]
    key4_last_agg_row = keys_config_res[key4].pop("last_agg_row")
    assert key4_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key4] == keys_config_res[key4]
    #
    all_cols_in_ref = {"in_spec", ordered_on_dflt, bin_on_spec, ordered_on_alt, "in_dflt"}
    assert set(all_cols_in_res) == all_cols_in_ref
    assert not trim_start
    assert not seed_index_restart_set
    assert not reduction_bin_cols_res
    assert not reduction_seed_chunk_cols_res
    assert not reduction_agg_res


def test_setup_4_keys_with_default_parameters_for_writing_n_reduction(tmp_path):
    # Test config for each key and for seed is correctly consolidated,
    # with default values when parameters are not specified at key level.
    # 'max_row_group_size' and 'max_nirgs' are default parameters for writing.
    # 'post' also has a default value.
    # 'reduction' is `True`.
    # Key 1 has mostly parameters defined by default values.
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    ordered_on_alt = "ts_alt"
    ordered_on_dflt = "ts_dflt"
    bin_on_spec = "bin_on_spec"
    bin_out_spec = "bin_out_spec"
    key1 = Indexer("some_default")
    key2 = Indexer("only_specific")
    key3 = Indexer("only_default1")
    key4 = Indexer("only_default2")
    tgrouper = Grouper(key=ordered_on_alt, freq="1H")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)

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
        key1: {"agg": {"out_spec": ("in_spec", "first")}, "by": tgrouper, "post": dummy_post_spec},
        key2: {
            "agg": {"out_spec": ("in_spec", "first")},
            "by": dummy_by,
            "post": None,
            "max_row_group_size": 3000,
        },
        key3: {"by": dummy_by, "bin_on": (bin_on_spec, bin_out_spec)},
        key4: {"bin_on": bin_on_spec},
    }
    trim_start = True
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on_dflt,
        "trim_start": trim_start,
        "agg": {"out_dflt": ("in_dflt", "last")},
        "post": dummy_post_dflt,
        "reduction": True,
        "max_row_group_size": 1000,
        "max_nirgs": 4,
    }
    # Test.
    (
        all_cols_in_res,
        trim_start,
        seed_index_restart_set,
        reduction_bin_cols_res,
        reduction_seed_chunk_cols_res,
        reduction_agg_res,
        keys_config_res,
    ) = _setup(**parameter_in)
    # Reference results.
    tgrouper_key1_ref = copy(tgrouper)
    tgrouper_key1_ref.key = f"{REDUCTION_BIN_COL_PREFIX}0"
    keys_config_ref = {
        key1: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": tgrouper,
            "reduction_bin_col": f"{REDUCTION_BIN_COL_PREFIX}0",
            "bins": tgrouper_key1_ref,
            "bin_out_col": ordered_on_alt,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec__first", "first")},
            "post": dummy_post_spec,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_alt,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key2: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": f"{REDUCTION_BIN_COL_PREFIX}1",
            "bins": f"{REDUCTION_BIN_COL_PREFIX}1",
            "bin_out_col": None,
            "self_agg": {"out_spec": ("out_spec", "first")},
            "agg": {"out_spec": ("in_spec__first", "first")},
            "post": None,
            "max_agg_row_group_size": 3000,
            "write_config": {
                "max_row_group_size": 3000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key3: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": [ordered_on_dflt, bin_on_spec],
            "by": dummy_by,
            "reduction_bin_col": f"{REDUCTION_BIN_COL_PREFIX}2",
            "bins": f"{REDUCTION_BIN_COL_PREFIX}2",
            "bin_out_col": bin_out_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt__last", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_out_spec,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
        key4: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": None,
            "by": None,
            "reduction_bin_col": bin_on_spec,
            "bins": bin_on_spec,
            "bin_out_col": bin_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt__last", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": bin_on_spec,
            },
            "binning_buffer": {},
            "post_buffer": {},
            "agg_chunks_buffer": [],
        },
    }
    reduction_agg_ref = {
        "in_spec__first": ("in_spec", "first"),
        "in_dflt__last": ("in_dflt", "last"),
    }
    # Check.
    key1_last_agg_row = keys_config_res[key1].pop("last_agg_row")
    assert key1_last_agg_row.equals(pDataFrame())
    key1_bins_res = keys_config_res[key1].pop("bins")
    key1_bins_ref = keys_config_ref[key1].pop("bins")
    assert key1_bins_res.__dict__ == key1_bins_ref.__dict__
    assert keys_config_ref[key1] == keys_config_res[key1]
    key2_last_agg_row = keys_config_res[key2].pop("last_agg_row")
    assert key2_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key2] == keys_config_res[key2]
    key3_last_agg_row = keys_config_res[key3].pop("last_agg_row")
    assert key3_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key3] == keys_config_res[key3]
    key4_last_agg_row = keys_config_res[key4].pop("last_agg_row")
    assert key4_last_agg_row.equals(pDataFrame())
    assert keys_config_ref[key4] == keys_config_res[key4]
    #
    all_cols_in_ref = {"in_spec", ordered_on_dflt, bin_on_spec, ordered_on_alt, "in_dflt"}
    assert set(all_cols_in_res) == all_cols_in_ref
    assert reduction_agg_res == reduction_agg_ref
    assert not trim_start
    assert not seed_index_restart_set
    reduction_seed_chunk_cols_ref = {"in_dflt", "in_spec", bin_on_spec}
    assert set(reduction_seed_chunk_cols_res) == reduction_seed_chunk_cols_ref
    reduction_bin_cols_ref = [
        f"{REDUCTION_BIN_COL_PREFIX}0",
        f"{REDUCTION_BIN_COL_PREFIX}1",
        f"{REDUCTION_BIN_COL_PREFIX}2",
        bin_on_spec,
    ]
    assert reduction_bin_cols_res == reduction_bin_cols_ref


@pytest.mark.parametrize("reduction1,reduction2", [(False, False), (True, True), (True, False)])
def test_parquet_seed_3_keys(tmp_path, reduction1, reduction2):
    # Test with parquet seed, and 4 keys.
    # - key 1: time grouper '2T', agg 'first', and 'last',
    # - key 2: time grouper '13T', agg 'first', and 'max',
    # - key 3: 'by' as callable, every 4 rows, agg 'min', 'max',
    # - key 4: 'by' as None, and direct use of a column,
    #          agg 'first' on 'ordered_on' and on 'val'.
    max_row_group_size = 6
    start = Timestamp("2020/01/01")
    rr = np.random.default_rng(1)
    N = 24
    rand_ints = rr.integers(100, size=N)
    rand_ints.sort()
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    N_third = int(N / 3)
    bin_val = np.array(
        [1] * (N_third - 2) + [2] * (N_third - 4) + [3] * (N - 2 * N_third) + [4] * 6
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
    key4 = Indexer("agg_direct_bin")

    # Setup binning for key3.
    def by_4rows(data: Series, buffer: dict):
        """Bin by group of 4 rows. Label for bins are values from `ordered_on`."""
        # A pandas Series is returned, with name being that of the 'ordered_on'
        # column. Because of pandas magic, this column will then be in aggregation
        # results, and oups will be able to use it for writing data.
        # With actual setting, without this trick, 'streamagg' could not write
        # the results (no 'ordered_on' column in results).
        ordered_on = data.name
        group_keys = pDataFrame(data)
        # Setup 1st key of groups from previous binning.
        row_offset = 4 - buffer["row_offset"] if "row_offset" in buffer else 0
        group_keys["tmp"] = data.iloc[row_offset::4]
        if row_offset and "last_key" in buffer:
            # Initialize 1st row if row_offset is not 0.
            group_keys.iloc[0, group_keys.columns.get_loc("tmp")] = buffer["last_key"]
        group_keys[ordered_on] = group_keys["tmp"].ffill()
        group_keys = Series(group_keys[ordered_on], name=ordered_on)
        keys, counts = np.unique(group_keys, return_counts=True)
        # Update buffer in-place for next binning.
        if "row_offset" in buffer and buffer["row_offset"] != 4:
            buffer["row_offset"] = counts[-1] + buffer["row_offset"]
        else:
            buffer["row_offset"] = counts[-1]
        buffer["last_key"] = keys[-1]
        return group_keys

    # Setup aggregation.
    key1_cf = {
        "by": Grouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "last": ("val", "last")},
    }
    key2_cf = {
        "by": Grouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "max": ("val", "max")},
    }
    key3_cf = {
        "by": Grouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {"min": ("val", "min"), "max": ("val", "max")},
    }
    key4_cf = {
        "bin_on": bin_on,
        "agg": {ordered_on: (ordered_on, "first"), "first": ("val", "first")},
    }
    key_configs = {key1: key1_cf, key2: key2_cf, key3: key3_cf, key4: key4_cf}
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        key=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Test results
    # Remove last 'group' as per 'ordered_on' in 'seed_df'.
    seed_df_trim = seed_df[seed_df[ordered_on] < seed_df[ordered_on].iloc[-1]]
    ref_res = {
        key: seed_df_trim.groupby(key_configs[key]["by"])
        .agg(**key_configs[key]["agg"])
        .reset_index()
        for key in [key1, key2, key3]
    } | {
        key4: seed_df_trim.groupby(key_configs[key4]["bin_on"])
        .agg(**key_configs[key4]["agg"])
        .reset_index()
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100, bin_on: bin_val + 10})
    fp_write(
        seed_path, seed_df2, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        key=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    seed_df2_trim = seed_df2[seed_df2[ordered_on] < seed_df2[ordered_on].iloc[-1]]
    ref_res = {
        key: pconcat([seed_df, seed_df2_trim])
        .groupby(key_configs[key]["by"])
        .agg(**key_configs[key]["agg"])
        .reset_index()
        for key in [key1, key2, key3]
    } | {
        key4: pconcat([seed_df, seed_df2_trim])
        .groupby(key_configs[key4]["bin_on"])
        .agg(**key_configs[key4]["agg"])
        .reset_index()
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)
    # 2nd append of new data.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400, bin_on: bin_val + 40})
    fp_write(
        seed_path, seed_df3, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        key=key_configs,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    seed_df3_trim = seed_df3[seed_df3[ordered_on] < seed_df3[ordered_on].iloc[-1]]
    ref_res = {
        key: pconcat([seed_df, seed_df2, seed_df3_trim])
        .groupby(key_configs[key]["by"])
        .agg(**key_configs[key]["agg"])
        .reset_index()
        for key in [key1, key2, key3]
    } | {
        key4: pconcat([seed_df, seed_df2, seed_df3_trim])
        .groupby(key_configs[key4]["bin_on"])
        .agg(**key_configs[key4]["agg"])
        .reset_index()
    }
    for key, ref_df in ref_res.items():
        rec_res = store[key].pdf
        assert rec_res.equals(ref_df)


def test_exception_setup_no_bin_on_nor_by(tmp_path):
    ordered_on = "ts"
    key = Indexer("agg_res")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    keys_config = {key: {"agg": {"out_spec": ("in_spec", "first")}}}
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on,
        "trim_start": True,
        "agg": None,
        "post": None,
        "reduction": True,
    }
    # Test.
    with pytest.raises(ValueError, match="^at least one among"):
        (all_cols_in, trim_start, seed_index_restart_set, reduction_agg, keys_config_res) = _setup(
            **parameter_in
        )


def test_exception_setup_no_bin_on_nor_by_key_when_grouper(tmp_path):
    ordered_on = "ts"
    key = Indexer("agg_res")
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    keys_config = {key: {"agg": {"out_spec": ("in_spec", "first")}, "by": Grouper(freq="1H")}}
    parameter_in = {
        "store": store,
        "keys": keys_config,
        "ordered_on": ordered_on,
        "trim_start": True,
        "agg": None,
        "post": None,
        "reduction": True,
    }
    # Test.
    with pytest.raises(ValueError, match="^no column name defined to bin"):
        (all_cols_in, trim_start, seed_index_restart_set, reduction_agg, keys_config_res) = _setup(
            **parameter_in
        )


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
        "by": Grouper(key=ordered_on, freq="2T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "last": ("val", "last")},
    }
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        key={key1: key1_cf},
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Extend seed.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100})
    fp_write(
        seed_path, seed_df2, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
    # Streamagg with 'key2'.
    key2 = Indexer("agg_13T")
    key2_cf = {
        "by": Grouper(key=ordered_on, freq="13T", closed="left", label="left"),
        "agg": {"first": ("val", "first"), "max": ("val", "max")},
    }
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        store=store,
        key={key2: key2_cf},
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Extend seed again.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400})
    fp_write(
        seed_path, seed_df3, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
    # Streamagg with 'key1' and 'key2'.
    key_configs = {key1: key1_cf, key2: key2_cf}
    # Test.
    with pytest.raises(
        ValueError, match="^not possible to aggregate on multiple keys with existing"
    ):
        streamagg(
            seed=seed,
            ordered_on=ordered_on,
            store=store,
            key=key_configs,
            discard_last=True,
            max_row_group_size=max_row_group_size,
        )
