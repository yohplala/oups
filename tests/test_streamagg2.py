#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from copy import copy
from os import path as os_path

import pytest
from pandas import DataFrame as pDataFrame
from pandas import Grouper

from oups import ParquetSet
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
    # Key 2 has mostly parameters defined by specific values.
    # Keys 3 & 4 have only default parameters, except minimally compulsory
    # specific parameters.
    ordered_on_alt = "ts_alt"
    ordered_on_spec = "ts_spec"
    ordered_on_dflt = "ts_dflt"
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
        key3: {"by": dummy_by},
        key4: {"bin_on": ordered_on_spec},
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
    (all_cols_in, trim_start, seed_index_restart_set, reduction_agg_res, keys_config_res) = _setup(
        **parameter_in
    )
    # Reference results.
    keys_config_ref = {
        key1: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": [ordered_on_dflt, ordered_on_alt],
            "by": tgrouper,
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
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": None,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
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
            "cols_to_by": [ordered_on_dflt, ordered_on_spec],
            "by": None,
            "reduction_bin_col": None,
            "bins": ordered_on_spec,
            "bin_out_col": ordered_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_spec,
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
    assert not seed_index_restart_set
    assert not trim_start
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
    ordered_on_spec = "ts_spec"
    ordered_on_dflt = "ts_dflt"
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
        key3: {"by": dummy_by},
        key4: {"bin_on": ordered_on_spec},
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
    (all_cols_in, trim_start, seed_index_restart_set, reduction_agg_res, keys_config_res) = _setup(
        **parameter_in
    )
    # Reference results.
    keys_config_ref = {
        key1: {
            "agg_n_rows": 0,
            "agg_mean_row_group_size": 0,
            "agg_res": None,
            "agg_res_len": None,
            "isfbn": True,
            "cols_to_by": [ordered_on_dflt, ordered_on_alt],
            "by": tgrouper,
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
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": None,
            "bins": None,
            "bin_out_col": None,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": None,
            "max_agg_row_group_size": MAX_ROW_GROUP_SIZE,
            "write_config": {"ordered_on": ordered_on_dflt, "duplicates_on": ordered_on_dflt},
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
            "cols_to_by": [ordered_on_dflt, ordered_on_spec],
            "by": None,
            "reduction_bin_col": None,
            "bins": ordered_on_spec,
            "bin_out_col": ordered_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt", "last")},
            "post": None,
            "max_agg_row_group_size": MAX_ROW_GROUP_SIZE,
            "write_config": {"ordered_on": ordered_on_dflt, "duplicates_on": ordered_on_spec},
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
    assert not seed_index_restart_set
    assert not trim_start
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
    ordered_on_spec = "ts_spec"
    ordered_on_dflt = "ts_dflt"
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
        key3: {"by": dummy_by},
        key4: {"bin_on": ordered_on_spec},
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
    (all_cols_in, trim_start, seed_index_restart_set, reduction_agg_res, keys_config_res) = _setup(
        **parameter_in
    )
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
            "cols_to_by": [ordered_on_dflt, ordered_on_alt],
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
            "cols_to_by": ordered_on_dflt,
            "by": dummy_by,
            "reduction_bin_col": f"{REDUCTION_BIN_COL_PREFIX}2",
            "bins": f"{REDUCTION_BIN_COL_PREFIX}2",
            "bin_out_col": None,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt__last", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_dflt,
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
            "cols_to_by": [ordered_on_dflt, ordered_on_spec],
            "by": ordered_on_spec,
            "reduction_bin_col": f"{REDUCTION_BIN_COL_PREFIX}3",
            "bins": f"{REDUCTION_BIN_COL_PREFIX}3",
            "bin_out_col": ordered_on_spec,
            "self_agg": {"out_dflt": ("out_dflt", "last")},
            "agg": {"out_dflt": ("in_dflt__last", "last")},
            "post": dummy_post_dflt,
            "max_agg_row_group_size": 1000,
            "write_config": {
                "max_row_group_size": 1000,
                "max_nirgs": 4,
                "ordered_on": ordered_on_dflt,
                "duplicates_on": ordered_on_spec,
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
    assert set(all_cols_in) == {
        ordered_on_dflt,
        ordered_on_spec,
        "in_dflt",
        "in_spec",
        ordered_on_alt,
    }
    assert reduction_agg_res == reduction_agg_ref
    assert not seed_index_restart_set
    assert not trim_start


def test_setup_exception_no_bin_on_nor_by(tmp_path):
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


def test_setup_exception_no_bin_on_nor_by_key_when_grouper(tmp_path):
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


# Tester que 'bins' si reduction
#  - si 'by' est callable, récupère nom générique de colonne
#  - si 'by' est grouper, récupère nolm générique dans attribute 'by'

# Test avec 2 keys (sampler) réalisés séparemment puis ensemble: vérifier que chainagg
# réalise le changement de nom de l'index dans 'last_agg_row' pour redémarrer.


# Test exception when calling streamagg with agg = None & key not a dict: not possible.

# Test exception when, with aggregation restart, seed_index_restart_set has
# 2 different values (from 2 different agrgegation resultes)


# Test maxx row_group_sizedefined in dict of a keys is forwarded in write_config
# Test some default option vs non default options
# Check 'reduction_agg' and modified 'agg'
