#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh

Test utils.
- Check pandas DataFrame equality:
from pandas.testing import assert_frame_equal
- Run pytest in iPython:
run -m pytest /home/yoh/Documents/code/oups/tests/test_aggstream/test_aggstream_init.py
- Initialize store object:
tmp_path = os_path.expanduser('~/Documents/code/data/oups')
store = ParquetSet(os_path.join(tmp_path, "store"), Indexer)

"""
from copy import deepcopy
from os import path as os_path

import pytest
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas.core.resample import TimeGrouper

from oups import AggStream
from oups import ParquetSet
from oups import toplevel
from oups.aggstream.aggstream import KEY_AGG
from oups.aggstream.aggstream import KEY_AGG_RES_BUFFER
from oups.aggstream.aggstream import KEY_BIN_ON_OUT
from oups.aggstream.aggstream import KEY_BIN_RES_BUFFER
from oups.aggstream.aggstream import KEY_CHECK
from oups.aggstream.aggstream import KEY_CHECK_BUFFER
from oups.aggstream.aggstream import KEY_FILTERS
from oups.aggstream.aggstream import KEY_POST
from oups.aggstream.aggstream import KEY_POST_BUFFER
from oups.aggstream.aggstream import KEY_RESTART_INDEX
from oups.aggstream.aggstream import KEY_SEGAGG_BUFFER
from oups.aggstream.aggstream import NO_FILTER_ID
from oups.aggstream.jcumsegagg import FIRST
from oups.aggstream.jcumsegagg import LAST
from oups.aggstream.jcumsegagg import SUM
from oups.aggstream.segmentby import KEY_ORDERED_ON
from oups.store.writer import KEY_DUPLICATES_ON
from oups.store.writer import KEY_MAX_ROW_GROUP_SIZE


@toplevel
class Indexer:
    dataset_ref: str


@pytest.fixture
def store(tmp_path):
    # Reuse pre-defined Indexer.
    return ParquetSet(os_path.join(tmp_path, "store"), Indexer)


def always_true(**kwargs):
    # Dummy function for use as 'bin_by', or 'post'...
    return True


def always_false(**kwargs):
    # Dummy function for use as 'bin_by', or 'post'...
    return False


@pytest.mark.parametrize(
    (
        "root_parameters, ref_seed_config, ref_keys_config, ref_agg_pd, "
        "ref_filter_ids_to_keys, ref_min_number_of_keys_per_filter"
    ),
    [
        (
            # Test  1 /
            # Single key at root level.
            # root_parameters
            {
                KEY_MAX_ROW_GROUP_SIZE: 6,
                KEY_ORDERED_ON: "ts",
                "keys": Indexer("key1"),
                "bin_by": TimeGrouper(key="ts", freq="1H", closed="left", label="left"),
                KEY_AGG: {"agg_out": ("val", SUM)},
            },
            # ref_seed_config
            {
                KEY_ORDERED_ON: "ts",
                KEY_CHECK: None,
                KEY_CHECK_BUFFER: {},
                KEY_FILTERS: {NO_FILTER_ID: None},
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                str(Indexer("key1")): {
                    KEY_BIN_ON_OUT: "ts",
                    KEY_POST: None,
                    "write_config": {
                        KEY_ORDERED_ON: "ts",
                        KEY_MAX_ROW_GROUP_SIZE: 6,
                        KEY_DUPLICATES_ON: "ts",
                    },
                },
            },
            # ref_agg_pd
            {str(Indexer("key1")): {"agg_out": ("val", SUM)}},
            # ref_filter_ids_to_keys
            {NO_FILTER_ID: [str(Indexer("key1"))]},
            # ref_min_number_of_keys_per_filter
            1,
        ),
        (
            # Test 2 /
            # Test that config for each key and for seed is correctly consolidated,
            # with default values when parameters are not specified at key level.
            # 'max_row_group_size' and 'max_nirgs' are default parameters
            # for writing. 'post' also has a default value.
            # Key 1 has mostly parameters defined by default values.
            #       'by' is a pd.TimeGrouper.
            # Key 2 has mostly parameters defined by specific values.
            # Keys 3 & 4 have only default parameters, except minimally compulsory
            # specific parameters.
            # root_parameters
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_MAX_ROW_GROUP_SIZE: 1000,
                "max_nirgs": 4,
                KEY_CHECK: always_true,
                KEY_AGG: {"out_dflt": ("in_dflt", LAST)},
                KEY_POST: always_true,
                "keys": {
                    Indexer("key1_some_default"): {
                        KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                        "bin_by": TimeGrouper(key="ts_dflt", freq="1H"),
                        KEY_POST: always_false,
                    },
                    Indexer("key2_only_specific"): {
                        KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                        "bin_by": always_true,
                        KEY_POST: None,
                        KEY_MAX_ROW_GROUP_SIZE: 3000,
                        KEY_ORDERED_ON: "ts_spec",
                    },
                    Indexer("key3_only_default"): {
                        "bin_by": always_false,
                        "bin_on": ("bin_on_spec", "bin_out_spec"),
                    },
                    Indexer("key4_most_default"): {
                        "bin_by": TimeGrouper(key="ts_dflt", freq="1H"),
                        KEY_ORDERED_ON: "ts_spec",
                    },
                },
            },
            # ref_seed_config
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_CHECK: always_true,
                KEY_CHECK_BUFFER: {},
                KEY_FILTERS: {NO_FILTER_ID: None},
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                str(Indexer("key1_some_default")): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_false,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                },
                str(Indexer("key2_only_specific")): {
                    KEY_BIN_ON_OUT: None,
                    KEY_POST: None,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 3000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_spec",
                    },
                },
                str(Indexer("key3_only_default")): {
                    KEY_BIN_ON_OUT: "bin_out_spec",
                    KEY_POST: always_true,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "bin_out_spec",
                    },
                },
                str(Indexer("key4_most_default")): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_true,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                },
            },
            # ref_agg_pd
            {
                str(Indexer("key1_some_default")): {"out_spec": ("in_spec", FIRST)},
                str(Indexer("key2_only_specific")): {"out_spec": ("in_spec", FIRST)},
                str(Indexer("key3_only_default")): {"out_dflt": ("in_dflt", LAST)},
                str(Indexer("key4_most_default")): {"out_dflt": ("in_dflt", LAST)},
            },
            # ref_filter_ids_to_keys
            {
                NO_FILTER_ID: [
                    str(Indexer("key1_some_default")),
                    str(Indexer("key2_only_specific")),
                    str(Indexer("key3_only_default")),
                    str(Indexer("key4_most_default")),
                ],
            },
            # ref_min_number_of_keys_per_filter
            4,
        ),
        (
            # Test 3 /
            # Same as test 2, but with 2 filters.
            # - filter 1 has key 1, 2 & 3,
            # - filter 2 has key 4.
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_MAX_ROW_GROUP_SIZE: 1000,
                "max_nirgs": 4,
                KEY_CHECK: always_true,
                KEY_AGG: {"out_dflt": ("in_dflt", LAST)},
                KEY_POST: always_true,
                "keys": {
                    "filter1": {
                        Indexer("key1_some_default"): {
                            KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                            "bin_by": TimeGrouper(key="ts_dflt", freq="1H"),
                            KEY_POST: always_false,
                        },
                        Indexer("key2_only_specific"): {
                            KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                            "bin_by": always_true,
                            KEY_POST: None,
                            KEY_MAX_ROW_GROUP_SIZE: 3000,
                            KEY_ORDERED_ON: "ts_spec",
                        },
                        Indexer("key3_only_default"): {
                            "bin_by": always_false,
                            "bin_on": ("bin_on_spec", "bin_out_spec"),
                        },
                    },
                    "filter2": {
                        Indexer("key4_most_default"): {
                            "bin_by": TimeGrouper(key="ts_dflt", freq="1H"),
                            KEY_ORDERED_ON: "ts_spec",
                        },
                    },
                },
                KEY_FILTERS: {
                    "filter1": [("ts_dflt", ">=", 4)],
                    "filter2": [[("in_spec", "<=", 10)]],
                },
            },
            # ref_seed_config
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_CHECK: always_true,
                KEY_CHECK_BUFFER: {},
                KEY_FILTERS: {
                    "filter1": [[("ts_dflt", ">=", 4)]],
                    "filter2": [[("in_spec", "<=", 10)]],
                },
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                str(Indexer("key1_some_default")): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_false,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                },
                str(Indexer("key2_only_specific")): {
                    KEY_BIN_ON_OUT: None,
                    KEY_POST: None,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 3000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_spec",
                    },
                },
                str(Indexer("key3_only_default")): {
                    KEY_BIN_ON_OUT: "bin_out_spec",
                    KEY_POST: always_true,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "bin_out_spec",
                    },
                },
                str(Indexer("key4_most_default")): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_true,
                    "write_config": {
                        KEY_MAX_ROW_GROUP_SIZE: 1000,
                        "max_nirgs": 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                },
            },
            # ref_agg_pd
            {
                str(Indexer("key1_some_default")): {"out_spec": ("in_spec", FIRST)},
                str(Indexer("key2_only_specific")): {"out_spec": ("in_spec", FIRST)},
                str(Indexer("key3_only_default")): {"out_dflt": ("in_dflt", LAST)},
                str(Indexer("key4_most_default")): {"out_dflt": ("in_dflt", LAST)},
            },
            # ref_filter_ids_to_keys
            {
                "filter1": [
                    str(Indexer("key1_some_default")),
                    str(Indexer("key2_only_specific")),
                    str(Indexer("key3_only_default")),
                ],
                "filter2": [str(Indexer("key4_most_default"))],
            },
            # ref_min_number_of_keys_per_filter
            1,
        ),
    ],
)
def test_aggstream_init(
    store,
    root_parameters,
    ref_seed_config,
    ref_keys_config,
    ref_agg_pd,
    ref_filter_ids_to_keys,
    ref_min_number_of_keys_per_filter,
):
    # Setup streamed aggregation.
    as_ = AggStream(store=store, **root_parameters)
    # Check
    assert as_.seed_config == ref_seed_config
    # Do not check 'seg_config' in 'keys_config'.
    res_keys_config = deepcopy(as_.keys_config)
    for key, ref in ref_keys_config.items():
        del res_keys_config[key]["seg_config"]
        del res_keys_config[key]["dirpath"]
        assert res_keys_config[key] == ref
    ref_agg_buffers = {
        "agg_n_rows": 0,
        "agg_res": None,
        "bin_res": None,
        KEY_AGG_RES_BUFFER: [],
        KEY_BIN_RES_BUFFER: [],
        KEY_SEGAGG_BUFFER: {},
        KEY_POST_BUFFER: {},
    }
    for key in ref_keys_config:
        assert as_.agg_buffers[key] == ref_agg_buffers
    # Other AggStream attributes.
    main_ref_attr = {
        "store": store,
        "agg_pd": ref_agg_pd,
        "agg_cs": {},
        "filter_ids_to_keys": ref_filter_ids_to_keys,
        "min_number_of_keys_per_filter": ref_min_number_of_keys_per_filter,
    }
    for attr in main_ref_attr:
        assert getattr(as_, attr) == main_ref_attr[attr]


def test_exception_not_key_of_streamagg_results(store):
    # Test error message provided key is not that of streamagg results.
    # Store some data using 'key'.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30"])
    ordered_on = "ts_order"
    val = range(1, len(ts) + 1)
    seed_pdf = pDataFrame({ordered_on: ts, "val": val})
    key = Indexer("agg_res")
    store[key] = seed_pdf
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    with pytest.raises(ValueError, match="^provided 'agg_res'"):
        AggStream(
            ordered_on=ordered_on,
            agg=agg,
            store=store,
            keys=key,
            bin_by=bin_by,
        )


def test_exception_setup_no_bin_by(store):
    ordered_on = "ts"
    key = Indexer("agg_res")
    keys_config = {key: {"agg": {"out_spec": ("in_spec", FIRST)}}}
    # Test.
    with pytest.raises(ValueError, match="^'bin_by' parameter is missing"):
        AggStream(
            store=store,
            keys=keys_config,
            ordered_on=ordered_on,
            agg=None,
            post=None,
        )
