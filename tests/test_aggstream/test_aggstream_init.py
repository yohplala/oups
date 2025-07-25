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

import pytest
from pandas import DataFrame
from pandas import DatetimeIndex
from pandas.core.resample import TimeGrouper

from oups import AggStream
from oups import Store
from oups import toplevel
from oups.aggstream.aggstream import FIRST
from oups.aggstream.aggstream import KEY_AGG
from oups.aggstream.aggstream import KEY_AGG_IN_MEMORY_SIZE
from oups.aggstream.aggstream import KEY_AGG_RES_TYPE
from oups.aggstream.aggstream import KEY_BIN_BY
from oups.aggstream.aggstream import KEY_BIN_ON
from oups.aggstream.aggstream import KEY_BIN_ON_OUT
from oups.aggstream.aggstream import KEY_BIN_RES
from oups.aggstream.aggstream import KEY_BIN_RES_BUFFER
from oups.aggstream.aggstream import KEY_DUPLICATES_ON
from oups.aggstream.aggstream import KEY_FILTERS
from oups.aggstream.aggstream import KEY_MAX_IN_MEMORY_SIZE_B
from oups.aggstream.aggstream import KEY_MAX_IN_MEMORY_SIZE_MB
from oups.aggstream.aggstream import KEY_ORDERED_ON
from oups.aggstream.aggstream import KEY_POST
from oups.aggstream.aggstream import KEY_POST_BUFFER
from oups.aggstream.aggstream import KEY_PRE
from oups.aggstream.aggstream import KEY_PRE_BUFFER
from oups.aggstream.aggstream import KEY_RESTART_INDEX
from oups.aggstream.aggstream import KEY_SEG_CONFIG
from oups.aggstream.aggstream import KEY_SEGAGG_BUFFER
from oups.aggstream.aggstream import KEY_SNAP_BY
from oups.aggstream.aggstream import KEY_SNAP_RES
from oups.aggstream.aggstream import KEY_SNAP_RES_BUFFER
from oups.aggstream.aggstream import KEY_WRITE_CONFIG
from oups.aggstream.aggstream import LAST
from oups.aggstream.aggstream import NO_FILTER_ID
from oups.aggstream.aggstream import SUM
from oups.aggstream.aggstream import AggResType
from oups.aggstream.aggstream import FilterApp
from oups.defines import KEY_MAX_N_OFF_TARGET_RGS
from oups.defines import KEY_ROW_GROUP_TARGET_SIZE


@toplevel
class Indexer:
    dataset_ref: str


key = Indexer("agg_res")


@pytest.fixture
def store(tmp_path):
    # Reuse pre-defined Indexer.
    #    return Store(os_path.join(tmp_path, "store"), Indexer)
    return Store(tmp_path / "store", Indexer)


def always_true(**kwargs):
    # Dummy function for use as 'bin_by', or 'post'...
    return True


def always_false(**kwargs):
    # Dummy function for use as 'bin_by', or 'post'...
    return False


@pytest.mark.parametrize(
    ("root_parameters, ref_seed_config, ref_keys_config, ref_agg_pd, " "ref_filter_apps"),
    [
        (
            # Test  1 /
            # Single key at root level.
            # root_parameters
            {
                KEY_ROW_GROUP_TARGET_SIZE: 6,
                KEY_ORDERED_ON: "ts",
                "keys": Indexer("key1"),
                KEY_BIN_BY: TimeGrouper(key="ts", freq="1h", closed="left", label="left"),
                KEY_AGG: {"agg_out": ("val", SUM)},
            },
            # ref_seed_config
            {
                KEY_ORDERED_ON: "ts",
                KEY_PRE: None,
                KEY_PRE_BUFFER: {},
                KEY_FILTERS: {NO_FILTER_ID: None},
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                Indexer("key1"): {
                    KEY_BIN_ON_OUT: "ts",
                    KEY_POST: None,
                    KEY_WRITE_CONFIG: {
                        KEY_ORDERED_ON: "ts",
                        KEY_ROW_GROUP_TARGET_SIZE: 6,
                        KEY_DUPLICATES_ON: "ts",
                    },
                    KEY_MAX_IN_MEMORY_SIZE_B: 146800640,
                    KEY_AGG_RES_TYPE: AggResType.BINS,
                },
            },
            # ref_agg_pd
            {Indexer("key1"): {"agg_out": ("val", SUM)}},
            # ref_filter_apps
            {NO_FILTER_ID: FilterApp([Indexer("key1")], "NA")},
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
                KEY_ROW_GROUP_TARGET_SIZE: 1000,
                KEY_MAX_N_OFF_TARGET_RGS: 4,
                KEY_PRE: always_true,
                KEY_AGG: {"out_dflt": ("in_dflt", LAST)},
                KEY_POST: always_true,
                KEY_MAX_IN_MEMORY_SIZE_MB: 10,
                "keys": {
                    Indexer("key1_some_default"): {
                        KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                        KEY_BIN_BY: TimeGrouper(key="ts_dflt", freq="1h"),
                        KEY_POST: always_false,
                        KEY_MAX_IN_MEMORY_SIZE_MB: 1,
                    },
                    Indexer("key2_only_specific"): {
                        KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                        KEY_BIN_BY: always_true,
                        KEY_POST: None,
                        KEY_ROW_GROUP_TARGET_SIZE: 3000,
                        KEY_ORDERED_ON: "ts_spec",
                    },
                    Indexer("key3_only_default"): {
                        KEY_BIN_BY: always_false,
                        KEY_BIN_ON: ("bin_on_spec", "bin_out_spec"),
                    },
                    Indexer("key4_most_default"): {
                        KEY_BIN_BY: TimeGrouper(key="ts_dflt", freq="1h"),
                        KEY_ORDERED_ON: "ts_spec",
                    },
                },
            },
            # ref_seed_config
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_PRE: always_true,
                KEY_PRE_BUFFER: {},
                KEY_FILTERS: {NO_FILTER_ID: None},
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                Indexer("key1_some_default"): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_false,
                    KEY_MAX_IN_MEMORY_SIZE_B: 1048576,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                    KEY_AGG_RES_TYPE: AggResType.BINS,
                },
                Indexer("key2_only_specific"): {
                    KEY_BIN_ON_OUT: None,
                    KEY_POST: None,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 3000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_spec",
                    },
                    KEY_AGG_RES_TYPE: AggResType.BINS,
                },
                Indexer("key3_only_default"): {
                    KEY_BIN_ON_OUT: "bin_out_spec",
                    KEY_POST: always_true,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "bin_out_spec",
                    },
                    KEY_AGG_RES_TYPE: AggResType.BINS,
                },
                Indexer("key4_most_default"): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_true,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                    KEY_AGG_RES_TYPE: AggResType.BINS,
                },
            },
            # ref_agg_pd
            {
                Indexer("key1_some_default"): {"out_spec": ("in_spec", FIRST)},
                Indexer("key2_only_specific"): {"out_spec": ("in_spec", FIRST)},
                Indexer("key3_only_default"): {"out_dflt": ("in_dflt", LAST)},
                Indexer("key4_most_default"): {"out_dflt": ("in_dflt", LAST)},
            },
            # ref_filter_apps
            {
                NO_FILTER_ID: FilterApp(
                    [
                        Indexer("key1_some_default"),
                        Indexer("key2_only_specific"),
                        Indexer("key3_only_default"),
                        Indexer("key4_most_default"),
                    ],
                    "NA",
                ),
            },
        ),
        (
            # Test 3 /
            # Same as test 2, but with 2 filters.
            # - filter 1 has key 1, 2 & 3,
            # - filter 2 has key 4.
            # Add 'snap_by' parameter to consolidate within each key's config.
            {
                KEY_ORDERED_ON: "ts_dflt",
                KEY_ROW_GROUP_TARGET_SIZE: 1000,
                KEY_MAX_N_OFF_TARGET_RGS: 4,
                KEY_PRE: always_true,
                KEY_AGG: {"out_dflt": ("in_dflt", LAST)},
                KEY_POST: always_true,
                KEY_SNAP_BY: TimeGrouper(key="ts_dflt", freq="30min"),
                KEY_MAX_IN_MEMORY_SIZE_MB: 10,
                "keys": {
                    "filter1": {
                        Indexer("key1_some_default"): {
                            KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                            KEY_BIN_BY: TimeGrouper(key="ts_dflt", freq="1h"),
                            KEY_POST: always_false,
                        },
                        Indexer("key2_only_specific"): {
                            KEY_AGG: {"out_spec": ("in_spec", FIRST)},
                            KEY_BIN_BY: always_true,
                            KEY_POST: None,
                            KEY_ROW_GROUP_TARGET_SIZE: 3000,
                            KEY_ORDERED_ON: "ts_spec",
                            KEY_MAX_IN_MEMORY_SIZE_MB: 1,
                        },
                        Indexer("key3_only_default"): {
                            KEY_BIN_BY: always_false,
                            KEY_BIN_ON: ("bin_on_spec", "bin_out_spec"),
                        },
                    },
                    "filter2": {
                        Indexer("key4_most_default"): {
                            KEY_BIN_BY: TimeGrouper(key="ts_dflt", freq="1h"),
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
                KEY_PRE: always_true,
                KEY_PRE_BUFFER: {},
                KEY_FILTERS: {
                    "filter1": [[("ts_dflt", ">=", 4)]],
                    "filter2": [[("in_spec", "<=", 10)]],
                },
                KEY_RESTART_INDEX: None,
            },
            # ref_keys_config
            {
                Indexer("key1_some_default"): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_false,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                    KEY_AGG_RES_TYPE: AggResType.SNAPS,
                },
                Indexer("key2_only_specific"): {
                    KEY_BIN_ON_OUT: None,
                    KEY_POST: None,
                    KEY_MAX_IN_MEMORY_SIZE_B: 1048576,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 3000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_spec",
                    },
                    KEY_AGG_RES_TYPE: AggResType.SNAPS,
                },
                Indexer("key3_only_default"): {
                    KEY_BIN_ON_OUT: "bin_out_spec",
                    KEY_POST: always_true,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_dflt",
                        KEY_DUPLICATES_ON: "bin_out_spec",
                    },
                    KEY_AGG_RES_TYPE: AggResType.SNAPS,
                },
                Indexer("key4_most_default"): {
                    KEY_BIN_ON_OUT: "ts_dflt",
                    KEY_POST: always_true,
                    KEY_MAX_IN_MEMORY_SIZE_B: 10485760,
                    KEY_WRITE_CONFIG: {
                        KEY_ROW_GROUP_TARGET_SIZE: 1000,
                        KEY_MAX_N_OFF_TARGET_RGS: 4,
                        KEY_ORDERED_ON: "ts_spec",
                        KEY_DUPLICATES_ON: "ts_dflt",
                    },
                    KEY_AGG_RES_TYPE: AggResType.SNAPS,
                },
            },
            # ref_agg_pd
            {
                Indexer("key1_some_default"): {"out_spec": ("in_spec", FIRST)},
                Indexer("key2_only_specific"): {"out_spec": ("in_spec", FIRST)},
                Indexer("key3_only_default"): {"out_dflt": ("in_dflt", LAST)},
                Indexer("key4_most_default"): {"out_dflt": ("in_dflt", LAST)},
            },
            # ref_filter_apps
            {
                "filter1": FilterApp(
                    [
                        Indexer("key1_some_default"),
                        Indexer("key2_only_specific"),
                        Indexer("key3_only_default"),
                    ],
                    "NA",
                ),
                "filter2": FilterApp([Indexer("key4_most_default")], "NA"),
            },
        ),
    ],
)
def test_aggstream_init(
    store,
    root_parameters,
    ref_seed_config,
    ref_keys_config,
    ref_agg_pd,
    ref_filter_apps,
):
    # Setup streamed aggregation.
    as_ = AggStream(store=store, **root_parameters)
    # Check
    assert as_.seed_config == ref_seed_config
    # Do not check 'seg_config' in 'keys_config'.
    res_keys_config = deepcopy(as_.keys_config)
    if KEY_SNAP_BY in root_parameters:
        # Check 'snap_by' is initialized in 'seg_config':
        ref_grouper = root_parameters[KEY_SNAP_BY]
        ref_grouper_attr = {
            "key": ref_grouper.key,
            "freq": ref_grouper.freq,
            "axis": ref_grouper.axis,
            "sort": ref_grouper.sort,
            "dropna": ref_grouper.dropna,
            "closed": ref_grouper.closed,
            "label": ref_grouper.label,
            "how": ref_grouper.how,
            "convention": ref_grouper.convention,
            "origin": ref_grouper.origin,
        }
        for key in ref_keys_config:
            res_grouper = res_keys_config[key][KEY_SEG_CONFIG][KEY_SNAP_BY]
            for attr in ref_grouper_attr:
                assert getattr(res_grouper, attr) == ref_grouper_attr[attr]
    for key, ref in ref_keys_config.items():
        del res_keys_config[key][KEY_SEG_CONFIG]
        assert res_keys_config[key] == ref
    ref_agg_buffers = {
        KEY_AGG_IN_MEMORY_SIZE: 0,
        KEY_SNAP_RES: None,
        KEY_BIN_RES: None,
        KEY_SNAP_RES_BUFFER: [],
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
    }
    for attr in main_ref_attr:
        assert getattr(as_, attr) == main_ref_attr[attr]
    for filt_id in ref_filter_apps:
        # Only checking list of keys, not number of parallel jobs as this
        # figure is platform dependent.
        assert as_.filter_apps[filt_id].keys == ref_filter_apps[filt_id].keys
        # But check number of parallel jobs is found as key in ``as_.p_jobs``.
        assert as_.filter_apps[filt_id].n_jobs in as_.p_jobs


def test_exception_not_key_of_streamagg_results(store):
    # Test error message provided key is not that of streamagg results.
    # Store some data using 'key'.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30"])
    ordered_on = "ts_order"
    val = range(1, len(ts) + 1)
    seed_pdf = DataFrame({ordered_on: ts, "val": val})
    store[key].write(ordered_on=ordered_on, df=seed_pdf)
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1h", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    with pytest.raises(ValueError, match="^provided 'agg_res'"):
        AggStream(
            ordered_on=ordered_on,
            agg=agg,
            store=store,
            keys=key,
            bin_by=bin_by,
        )


@pytest.mark.parametrize(
    "other_parameters, exception_mess",
    [
        # 0 / No 'bin_by'
        (
            {"keys": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}}},
            "^'bin_by' parameter is missing",
        ),
        # 1 / Parameter not in 'AggStream.__init__' nor in 'write' signature.
        # Because '**kwargs' is used in 'AggStream.__init__', test the
        # implemented check.
        (
            {
                "keys": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}},
                "my_invented_parameter": 0,
            },
            "^'my_invented_parameter' is neither",
        ),
        # 2 / Different filter ids between 'keys' and 'filters' parameters.
        (
            {
                "keys": {"filter1": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}}},
                "filters": {"filter2": ["val", ">=", 0]},
            },
            "^not possible to have different lists of filter ids",
        ),
        # 3 / Filter syntax used in 'keys' parameter, without 'filters' parameter.
        (
            {
                "keys": {"filter1": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}}},
            },
            "^not possible to use filter syntax for `keys`",
        ),
        # 4 / Use "no filter" filter id in 'filters' to set a filter.
        (
            {
                "keys": {NO_FILTER_ID: {key: {"agg": {"out_spec": ("in_spec", FIRST)}}}},
                "filters": {NO_FILTER_ID: ["val", ">=", 0]},
            },
            f"^not possible to use '{NO_FILTER_ID}'",
        ),
        # 5 / Same key for different filter ids.
        (
            {
                "keys": {
                    "filter1": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}},
                    "filter2": {key: {"agg": {"out_spec": ("in_spec", FIRST)}}},
                },
                "filters": {
                    "filter1": ["val", ">=", 0],
                    "filter2": ["val", ">=", 2],
                },
            },
            "^not possible to have key",
        ),
    ],
)
def test_exception_aggstream_parameters(store, other_parameters, exception_mess):
    ordered_on = "ts"
    conf = {
        "store": store,
        "ordered_on": ordered_on,
    } | other_parameters
    # Test.
    with pytest.raises(ValueError, match=exception_mess):
        AggStream(
            **conf,
        )
