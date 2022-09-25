#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""

from numpy import dtype
from pandas import DataFrame as pDataFrame
from pandas import Timestamp as pTimestamp

from oups.chaingroupby import setup_cgb_agg


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


def test_setup_cgb_agg():
    # Test config generation for aggregation step in chaingroupby.
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
        "val1_first": ("val1_float", "first"),
        "val2_first": ("val2_float", "first"),
        "val2_sum": ("val2_float", "sum"),
        "val4_first": ("val4_datetime", "first"),
        "val3_last": ("val3_int", "last"),
    }
    cgb_agg_cfg_res = setup_cgb_agg(agg_cfg, df.dtypes.to_dict())
    cgb_agg_cfg_ref = {
        dtype("float64"): [
            ["val1_float", "val2_float"],
            [(0, "first"), (1, "first"), (1, "sum")],
            ["val1_first", "val2_first", "val2_sum"],
        ],
        dtype("<M8[ns]"): [["val4_datetime"], [(0, "first")], ["val4_first"]],
        dtype("int64"): [["val3_int"], [(0, "last")], ["val3_last"]],
    }
    assert cgb_agg_cfg_res == cgb_agg_cfg_ref
