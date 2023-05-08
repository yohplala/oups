#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""

import pytest
from numpy import array
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat

from oups.cumsegagg import DTYPE_DATETIME64
from oups.cumsegagg import DTYPE_FLOAT64
from oups.cumsegagg import DTYPE_INT64
from oups.cumsegagg import DTYPE_NULLABLE_INT64
from oups.cumsegagg import KEY_LAST_CHUNK_RES
from oups.cumsegagg import cumsegagg
from oups.cumsegagg import setup_cumsegagg
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import SUM
from oups.segmentby import setup_segmentby


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@pytest.mark.parametrize(
    "end_indices, bin_by, last_chunk_res_ref, indices_of_null_res",
    [
        # 1/ 15mn bin left closed; right label
        #  'data'
        #  datetime value  qty  bins     row_idx (group)
        #      8:10   4.0    4  b1-8:15  0 (0)
        #      8:10   4.2    3  b1
        #      8:12   3.9    1  b1
        #      8:17   5.6    7  b2-8:30
        #      8:19   6.0    2  b2
        #      8:20   9.8    6  b2       5
        #                       b3-8:45  6 (1)
        #      9:00   4.5    2  b4-9:00
        #      9:10   1.1    8  b4
        #                       b5-9:15
        #      9:30   3.2    1  b6-9:30
        (
            [6, 11],
            Grouper(freq="15T", key="dti", closed="left", label="right"),
            [pDataFrame({FIRST: [5.6], SUM: [15]}), pDataFrame({FIRST: [3.2], SUM: [1]})],
            [
                pTimestamp("2020-01-01 08:45:00"),
                pTimestamp("2020-01-01 09:00:00"),
                pTimestamp("2020-01-01 09:30:00"),
            ],
        )
    ],
)
def test_cumsegagg_bin_with_null(end_indices, bin_by, last_chunk_res_ref, indices_of_null_res):
    # Test binning with null chunks.
    # 'data' as follow
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
        SUM: (qty, SUM),
    }
    # Reference results.
    bin_res_ref = data.groupby(bin_by).agg(**agg)
    # Update null int values.
    bin_res_ref[SUM] = bin_res_ref[SUM].astype(DTYPE_NULLABLE_INT64)
    bin_res_ref.loc[indices_of_null_res, SUM] = pNA
    # Initialize.
    agg = setup_cumsegagg(agg, data.dtypes.to_dict())
    bin_by = setup_segmentby(bin_by)
    start_idx = 0
    buffer = {}
    bin_res_to_concatenate = []
    # Run in loop.
    for i, end_idx in enumerate(end_indices):
        bin_res = cumsegagg(
            data=data.iloc[start_idx:end_idx], agg=agg, bin_by=bin_by, buffer=buffer
        )
        assert buffer[KEY_LAST_CHUNK_RES].equals(last_chunk_res_ref[i])
        bin_res_to_concatenate.append(bin_res)
        start_idx = end_idx
    bin_res = pconcat(bin_res_to_concatenate)
    bin_res = bin_res[~bin_res.index.duplicated(keep="last")]
    assert bin_res.equals(bin_res_ref)


# Questions:
#  - in cumsegagg, when restarting with 3 empty chunks, assuming the
#    there are 2 'snaps', and the 3rd is a 'bin' (which was in progress at
#    prev iter.)
#    With proposed methodo, it is not enough to simply let the 'in-progress data'
#    from previous iteration. the new intermediate chunk needs to be created.
#
#  We should really just restart jcsagg directly, without removing empyt bins
#  If values are the same, then they are the same and that's it.
# 'pinnu' should be forwarded with its last value / should work.
# length will be detected 0
# Make test
#   - with empty snap at prev iter, then new empty snaps at next iter then end of bin.
#   - with non empty snap at prev iter, then new empty one (res forwarded?) at next iter then end of bin
# replay also test case segmentby() 5
# 5/ 'bin_by' using 'by_x_rows', 'snap_by' as Series.
#    In this test case, 'unknown_bin_end' is False after 1st data
#    chunk.
#    Last rows in 'data' are not accounted for in 1st series of
#    snapshots (8:20 snasphot not present), but is in 2nd series.
