#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from os import path as os_path

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper as TimeGrouper
from pandas import Index as pIndex

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.streamagg import _get_streamagg_md


# from pandas.testing import assert_frame_equal
# from vaex import from_pandas


# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class Indexer:
    dataset_ref: str


def test_parquet_seed_time_grouper_sum_agg(tmp_path):
    # Test with parquet seed, time grouper and 'sum' aggregation.
    # No post, no discard_last.
    #
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG    TS   VAL ROW BIN LABEL | comments
    #  1   8:00   1    0   1  8:00 | 2 aggregated rows from same row group.
    #      8:30   2                |
    #      9:00   3        2  9:00 |
    #      9:30   4                |
    #  2  10:00   5    4   3 10:00 | no stitching with previous.
    #     10:20   6                | 1 aggregated row.
    #  3  10:40   7    6           | stitching with previous (same bin).
    #     11:00   8        4 11:00 | 2 aggregated rows, not incl. stitching
    #     11:30   9                | with prev agg row.
    #     12:00  10        5 12:00 |
    #  4  12:20  11   10           | stitching with previous (same bin).
    #     12:40  12                | 0 aggregated row, not incl stitching.
    #  5  13:00  13   12   6 13:00 | no stitching, 1 aggregated row.
    #     -------------------------- write data (max_row_group_size = 6)
    #  6  13:20  14   13           | stitching, 1 aggregated row.
    #     13:40  15                |
    #  7  14:00  16   15   7 14:00 | no stitching, 1 aggrgated row.
    #     14:20  17                |
    # Setup seed data.
    max_row_group_size = 6
    date = "2020/01/01 "
    ts = DatetimeIndex(
        [
            date + "08:00",
            date + "08:30",
            date + "09:00",
            date + "09:30",
            date + "10:00",
            date + "10:20",
            date + "10:40",
            date + "11:00",
            date + "11:30",
            date + "12:00",
            date + "12:20",
            date + "12:40",
            date + "13:00",
            date + "13:20",
            date + "13:40",
            date + "14:00",
            date + "14:20",
        ]
    )
    dti_ref = DatetimeIndex(
        [
            date + "08:00",
            date + "09:00",
            date + "10:00",
            date + "11:00",
            date + "12:00",
            date + "13:00",
            date + "14:00",
        ]
    )
    ordered_on = "ts"
    seed_df = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    row_group_offsets = [0, 4, 6, 10, 12, 13, 15]
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=row_group_offsets, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("seed")
    # Setup aggregation.
    by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg_col = "sum"
    agg = {agg_col: ("val", "sum")}
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [5, 2]
    assert n_rows_res == n_rows_ref
    # Check aggregated results: last row has been discarded with 'discard_last'
    # `True`.
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 16]
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_complete_index' is last-but-one timestamp (because of
    # 'discard_last').
    last_complete_index_res, binning_buffer_res, last_agg_row_res = _get_streamagg_md(store[key])
    last_complete_index_ref = pDataFrame({ordered_on: [ts[-2]]})
    assert last_complete_index_res.equals(last_complete_index_ref)
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 16}, index=pIndex([ts[-2]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)


# WiP
# test with ParquetFile seed
# test with vaex seed

# Test ValueError when not discard_last and not last_seed_index in seed metadata.

# discard_last : seed_index_end correctly taken into account?

# test when restarting with existing aggregated data, seed_index_start correctly taken into account?

# from seed data, test with additional columns and check only cols_in to be used are
# indeed loaded.

# Test with 'by' as callable, with 'buffer_binning' and without 'buffer_binning'.

# Test avec streamagg 1 se terminant exactement sure la bin en cours, & streamagg 2 reprenant sure une nouvelle bin
# Et quand streamagg est utile. (itération 2 démarrée au milieu de bin 1 par example)

# test with a single new rowin seed data.


# Test error bin_on not defined, but 'by' is a callable

# Test error message if 'bin_on' is already used as an output column name from aggregation

# test case when one aggregation chunk is a single line and is not agged with next aggregation result (for instance
# in row groups of seed data, a single bin / single row, and next row group of seed data is a new bin)

# test with "last_complete_seed_index": use a streamagg result within the store
# does jsonification work with a Timestamp?

# test with 'duplicates_on' set, without 'bin_on' to check when result are recorded:
# no bin_on (to be removed during post') and that it works.

# Test min, max, sum, first, last
# Test error message agg func is not within above values min, max, sum, first, last
