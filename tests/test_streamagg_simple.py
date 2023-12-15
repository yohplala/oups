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
from pandas import NA as pNA
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import NaT as pNaT
from pandas import Series as pSeries
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat
from pandas.core.resample import TimeGrouper
from vaex import concat as vconcat
from vaex import from_pandas

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.cumsegagg import DTYPE_NULLABLE_INT64
from oups.jcumsegagg import FIRST
from oups.jcumsegagg import LAST
from oups.jcumsegagg import MAX
from oups.jcumsegagg import MIN
from oups.jcumsegagg import SUM
from oups.segmentby import by_x_rows
from oups.streamagg import KEY_LAST_SEED_INDEX
from oups.streamagg import KEY_POST_BUFFER
from oups.streamagg import KEY_STREAMAGG


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class Indexer:
    dataset_ref: str


def test_parquet_seed_time_grouper_sum_agg(tmp_path):
    # Test with parquet seed, time grouper and 'sum' aggregation.
    # No post.
    # Creation & 1st append, 'discard_last=True',
    # 2nd append, 'discard_last=False'.
    #
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RGS   TS   VAL ROW BIN LABEL | comments
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
    #  7  14:00  16   15   7 14:00 | no stitching, 1 aggregated row.
    #     14:20  17                |
    #
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
        ],
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
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg_col = SUM
    agg = {agg_col: ("val", SUM)}
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
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
    dti_ref = DatetimeIndex(
        [
            date + "08:00",
            date + "09:00",
            date + "10:00",
            date + "11:00",
            date + "12:00",
            date + "13:00",
            date + "14:00",
        ],
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp, and 'post_buffer' is empty.
    # 'segagg_buffer' is not checked as being not part of 'streamagg' scope,
    # but 'cumsegagg' scope.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    assert not streamagg_md[KEY_POST_BUFFER]
    # 1st append.
    # Complete seed_df with new data and continue aggregation.
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG    TS   VAL ROW BIN LABEL | comments
    #     14:00  16        1 14:00 | one-but-last row from prev seed data
    #     14:20  17                | last row from previous seed data
    #     15:10   1        2 15:00 | no stitching
    #     15:11   2        2 15:00 |
    ts = DatetimeIndex([date + "15:10", date + "15:11"])
    seed_df = pDataFrame({ordered_on: ts, "val": [1, 2]})
    fp_write(seed_path, seed_df, file_scheme="hive", append=True)
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [5, 3]
    assert n_rows_res == n_rows_ref
    # Check aggregated results: last row has been discarded with 'discard_last'
    # `True`.
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 33, 1]
    dti_ref = DatetimeIndex(
        [
            date + "08:00",
            date + "09:00",
            date + "10:00",
            date + "11:00",
            date + "12:00",
            date + "13:00",
            date + "14:00",
            date + "15:00",
        ],
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    assert not streamagg_md[KEY_POST_BUFFER]
    # 2nd append, with 'discard_last=False'.
    # Check aggregation till the end of seed data.
    # Check no 'last_seed_index' in metadata of aggregated results.
    ts = DatetimeIndex([date + "15:20", date + "15:21"])
    seed_df = pDataFrame({ordered_on: ts, "val": [11, 12]})
    fp_write(seed_path, seed_df, file_scheme="hive", append=True)
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=False,
        max_row_group_size=max_row_group_size,
    )
    # Test results (not trimming seed data).
    ref_res = seed.to_pandas().groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]


def test_vaex_seed_time_grouper_sum_agg(tmp_path):
    # Test with vaex seed, time grouper and 'sum' aggregation.
    # No post.
    # The 1st append, 'discard_last=True', 2nd append, 'discard_last=False'.
    #
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RGS   TS   VAL ROW BIN LABEL | comments
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
    #  5  13:00  13   12   6 13:00 |
    #  6  13:20  14   13           |
    #     13:40  15                |
    #  7  14:00  16   15   7 14:00 | no stitching, 1 aggregated row.
    #     14:20  17                |
    #
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
        ],
    )
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg_col = SUM
    agg = {agg_col: ("val", SUM)}
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [4, 3]
    assert n_rows_res == n_rows_ref
    # Check aggregated results: last row has been discarded with 'discard_last'
    # `True`.
    dti_ref = DatetimeIndex(
        [
            date + "08:00",
            date + "09:00",
            date + "10:00",
            date + "11:00",
            date + "12:00",
            date + "13:00",
            date + "14:00",
        ],
    )
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 16]
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    assert not streamagg_md[KEY_POST_BUFFER]
    # 1st append.
    # Complete seed_df with new data and continue aggregation.
    # 'discard_last=False'
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG    TS   VAL ROW BIN LABEL | comments
    #     14:00  16        1 14:00 | one-but-last row from prev seed data
    #     14:20  17                | last row from previous seed data
    #     15:10   1        2 15:00 | no stitching
    #     15:11   2        2 15:00 |
    ts = DatetimeIndex([date + "15:10", date + "15:11"])
    seed_pdf2 = pDataFrame({ordered_on: ts, "val": [1, 2]})
    seed_vdf = seed_vdf.concat(from_pandas(seed_pdf2))
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=False,
        max_row_group_size=max_row_group_size,
    )
    # Check aggregated results: last row has not been discarded.
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 33, 3]
    dti_ref = DatetimeIndex(
        [
            date + "08:00",
            date + "09:00",
            date + "10:00",
            date + "11:00",
            date + "12:00",
            date + "13:00",
            date + "14:00",
            date + "15:00",
        ],
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]


def test_parquet_seed_time_grouper_first_last_min_max_agg(tmp_path):
    # Test with parquet seed, time grouper and 'first', 'last', 'min', and
    # 'max' aggregation. No post, 'discard_last=True'.
    # 'Stress test' with appending new data twice.
    max_row_group_size = 6
    start = Timestamp("2020/01/01")
    rr = np.random.default_rng(1)
    N = 20
    rand_ints = rr.integers(100, size=N)
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    ordered_on = "ts"
    seed_df = pDataFrame(
        {ordered_on: ts + ts, "val": np.append(rand_ints, rand_ints + 1)},
    ).sort_values(ordered_on)
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="5T", closed="left", label="left")
    agg = {
        FIRST: ("val", FIRST),
        LAST: ("val", LAST),
        MIN: ("val", MIN),
        MAX: ("val", MAX),
    }
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = seed_df.iloc[:-2].groupby(bin_by).agg(**agg).reset_index()
    ref_res[[FIRST, LAST, MIN, MAX]] = ref_res[[FIRST, LAST, MIN, MAX]].astype(DTYPE_NULLABLE_INT64)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100}).sort_values(ordered_on)
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
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = pconcat([seed_df, seed_df2]).iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    ref_res[[FIRST, LAST, MIN, MAX]] = ref_res[[FIRST, LAST, MIN, MAX]].astype(DTYPE_NULLABLE_INT64)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 2nd append of new data.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400}).sort_values(ordered_on)
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
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = (
        pconcat([seed_df, seed_df2, seed_df3]).iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    )
    ref_res[[FIRST, LAST, MIN, MAX]] = ref_res[[FIRST, LAST, MIN, MAX]].astype(DTYPE_NULLABLE_INT64)
    rec_res = store[key]
    assert rec_res.pdf.equals(ref_res)
    n_rows_res = [rg.num_rows for rg in rec_res.pf.row_groups]
    n_rows_ref = [5, 3, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4]
    assert n_rows_res == n_rows_ref


def test_vaex_seed_time_grouper_first_last_min_max_agg(tmp_path):
    # Test with vaex seed, time grouper and 'first', 'last', 'min', and
    # 'max' aggregation. No post, 'discard_last=True'. 'Stress test' with
    # appending new data twice.
    max_row_group_size = 6
    start = Timestamp("2020/12/31")
    rr = np.random.default_rng(2)
    N = 20
    rand_ints = rr.integers(100, size=N)
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    ordered_on = "ts"
    seed_pdf = pDataFrame(
        {ordered_on: ts + ts, "val": np.append(rand_ints, rand_ints + 1)},
    ).sort_values(ordered_on)
    # Forcing dtype of 'seed_pdf' to float.
    seed_pdf["val"] = seed_pdf["val"].astype("float64")
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="5T", closed="left", label="left")
    agg = {
        FIRST: ("val", FIRST),
        LAST: ("val", LAST),
        MIN: ("val", MIN),
        MAX: ("val", MAX),
    }
    # Setup streamed aggregation.
    streamagg(
        seed=(max_row_group_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = seed_pdf.iloc[:-2].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append of new data.
    start = seed_pdf[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_pdf2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100}).sort_values(ordered_on)
    seed_vdf = vconcat([seed_vdf, from_pandas(seed_pdf2)])
    # Setup streamed aggregation.
    streamagg(
        seed=(max_row_group_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = pconcat([seed_pdf, seed_pdf2]).iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 2nd append of new data.
    start = seed_pdf2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_pdf3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400}).sort_values(ordered_on)
    seed_vdf = vconcat([seed_vdf, from_pandas(seed_pdf3)])
    # Setup streamed aggregation.
    streamagg(
        seed=(max_row_group_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = (
        pconcat([seed_pdf, seed_pdf2, seed_pdf3]).iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    )
    rec_res = store[key]
    assert rec_res.pdf.equals(ref_res)
    n_rows_res = [rg.num_rows for rg in rec_res.pf.row_groups]
    n_rows_ref = [4, 6, 3, 4, 5, 5, 4, 4, 3, 4, 5, 4, 3, 3]
    assert n_rows_res == n_rows_ref


def test_parquet_seed_duration_weighted_mean_from_post(tmp_path):
    # Test with parquet seed, time grouper and assess with 'post':
    #  - assess a 'duration' with 'first' and 'last' aggregation,
    #  - assess a 'weighted mean' by using 'sum' aggregation,
    #  - keep previous weighted mean in a specific column,
    #  - remove all columns from aggregation
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RGS   TS   VAL WEIGHT ROW BIN LABEL | comments
    #  1   8:00   1       1   0   1  8:00 | 2 aggregated rows from same row group.
    #      8:30   2       2               |
    #      9:00   3       1       2  9:00 |
    #      9:30   4       0               |
    #  2  10:00   5       2   4   3 10:00 | no stitching with previous.
    #     10:20   6       1               | 1 aggregated row.
    #  3  10:40   7       2   6           | stitching with previous (same bin).
    #     11:00   8       1       4 11:00 | 2 aggregated rows, not incl. stitching
    #     11:30   9       0               | with prev agg row.
    #     12:00  10       3       5 12:00 |
    #  4  12:20  11       2  10           | stitching with previous (same bin).
    #     12:40  12       1               | 0 aggregated row, not incl stitching.
    #  5  13:00  13       3  12   6 13:00 | no stitching, 1 aggregated row.
    #     --------------------------------- write data (max_row_group_size = 6)
    #  6  13:20  14       0  13   1       | stitching, 1 aggregated row.
    #     13:40  15       1               |
    #  7  14:00  16       2  15   7 14:00 | no stitching, 1 aggregated row.
    #     14:20  17       1               |
    #
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
        ],
    )
    ordered_on = "ts"
    weights = [1, 2, 1, 0, 2, 1, 2, 1, 0, 3, 2, 1, 3, 0, 1, 2, 1]
    seed_df = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1), "weight": weights})
    # Setup weighted mean: need 'weight' x 'val'.
    seed_df["weighted_val"] = seed_df["weight"] * seed_df["val"]
    row_group_offsets = [0, 4, 6, 10, 12, 13, 15]
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=row_group_offsets, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {
        FIRST: ("ts", FIRST),
        LAST: ("ts", LAST),
        "sum_weight": ("weight", SUM),
        "sum_weighted_val": ("weighted_val", SUM),
    }

    # Setup 'post'.
    def post(buffer: dict, bin_res: pDataFrame):
        """
        Compute duration, weighted mean and keep track of data to buffer.
        """
        # Compute 'duration'.
        bin_res[LAST] = (bin_res[LAST] - bin_res[FIRST]).view("int64")
        # Compute 'weighted_mean'.
        bin_res["sum_weighted_val"] = bin_res["sum_weighted_val"] / bin_res["sum_weight"]
        # Rename column 'first' and remove the others.
        bin_res.rename(
            columns={
                "sum_weighted_val": "weighted_mean",
                LAST: "duration",
            },
            inplace=True,
        )
        # Remove un-used columns.
        bin_res.drop(columns=["sum_weight", FIRST], inplace=True)
        # Keep number of iterations in 'post' (to test 'post_buffer' is
        # correctly updated in place, and recorded)
        if "iter_num" not in buffer:
            buffer["iter_num"] = 1
        else:
            buffer["iter_num"] += 1
        return bin_res

    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        post=post,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Check resulting dataframe.
    # Get reference results, discarding last row, because of 'discard_last'.
    ref_res_agg = seed_df.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    ref_res_post = post({}, ref_res_agg)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_post)
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [5, 2]
    assert n_rows_res == n_rows_ref
    # Check metadata.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    assert streamagg_md[KEY_POST_BUFFER] == {"iter_num": 2}
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    rr = np.random.default_rng(3)
    N = 30
    rand_ints = rr.integers(600, size=N)
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame(
        {ordered_on: ts, "val": rand_ints + 100, "weight": rand_ints},
    ).sort_values(ordered_on)
    # Setup weighted mean: need 'weight' x 'val'.
    seed_df2["weighted_val"] = seed_df2["weight"] * seed_df2["val"]
    fp_write(
        seed_path,
        seed_df2,
        row_group_offsets=row_group_offsets,
        file_scheme="hive",
        append=True,
    )
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        post=post,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res_agg = pconcat([seed_df, seed_df2]).iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    ref_res_post = post({}, ref_res_agg)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_post)
    # Check metadata.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_POST_BUFFER] == {"iter_num": 3}


def test_parquet_seed_time_grouper_bin_on_as_tuple(tmp_path):
    # Test with parquet seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # Change group keys column name with 'bin_on' set as a tuple.
    max_row_group_size = 4
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
        ],
    )
    ordered_on = "ts"
    bin_on = "ts_bin"
    ts_open = "ts_open"
    seed_pdf = pDataFrame({ordered_on: ts, bin_on: ts, "val": range(1, len(ts) + 1)})
    row_group_offsets = [0, 4, 6]
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, row_group_offsets=row_group_offsets, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {ordered_on: (ordered_on, LAST), SUM: ("val", SUM)}
    # Streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        bin_on=(ordered_on, ts_open),
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = seed_pdf.iloc[:-1].groupby(bin_by).agg(**agg)
    ref_res.index.name = ts_open
    ref_res.reset_index(inplace=True)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append of new data.
    ts2 = DatetimeIndex([date + "12:30", date + "13:00", date + "13:30", date + "14:00"])
    seed_pdf2 = pDataFrame(
        {ordered_on: ts2, bin_on: ts2, "val": range(len(ts) + 1, len(ts) + len(ts2) + 1)},
    )
    fp_write(seed_path, seed_pdf2, file_scheme="hive", append=True)
    seed = ParquetFile(seed_path)
    # 2nd streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        bin_on=(ordered_on, ts_open),
        trim_start=True,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res = pconcat([seed_pdf, seed_pdf2]).iloc[:-1].groupby(bin_by).agg(**agg)
    ref_res.index.name = ts_open
    ref_res.reset_index(inplace=True)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_by_callable_wo_bin_on(tmp_path):
    # Test with vaex seed, binning every 4 rows with 'first', and 'max'
    # aggregation. No post, `discard_last` set `True`.
    # Additionally, shows an example of how 'bin_by' as callable can output a
    # pandas series which name is re-used straight away in aggregation results.
    # (is re-used as 'ordered_on' column)
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for agg, BIN: bins
    # RG    TS   VAL       ROW BIN LABEL | comments
    #      8:00   1          0   1  8:00 |
    #      8:30   2                      |
    #      9:00   3                      |
    #      9:30   4                      |
    #     10:00   5          4   2 10:00 |
    #     10:20   6                      |
    #     10:40   7                      |
    #     11:00   8                      |
    #     11:30   9          8   3 11:30 |
    #     12:00  10                      |
    #     12:20  11                      |
    #     12:40  12                      |
    #     13:00  13         12   4 13:00 |
    #     -------------------------------- write data (max_row_group_size = 4)
    #  2  13:20  14                      | buffer_binning = {nrows : 1}
    #     13:40  15                      |
    #     14:00  16                      |
    #     14:20  17         16   5 14:20 | not in 1st agg results because of
    #     14:20  18                      | 'discard_last' True
    #
    # Setup seed data.
    max_vdf_chunk_size = 13
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
            date + "14:20",
        ],
    )
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    # Forcing dtype of 'seed_pdf' to float.
    seed_pdf["val"] = seed_pdf["val"].astype("float64")
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup streamed aggregation.
    agg = {
        FIRST: ("val", FIRST),
        MAX: ("val", MAX),
    }
    max_row_group_size = 4
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=by_x_rows,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed = seed_pdf.iloc[:-2]
    bin_starts = np.array([0, 4, 8, 12])
    bins = pSeries(pNaT, index=np.arange(len(trimmed_seed)))
    bins.iloc[bin_starts] = ts[bin_starts]
    bins.ffill(inplace=True)
    ref_res_agg = trimmed_seed.groupby(bins).agg(**agg)
    ref_res_agg.index.name = ordered_on
    ref_res_agg.reset_index(inplace=True)
    # Test results
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_agg)
    # Check metadata.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]

    # 1st append of new data.
    # RG    TS   VAL       ROW BIN LABEL | comments
    #  2  14:20  17         16   5 14:20 | previous data
    #     14:20  18                      |

    #     14:20   1          1           | new data
    #     14:30   2                      |
    #     15:00   3          3   6 15:00 |
    #     15:30   4                      |
    #     16:00   5                      |
    #     16:40   6                      | not in agg results because of
    #     16:40   7          7   7 16:40 | 'discard_last' True
    ts2 = DatetimeIndex(
        [
            date + "14:20",
            date + "14:30",
            date + "15:00",
            date + "15:30",
            date + "16:00",
            date + "16:40",
            date + "16:40",
        ],
    )
    seed_pdf2 = pDataFrame({ordered_on: ts2, "val": range(1, len(ts2) + 1)})
    # Forcing dtype of 'seed_pdf' to float.
    seed_pdf2["val"] = seed_pdf2["val"].astype("float64")
    seed_pdf2 = pconcat([seed_pdf, seed_pdf2], ignore_index=True)
    seed_vdf2 = from_pandas(seed_pdf2)
    # Setup streamed aggregation.
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf2),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=by_x_rows,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed2 = seed_pdf2.iloc[:-2]
    bin_starts = np.array([0, 4, 8, 12, 16, 20])
    bins = pSeries(pNaT, index=np.arange(len(trimmed_seed2)))
    bins.iloc[bin_starts] = trimmed_seed2.iloc[bin_starts].loc[:, ordered_on]
    bins.ffill(inplace=True)
    ref_res_agg2 = trimmed_seed2.groupby(bins).agg(**agg)
    ref_res_agg2.index.name = ordered_on
    ref_res_agg2.reset_index(inplace=True)
    # Test results
    rec_res2 = store[key].pdf
    assert rec_res2.equals(ref_res_agg2)
    # Check metadata.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts2[-1]


def test_vaex_seed_by_callable_with_bin_on(tmp_path):
    # Test with vaex seed, binning every time a '1' appear in column 'val'.
    # `discard_last` set `True`.
    # Additionally, show an example of how 'bin_on' as a tuple is used to
    # rename column of group keys.
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for agg, BIN: bins
    # RG    TS   VAL       ROW BIN LABEL | comments
    #  1   8:00   1          0   1     1 |
    #      8:30   2                      |
    #      9:00   3                      |
    #      9:30   1          3   2     2 |
    #     10:00   5                      |
    #     10:20   6                      |
    #     10:40   7                      |
    #     11:00   8                      |
    #     11:30   1          8   3     3 |
    #     12:00  10                      |
    #     12:20  11                      |
    #     12:40  12                      |
    #     13:00   1         12   4     4 |
    #     -------------------------------- write data (max_row_group_size = 4)
    #  2  13:20  14                      | buffer_binning = {last_key : 4}
    #     13:40  15                      |
    #     14:00  16                      |
    #     14:20   1         16   5     5 | not in 1st agg results because of
    #     14:20  18                      | 'discard_last' True
    #
    # Setup seed data.
    max_vdf_chunk_size = 13
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
            date + "14:20",
        ],
    )
    ordered_on = "ts"
    bin_on = "val"
    val = np.arange(1, len(ts) + 1)
    val[3] = 1
    val[8] = 1
    val[12] = 1
    val[16] = 1
    seed_pdf = pDataFrame({ordered_on: ts, bin_on: val})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")

    # Setup binning.
    def by_1val(on: pDataFrame, buffer: dict):
        """
        Start a new bin each time a 1 is spot.

        Returns
        -------
        next_chunk_starts, 1d array of int
        bin_labels, pSeries, the label of each bin
        n_null_bins,int, number of empty bins: always 0.
        bin_closed: str, does not matter as no snapshotting.
        bin_ends: pSeries, does not matter as no snapshotting.
        unknown_last_bin_end: bool, does not matter as no snapshotting.

        """
        ordered_on = on.columns[1]
        group_on = on.columns[0]
        # Setup 1st key of groups from previous binning.
        if "last_key" not in buffer:
            first_lab = on[ordered_on].iloc[0]
        else:
            first_lab = buffer["last_key"]
        ncs = (on[group_on] == 1).to_numpy().nonzero()[0]
        group_keys = on[ordered_on].iloc[ncs]
        if len(ncs) == 0:
            ncs = np.array([len(on)], dtype=int)
            group_keys = pSeries([first_lab])
        elif ncs[0] == 0:
            # A 1 is right at the start. Omit this bin.
            ncs = np.append(ncs[1:], len(on))
        else:
            # 1st bin is one that was started before.
            ncs = np.append(ncs, len(on))
            group_keys = pconcat([pSeries([first_lab]), group_keys])
        # pSeries(np.arange(len(ncs)), dtype=int) + offset
        buffer["last_key"] = group_keys.iloc[-1]
        return ncs, group_keys, 0, "left", ncs, True

    # Setup streamed aggregation.
    max_row_group_size = 4
    agg = {
        "ts": ("ts", FIRST),
        MAX: ("val", MAX),
    }
    bin_out_col = "group_keys"
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=by_1val,
        bin_on=(bin_on, bin_out_col),
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )

    def agg_with_same_bin_labels(seed_pdf):
        """
        Aggregate while managing same bin labels based on `ordered_on`.

        The 2 last rows are discarded (hard-coded), and corresponds to using
        ``discard_last=True`` in `streamagg` while the 2 last row of the seed
        data have same value in `ordered_on` column.

        """

        # Get reference results, discarding last row, because of 'discard_last'.
        trimmed_seed = seed_pdf.iloc[:-2].copy(deep=True)
        ncs, _, _, _, _, _ = by_1val(trimmed_seed[[bin_on, ordered_on]], {})
        trimmed_seed.loc[trimmed_seed["val"] == 1, "bins"] = np.arange(len(ncs))
        if trimmed_seed["bins"].iloc[0] is pNA:
            trimmed_seed["bins"].iloc[0] = -1
        trimmed_seed["bins"] = trimmed_seed["bins"].ffill()
        ref_res_agg = trimmed_seed.groupby("bins").agg(**agg)
        ref_res_agg.index.name = bin_out_col
        ref_res_agg.reset_index(inplace=True)
        # Set correct bin labels (same label can be used for several bins)
        ref_res_agg[bin_out_col] = (
            trimmed_seed.loc[trimmed_seed["val"] == 1, ordered_on]
            .reset_index(drop=True)
            .astype("datetime64[ns]")
        )
        if ref_res_agg[bin_out_col].iloc[0] == -1:
            ref_res_agg[bin_out_col].iloc[0] = trimmed_seed[bin_out_col].iloc[0]
        return ref_res_agg

    # Test results
    ref_res_agg = agg_with_same_bin_labels(seed_pdf)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_agg)
    # 1st append of new data.
    # RG    TS   VAL       ROW BIN LABEL | comments
    #  2  14:20   1         16   5     5 | previous data
    #     14:20  18                      |

    #     14:20   1          1   6     6 | new data
    #     14:30   2                      |
    #     15:00   3                      |
    #     15:30   1          4   7     7 |
    #     16:00   5                      |
    #     16:40   6                      | not in agg results because of
    #     16:40   7                      | 'discard_last' True
    ts2 = DatetimeIndex(
        [
            date + "14:20",
            date + "14:30",
            date + "15:00",
            date + "15:30",
            date + "16:00",
            date + "16:40",
            date + "16:40",
        ],
    )
    val = np.arange(1, len(ts2) + 1)
    val[3] = 1
    seed_pdf = pconcat([seed_pdf, pDataFrame({ordered_on: ts2, bin_on: val})], ignore_index=True)
    seed_vdf = from_pandas(seed_pdf)
    # Setup streamed aggregation.
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=by_1val,
        bin_on=(bin_on, bin_out_col),
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Test results
    ref_res_agg = agg_with_same_bin_labels(seed_pdf)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_agg)
    # Check metadata.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts2[-1]


def test_parquet_seed_time_grouper_trim_start(tmp_path):
    # Test with parquet seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # Test 'trim_start=False' when appending.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30", date + "09:00", date + "09:30"])
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        trim_start=True,
        discard_last=True,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    # 1st append. 2nd stremagg with 'trim_start=False'.
    ts2 = DatetimeIndex([date + "09:00", date + "09:30", date + "10:00", date + "10:30"])
    seed_pdf2 = pDataFrame({ordered_on: ts2, "val": range(1, len(ts) + 1)})
    seed_path2 = os_path.join(tmp_path, "seed2")
    fp_write(seed_path2, seed_pdf2, file_scheme="hive")
    seed2 = ParquetFile(seed_path2)
    # Streamed aggregation.
    streamagg(
        seed=seed2,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        trim_start=False,
        discard_last=True,
    )
    # Test results.
    seed_pdf_ref = pconcat([seed_pdf.iloc[:-1], seed_pdf2])
    ref_res = seed_pdf_ref.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_time_grouper_trim_start(tmp_path):
    # Test with vaex seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # Test 'trim_start=False' when appending.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30", date + "09:00", date + "09:30"])
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        trim_start=True,
        discard_last=True,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    # 1st append. 2nd stremagg with 'trim_start=False'.
    ts2 = DatetimeIndex([date + "09:00", date + "09:30", date + "10:00", date + "10:30"])
    seed_pdf2 = pDataFrame({ordered_on: ts2, "val": range(1, len(ts) + 1)})
    seed_vdf2 = from_pandas(seed_pdf2)
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf2,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        trim_start=False,
        discard_last=True,
    )
    # Test results.
    seed_pdf_ref = pconcat([seed_pdf.iloc[:-1], seed_pdf2])
    ref_res = seed_pdf_ref.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_time_grouper_agg_first(tmp_path):
    # Test with vaex seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # 1st agg ends on a full bin (no stitching required when re-starting).
    # For such a use case, streamagg is actually no needed.
    date = "2020/01/01 "
    ordered_on = "ts"
    ts = DatetimeIndex([date + "08:00", date + "08:30", date + "09:00", date + "10:00"])
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append, starting a new bin.
    ts2 = DatetimeIndex([date + "10:20", date + "10:40", date + "11:00", date + "11:30"])
    seed_pdf2 = pDataFrame({ordered_on: ts2, "val": range(1, len(ts2) + 1)})
    seed_pdf2 = pconcat([seed_pdf, seed_pdf2])
    seed_vdf2 = from_pandas(seed_pdf2)
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf2,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
    )
    # Test results.
    ref_res = seed_pdf2.iloc[:-1].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_single_row(tmp_path):
    # Test with vaex seed, time grouper and 'first' aggregation.
    # Single row.
    # No post, 'discard_last=True'.
    date = "2020/01/01 "
    ordered_on = "ts"
    ts = DatetimeIndex([date + "08:00"])
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
    )
    # Test results.
    assert key not in store


def test_parquet_seed_single_row(tmp_path):
    # Test with parquet seed, time grouper and 'first' aggregation.
    # Single row.
    # No post, 'discard_last=True'.
    date = "2020/01/01 "
    ordered_on = "ts"
    ts = DatetimeIndex([date + "08:00"])
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
    )
    # Test results.
    assert key not in store


def test_parquet_seed_single_row_within_seed(tmp_path):
    # Test with parquet seed, time grouper and 'first' aggregation.
    # Single row in the middle of otherwise larger chunks.
    # No post, 'discard_last=True'.
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for agg, BIN: bins
    # RG    TS   VAL       ROW BIN LABEL | comments
    #  1   8:00   1          0           |
    #      8:30   2                      |
    #      9:00   3                      |
    #      9:30   4          3           |
    #     10:00   5                      |
    #     --------------------------------
    #  2  10:20   6                      | right in the middle of a bin
    #     --------------------------------
    #  3  10:40   7          6           |
    #     11:00   8                      |
    #     11:30   9          8           |
    #     12:00  10                      |
    #     12:20  11                      |
    #     12:40  12                      |
    #     13:00  13         12           |
    #     13:20  14                      |
    #     13:40  15                      |
    #     14:00  16                      |
    #     14:20  15         16           |
    #     14:20  18                      |
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
            date + "14:20",
        ],
    )
    ordered_on = "ts"
    val = np.arange(1, len(ts) + 1)
    seed_pdf = pDataFrame({ordered_on: ts, "val": val})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, row_group_offsets=[0, 5, 6], file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-2].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_time_grouper_duplicates_on_wo_bin_on(tmp_path):
    # Test with vaex seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # Test 'duplicates_on=[ordered_on]' (without 'bin_on')
    # No error should raise at recording.
    # This is not a standard use!
    date = "2020/01/01 "
    ts_order = DatetimeIndex([date + "08:00", date + "08:30", date + "09:00", date + "09:30"])
    ts_bin = ts_order + +Timedelta("40T")
    ordered_on = "ts_order"
    val = range(1, len(ts_order) + 1)
    bin_on = "ts_bin"
    seed_pdf = pDataFrame({ordered_on: ts_order, bin_on: ts_bin, "val": val})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    agg = {SUM: ("val", SUM)}

    def post(buffer: dict, bin_res: pDataFrame):
        """
        Remove 'bin_on' column.
        """
        # Rename column 'bin_on' into 'ordered_on' to have an 'ordered_on'
        # column, while 'removing' 'bin_on' one.
        # This is not a standard use, as to rename 'bin_on' column, one should
        # use 'bin_on' parameter in the form of a tuple.
        bin_res.rename(
            columns={bin_on: ordered_on},
            inplace=True,
        )
        return bin_res

    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=by_x_rows,
        post=post,
        discard_last=True,
        duplicates_on=[],
    )
    # Test results.
    ref_res = pDataFrame({ordered_on: ts_order[:1], "sum": [6]})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


def test_vaex_seed_bin_on_col_sum_agg(tmp_path):
    # Test with vaex seed, time grouper and 'sum' aggregation.
    # No post.
    # The 1st append, 'discard_last=True', 2nd append, 'discard_last=False'.
    #
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # RGS   VAL ROW BIN    TS | comments
    #  1      1    0   1  8:10 | 2 aggregated rows from same row group.
    #         2                |
    #         3        2  9:10 |
    #         4                |
    #  2      5    4   3 10:10 | no stitching with previous.
    #         6                | 1 aggregated row.
    #  3      7    6           | stitching with previous (same bin).
    #         8        4 11:10 | 2 aggregated rows, not incl. stitching
    #         9                | with prev agg row.
    #        10        5 12:10 |
    #  4     11   10           | stitching with previous (same bin).
    #        12                | 0 aggregated row, not incl stitching.
    #  5     13   12   6 13:10 |
    #  6     14   13           |
    #        15                |
    #  7     16   15   7 14:10 | no stitching, 1 aggregated row.
    #        17                |
    #
    # Setup seed data.
    max_row_group_size = 6
    date = "2020/01/01 "
    ts = DatetimeIndex(
        [
            date + "08:10",  # 1
            date + "08:10",  # 1
            date + "09:10",  # 2
            date + "09:10",  # 2
            date + "10:10",  # 3
            date + "10:10",  # 3
            date + "10:10",  # 3
            date + "11:10",  # 4
            date + "11:10",  # 4
            date + "12:10",  # 5
            date + "12:10",  # 5
            date + "12:10",  # 5
            date + "13:10",  # 6
            date + "13:10",  # 6
            date + "13:10",  # 6
            date + "14:10",  # 7
            date + "14:10",  # 7
        ],
    )
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    agg_col = SUM
    agg = {agg_col: ("val", SUM)}
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
    )
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [6]
    assert n_rows_res == n_rows_ref
    # Check aggregated results: last row has been discarded with 'discard_last'
    # `True`.
    ref_res = seed_pdf.iloc[:-2].groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]
    # 1st append.
    # Complete seed_df with new data and continue aggregation.
    # 'discard_last=False'
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG   VAL  ROW BIN    TS | comments
    #       16        1 14:10 | one-but-last row from prev seed data
    #       17                | last row from previous seed data
    #        1        2 15:10 | no stitching
    #        2        2 15:10 |
    ts = DatetimeIndex(
        [
            date + "15:10",  # 8
            date + "15:10",  # 8
        ],
    )
    seed_pdf2 = pDataFrame({ordered_on: ts, "val": [1, 2]})
    seed_vdf = seed_vdf.concat(from_pandas(seed_pdf2))
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        keys=key,
        bin_by=bin_by,
        discard_last=False,
        max_row_group_size=max_row_group_size,
    )
    # Check aggregated results: last row has not been discarded.
    seed_pdf2 = pconcat([seed_pdf, seed_pdf2])
    ref_res = seed_pdf2.groupby(bin_by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    streamagg_md = store[key]._oups_metadata[KEY_STREAMAGG]
    assert streamagg_md[KEY_LAST_SEED_INDEX] == ts[-1]


def test_exception_not_key_of_streamagg_results(tmp_path):
    # Test error message provided key is not that of streamagg results.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30"])
    ordered_on = "ts_order"
    val = range(1, len(ts) + 1)
    seed_pdf = pDataFrame({ordered_on: ts, "val": val})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Store some data using 'key'.
    store[key] = seed_vdf
    # Setup aggregation.
    bin_by = TimeGrouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {SUM: ("val", SUM)}
    with pytest.raises(ValueError, match="^provided 'agg_res'"):
        streamagg(
            seed=seed_vdf,
            ordered_on=ordered_on,
            agg=agg,
            store=store,
            keys=key,
            bin_by=bin_by,
        )


# TODO
# in test case with snapshot: when snapshot is a TimeGrouper, make sure that stitching
# works same as for bin: that empty snapshots are generated between 2 row groups.
# test case, test parameter value not in 'streamagg' nor in 'write' signature.
