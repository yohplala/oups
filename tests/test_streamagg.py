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
from pandas import Grouper
from pandas import Index as pIndex
from pandas import Series
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat
from vaex import concat as vconcat
from vaex import from_pandas

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.streamagg import VAEX
from oups.streamagg import _get_streamagg_md


# from pandas.testing import assert_frame_equal
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class Indexer:
    dataset_ref: str


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_parquet_seed_time_grouper_sum_agg(tmp_path, reduction1, reduction2):
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
    key = Indexer("agg_res")
    # Setup aggregation.
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
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
        reduction=reduction1,
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
        ]
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp.
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 16}, index=pIndex([ts[-2]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref
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
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
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
        ]
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp.
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 1}, index=pIndex([dti_ref[-1]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref
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
        key=key,
        by=by,
        discard_last=False,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results (not trimming seed data).
    ref_res = seed.to_pandas().groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp.
    (
        last_seed_index_res,
        _,
        _,
        _,
    ) = _get_streamagg_md(store[key])
    assert last_seed_index_res == ts[-1]


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_time_grouper_sum_agg(tmp_path, reduction1, reduction2):
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
        ]
    )
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg_col = "sum"
    agg = {agg_col: ("val", "sum")}
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
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
        ]
    )
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 16]
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_complete_index' is last-but-one timestamp (because of
    # 'discard_last').
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 16}, index=pIndex([ts[-2]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref
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
        key=key,
        by=by,
        discard_last=False,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
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
        ]
    )
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp (because of 'discard_last').
    (
        last_seed_index_res,
        _,
        _,
        _,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_parquet_seed_time_grouper_first_last_min_max_agg(tmp_path, reduction1, reduction2):
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
        {ordered_on: ts + ts, "val": np.append(rand_ints, rand_ints + 1)}
    ).sort_values(ordered_on)
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_df, row_group_offsets=max_row_group_size, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    by = Grouper(key=ordered_on, freq="5T", closed="left", label="left")
    agg = {
        "first": ("val", "first"),
        "last": ("val", "last"),
        "min": ("val", "min"),
        "max": ("val", "max"),
    }
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
        reduction=reduction1,
    )
    # Test results
    ref_res = seed_df.iloc[:-2].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame({ordered_on: ts, "val": rand_ints + 100}).sort_values(ordered_on)
    fp_write(
        seed_path, seed_df2, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
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
        reduction=reduction1,
    )
    # Test results
    ref_res = pconcat([seed_df, seed_df2]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 2nd append of new data.
    start = seed_df2[ordered_on].iloc[-1]
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df3 = pDataFrame({ordered_on: ts, "val": rand_ints + 400}).sort_values(ordered_on)
    fp_write(
        seed_path, seed_df3, row_group_offsets=max_row_group_size, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
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
        reduction=reduction2,
    )
    # Test results
    ref_res = pconcat([seed_df, seed_df2, seed_df3]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key]
    assert rec_res.pdf.equals(ref_res)
    n_rows_res = [rg.num_rows for rg in rec_res.pf.row_groups]
    n_rows_ref = [5, 3, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4]
    assert n_rows_res == n_rows_ref


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_time_grouper_first_last_min_max_agg(tmp_path, reduction1, reduction2):
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
    by = Grouper(key=ordered_on, freq="5T", closed="left", label="left")
    agg = {
        "first": ("val", "first"),
        "last": ("val", "last"),
        "min": ("val", "min"),
        "max": ("val", "max"),
    }
    # Setup streamed aggregation.
    streamagg(
        seed=(max_row_group_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Test results
    ref_res = seed_pdf.iloc[:-2].groupby(by).agg(**agg).reset_index()
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
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    ref_res = pconcat([seed_pdf, seed_pdf2]).iloc[:-1].groupby(by).agg(**agg).reset_index()
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
        key=key,
        by=by,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    ref_res = (
        pconcat([seed_pdf, seed_pdf2, seed_pdf3]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    )
    rec_res = store[key]
    assert rec_res.pdf.equals(ref_res)
    n_rows_res = [rg.num_rows for rg in rec_res.pf.row_groups]
    n_rows_ref = [5, 5, 5, 3, 4, 5, 4, 4, 3, 4, 5, 4, 3, 3]
    assert n_rows_res == n_rows_ref


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_parquet_seed_duration_weighted_mean_from_post(tmp_path, reduction1, reduction2):
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
        ]
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {
        "first": ("ts", "first"),
        "last": ("ts", "last"),
        "sum_weight": ("weight", "sum"),
        "sum_weighted_val": ("weighted_val", "sum"),
    }

    # Setup 'post'.
    def post(agg_res: pDataFrame, isfbn: bool, post_buffer: dict):
        """Compute duration, weighted mean and keep track of data to buffer."""
        # Compute 'duration'.
        agg_res["last"] = (agg_res["last"] - agg_res["first"]).view("int64")
        # Compute 'weighted_mean'.
        agg_res["sum_weighted_val"] = agg_res["sum_weighted_val"] / agg_res["sum_weight"]
        # Keep 'weighted_mean' from previous row and update 'post_buffer'.
        if post_buffer:
            if isfbn:
                first = post_buffer["last_weighted_mean"]
            else:
                first = post_buffer["prev_last_weighted_mean"]
        else:
            first = np.nan
        agg_res["first"] = agg_res["sum_weighted_val"].shift(1, fill_value=first)
        # Rename column 'first' and remove the others.
        agg_res.rename(
            columns={
                "first": "prev_weighted_mean",
                "sum_weighted_val": "weighted_mean",
                "last": "duration",
            },
            inplace=True,
        )
        # Update for next iteration.
        post_buffer["prev_last_weighted_mean"] = agg_res["prev_weighted_mean"].iloc[-1]
        post_buffer["last_weighted_mean"] = agg_res["weighted_mean"].iloc[-1]
        # Remove un-used columns.
        agg_res.drop(columns=["sum_weight"], inplace=True)
        return agg_res

    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        post=post,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Check resulting dataframe.
    # Get reference results, discarding last row, because of 'discard_last'.
    ref_res_agg = seed_df.iloc[:-1].groupby(by).agg(**agg).reset_index()
    ref_res_post = post(ref_res_agg, None, {})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_post)
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [5, 2]
    assert n_rows_res == n_rows_ref
    # Check metadata.
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(
        data={"first": ts[-2], "last": ts[-2], "sum_weight": [2], "sum_weighted_val": [32]},
        index=pIndex([ts[-2]], name=ordered_on),
    )
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {"prev_last_weighted_mean": 13.5, "last_weighted_mean": 16.0}
    assert post_buffer_res == post_buffer_ref
    # 1st append of new data.
    start = seed_df[ordered_on].iloc[-1]
    rr = np.random.default_rng(3)
    N = 30
    rand_ints = rr.integers(600, size=N)
    ts = [start + Timedelta(f"{mn}T") for mn in rand_ints]
    seed_df2 = pDataFrame(
        {ordered_on: ts, "val": rand_ints + 100, "weight": rand_ints}
    ).sort_values(ordered_on)
    # Setup weighted mean: need 'weight' x 'val'.
    seed_df2["weighted_val"] = seed_df2["weight"] * seed_df2["val"]
    fp_write(
        seed_path, seed_df2, row_group_offsets=row_group_offsets, file_scheme="hive", append=True
    )
    seed = ParquetFile(seed_path)
    # Setup streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        post=post,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    ref_res_agg = pconcat([seed_df, seed_df2]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    ref_res_post = post(ref_res_agg, None, {})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_post)


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_parquet_seed_time_grouper_bin_on_as_tuple(tmp_path, reduction1, reduction2):
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
        ]
    )
    ordered_on = "ts"
    bin_on = "ts_bin"
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
    by = Grouper(key=bin_on, freq="1H", closed="left", label="left")
    agg = {ordered_on: (ordered_on, "last"), "sum": ("val", "sum")}
    # Streamed aggregation.
    # Test error message as name of column to use for binning defined with 'by'
    # and with 'bin_on' is not the same.
    with pytest.raises(ValueError, match="^two different columns"):
        streamagg(
            seed=seed,
            ordered_on=ordered_on,
            agg=agg,
            store=store,
            key=key,
            by=by,
            bin_on=bin_on + "_",
            discard_last=True,
            max_row_group_size=max_row_group_size,
        )
    # Test with renamed column for group keys.
    ts_open = "ts_open"
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        bin_on=(bin_on, ts_open),
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Test results
    ref_res = seed_pdf.iloc[:-1].groupby(by).agg(**agg)
    ref_res.index.name = ts_open
    ref_res.reset_index(inplace=True)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # 1st append of new data.
    ts2 = DatetimeIndex([date + "12:30", date + "13:00", date + "13:30", date + "14:00"])
    seed_pdf2 = pDataFrame(
        {ordered_on: ts2, bin_on: ts2, "val": range(len(ts) + 1, len(ts) + len(ts2) + 1)}
    )
    fp_write(seed_path, seed_pdf2, file_scheme="hive", append=True)
    seed = ParquetFile(seed_path)
    # 2nd streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        bin_on=(bin_on, ts_open),
        trim_start=True,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Test results
    ref_res = pconcat([seed_pdf, seed_pdf2]).iloc[:-1].groupby(by).agg(**agg)
    ref_res.index.name = ts_open
    ref_res.reset_index(inplace=True)
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_by_callable_wo_bin_on(tmp_path, reduction1, reduction2):
    # Test with vaex seed, binning every 4 rows with 'first', and 'max'
    # aggregation. No post, `discard_last` set `True`.
    # Additionally, shows an example of how 'by' as callable can output a
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
        ]
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

    # Setup binning.
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

    # Setup streamed aggregation.
    agg = {
        "first": ("val", "first"),
        "max": ("val", "max"),
    }
    max_row_group_size = 4
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by_4rows,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed = seed_pdf.iloc[:-2]
    bins = by_4rows(trimmed_seed[ordered_on], {})
    ref_res_agg = seed_pdf.iloc[:-2].groupby(bins).agg(**agg).reset_index()
    # Test results
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_agg)
    # Check metadata.
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {"row_offset": 4, "last_key": ts[-6]}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(
        data={"first": 13.0, "max": 16.0},
        index=pIndex([ts[-6]], name=ordered_on),
    )
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref

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
        ]
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
        key=key,
        by=by_4rows,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed2 = seed_pdf2.iloc[:-2]
    bins = by_4rows(trimmed_seed2[ordered_on], {})
    ref_res_agg2 = seed_pdf2.iloc[:-1].groupby(bins).agg(**agg).reset_index()
    # Test results
    rec_res2 = store[key].pdf
    assert rec_res2.equals(ref_res_agg2)
    # Check binning buffer stored in metadata.
    (
        last_seed_index_res2,
        _,
        binning_buffer_res2,
        _,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref2 = ts2[-1]
    assert last_seed_index_res2 == last_seed_index_ref2
    binning_buffer_ref2 = {"row_offset": 3, "last_key": Timestamp("2020-01-01 15:00:00")}
    assert binning_buffer_res2 == binning_buffer_ref2


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_by_callable_with_bin_on(tmp_path, reduction1, reduction2):
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
        ]
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
    def by_1val(data: pDataFrame, buffer: dict):
        """Start a new bin each time a 1 is spot."""
        # A pandas Series is returned.
        # Its name does not matter as 'bin_on' in streamagg is a tuple which
        # 2nd item will define the column name for group keys.
        group_on = data.columns[1]
        # Setup 1st key of groups from previous binning.
        if "last_key" not in buffer:
            offset = 0
        else:
            offset = buffer["last_key"]
        group_keys = Series(np.zeros(len(data)), dtype=int)
        group_keys.loc[data[group_on] == 1] = 1
        group_keys = group_keys.cumsum() + offset
        buffer["last_key"] = group_keys.iloc[-1]
        return group_keys

    # Setup streamed aggregation.
    max_row_group_size = 4
    agg = {
        "ts": ("ts", "first"),
        "max": ("val", "max"),
    }
    bin_out_col = "group_keys"
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by_1val,
        bin_on=(bin_on, bin_out_col),
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed = seed_pdf.iloc[:-2]
    bins = by_1val(trimmed_seed[[ordered_on, bin_on]], {})
    ref_res_agg = seed_pdf.iloc[:-2].groupby(bins).agg(**agg)
    ref_res_agg.index.name = bin_out_col
    ref_res_agg.reset_index(inplace=True)
    # Test results
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_agg)
    # 1st append of new data.
    # RG    TS   VAL       ROW BIN LABEL | comments
    #  2  14:20  17         16   5     5 | previous data
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
        ]
    )
    val = np.arange(1, len(ts2) + 1)
    val[3] = 1
    seed_pdf2 = pDataFrame({ordered_on: ts2, bin_on: val})
    seed_pdf2 = pconcat([seed_pdf, seed_pdf2], ignore_index=True)
    seed_vdf2 = from_pandas(seed_pdf2)
    # Setup streamed aggregation.
    streamagg(
        seed=(max_vdf_chunk_size, seed_vdf2),
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by_1val,
        bin_on=(bin_on, bin_out_col),
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Get reference results, discarding last row, because of 'discard_last'.
    trimmed_seed2 = seed_pdf2.iloc[:-2]
    bins = by_1val(trimmed_seed2[[ordered_on, bin_on]], {})
    ref_res_agg2 = seed_pdf2.iloc[:-1].groupby(bins).agg(**agg)
    ref_res_agg2.index.name = bin_out_col
    ref_res_agg2.reset_index(inplace=True)
    ref_res_agg2[bin_out_col] = ref_res_agg2[bin_out_col].astype(int)
    # Test results
    rec_res2 = store[key].pdf
    assert rec_res2.equals(ref_res_agg2)
    # Check binning buffer stored in metadata.
    (
        last_seed_index_res2,
        _,
        binning_buffer_res2,
        _,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref2 = ts2[-1]
    assert last_seed_index_res2 == last_seed_index_ref2
    binning_buffer_ref2 = {"last_key": 7}
    assert binning_buffer_res2 == binning_buffer_ref2


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_parquet_seed_time_grouper_trim_start(tmp_path, reduction1, reduction2):
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation.
    streamagg(
        seed=seed,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        trim_start=True,
        discard_last=True,
        reduction=reduction1,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    last_seed_index_ref = ts[-1]
    last_seed_index_res, _, _, _ = _get_streamagg_md(store[key])
    assert last_seed_index_res == last_seed_index_ref
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
        key=key,
        by=by,
        trim_start=False,
        discard_last=True,
        reduction=reduction2,
    )
    # Test results.
    seed_pdf_ref = pconcat([seed_pdf.iloc[:-1], seed_pdf2])
    ref_res = seed_pdf_ref.iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_time_grouper_trim_start(tmp_path, reduction1, reduction2):
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        trim_start=True,
        discard_last=True,
        reduction=reduction1,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index'.
    last_seed_index_ref = ts[-1]
    last_seed_index_res, _, _, _ = _get_streamagg_md(store[key])
    assert last_seed_index_res == last_seed_index_ref
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
        key=key,
        by=by,
        trim_start=False,
        discard_last=True,
        reduction=reduction2,
    )
    # Test results.
    seed_pdf_ref = pconcat([seed_pdf.iloc[:-1], seed_pdf2])
    ref_res = seed_pdf_ref.iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_time_grouper_agg_first(tmp_path, reduction1, reduction2):
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        reduction=reduction1,
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-1].groupby(by).agg(**agg).reset_index()
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
        key=key,
        by=by,
        reduction=reduction2,
    )
    # Test results.
    ref_res = seed_pdf2.iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize("reduction", [False, True, VAEX])
def test_vaex_seed_single_row(tmp_path, reduction):
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        reduction=reduction,
    )
    # Test results.
    assert key not in store


@pytest.mark.parametrize("reduction", [False, True, VAEX])
def test_parquet_seed_single_row(tmp_path, reduction):
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed, ordered_on=ordered_on, agg=agg, store=store, key=key, by=by, reduction=reduction
    )
    # Test results.
    assert key not in store


@pytest.mark.parametrize("reduction", [False, True, VAEX])
def test_parquet_seed_single_row_within_seed(tmp_path, reduction):
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
        ]
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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    # Streamed aggregation: no aggregation, but no error message.
    streamagg(
        seed=seed, ordered_on=ordered_on, agg=agg, store=store, key=key, by=by, reduction=reduction
    )
    # Test results.
    ref_res = seed_pdf.iloc[:-2].groupby(by).agg(**agg).reset_index()
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize("reduction", [False, True, VAEX])
def test_vaex_seed_time_grouper_duplicates_on_wo_bin_on(tmp_path, reduction):
    # Test with vaex seed, time grouper and 'first' aggregation.
    # No post, 'discard_last=True'.
    # Test 'duplicates_on=[ordered_on]' (without 'bin_on')
    # and removing 'bin_on' during post.
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
    by = Grouper(key=bin_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}

    def post(agg_res: pDataFrame, isfbn: bool, post_buffer: dict):
        """Remove 'bin_on' column."""
        # Rename column 'bin_on' into 'ordered_on' to have an 'ordered_on'
        # column, while 'removing' 'bin_on' one.
        agg_res.rename(
            columns={bin_on: ordered_on},
            inplace=True,
        )
        return agg_res

    # Streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=by,
        post=post,
        discard_last=True,
        duplicates_on=[],
        reduction=reduction,
    )
    # Test results.
    ref_res = (
        seed_pdf.iloc[:-1].groupby(by).agg(**agg).reset_index().rename(columns={bin_on: ordered_on})
    )
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)


@pytest.mark.parametrize(
    "reduction1,reduction2",
    [(False, False), (True, True), (True, False), (VAEX, VAEX), (VAEX, False)],
)
def test_vaex_seed_bin_on_col_sum_agg(tmp_path, reduction1, reduction2):
    # Test with vaex seed, time grouper and 'sum' aggregation.
    # No post.
    # The 1st append, 'discard_last=True', 2nd append, 'discard_last=False'.
    #
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # RGS   TS   VAL ROW BIN LABEL | comments
    #  1      1   1    0   1  8:00 | 2 aggregated rows from same row group.
    #         1   2                |
    #         2   3        2  9:00 |
    #         2   4                |
    #  2      3   5    4   3 10:00 | no stitching with previous.
    #         3   6                | 1 aggregated row.
    #  3      3   7    6           | stitching with previous (same bin).
    #         4   8        4 11:00 | 2 aggregated rows, not incl. stitching
    #         4   9                | with prev agg row.
    #         5  10        5 12:00 |
    #  4      5  11   10           | stitching with previous (same bin).
    #         5  12                | 0 aggregated row, not incl stitching.
    #  5      6  13   12   6 13:00 |
    #  6      6  14   13           |
    #         6  15                |
    #  7      7  16   15   7 14:00 | no stitching, 1 aggregated row.
    #         7  17                |
    #
    # Setup seed data.
    max_row_group_size = 6
    ts = [1] * 2 + [2] * 2 + [3] * 3 + [4] * 2 + [5] * 3 + [6] * 3 + [7] * 2
    ordered_on = "ts"
    seed_pdf = pDataFrame({ordered_on: ts, "val": range(1, len(ts) + 1)})
    seed_vdf = from_pandas(seed_pdf)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    agg_col = "sum"
    agg = {agg_col: ("val", "sum")}
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=None,
        bin_on=ordered_on,
        discard_last=True,
        max_row_group_size=max_row_group_size,
        reduction=reduction1,
    )
    # Check number of rows of each row groups in aggregated results.
    pf_res = store[key].pf
    n_rows_res = [rg.num_rows for rg in pf_res.row_groups]
    n_rows_ref = [6]
    assert n_rows_res == n_rows_ref
    # Check aggregated results: last row has been discarded with 'discard_last'
    # `True`.
    dti_ref = [1, 2, 3, 4, 5, 6]
    agg_sum_ref = [3, 7, 18, 17, 33, 42]
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_complete_index' is last-but-one timestamp (because of
    # 'discard_last').
    (
        last_seed_index_res,
        last_agg_row_res,
        binning_buffer_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 42}, index=pIndex([ts[-3]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref
    # 1st append.
    # Complete seed_df with new data and continue aggregation.
    # 'discard_last=False'
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG    TS   VAL ROW BIN LABEL | comments
    #         7  16        1 14:00 | one-but-last row from prev seed data
    #         7  17                | last row from previous seed data
    #         8   1        2 15:00 | no stitching
    #         8   2        2 15:00 |
    ts = [8] * 2
    seed_pdf2 = pDataFrame({ordered_on: ts, "val": [1, 2]})
    seed_vdf = seed_vdf.concat(from_pandas(seed_pdf2))
    # Setup streamed aggregation.
    streamagg(
        seed=seed_vdf,
        ordered_on=ordered_on,
        agg=agg,
        store=store,
        key=key,
        by=None,
        bin_on=ordered_on,
        discard_last=False,
        max_row_group_size=max_row_group_size,
        reduction=reduction2,
    )
    # Check aggregated results: last row has not been discarded.
    agg_sum_ref = [3, 7, 18, 17, 33, 42, 33, 3]
    dti_ref = [1, 2, 3, 4, 5, 6, 7, 8]
    ref_res = pDataFrame({ordered_on: dti_ref, agg_col: agg_sum_ref})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res)
    # Check 'last_seed_index' is last timestamp (because of 'discard_last').
    (
        last_seed_index_res,
        _,
        _,
        _,
    ) = _get_streamagg_md(store[key])
    last_seed_index_ref = ts[-1]
    assert last_seed_index_res == last_seed_index_ref


def test_exception_bin_on(tmp_path):
    # Test error message when 'bin_on' is also a name for an output aggregation
    # column.
    date = "2020/01/01 "
    ordered_on = "ts"
    ts = DatetimeIndex([date + "08:00"])
    bin_on = "ts2"
    seed_pdf = pDataFrame({ordered_on: ts, bin_on: ts, "val": range(1, len(ts) + 1)})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, file_scheme="hive")
    seed = ParquetFile(seed_path)
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    key = Indexer("agg_res")
    # Setup aggregation.
    by = Grouper(key=bin_on, freq="1H", closed="left", label="left")
    agg = {bin_on: ("val", "sum")}
    # Streamed aggregation, check error message.
    with pytest.raises(ValueError, match="^not possible to have"):
        streamagg(seed=seed, ordered_on=ordered_on, agg=agg, store=store, key=key, by=by)


def test_exception_unknown_agg_function(tmp_path):
    # Test error message when agg func is not within those allowed.
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
    # Setup aggregation.
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "unknown")}
    with pytest.raises(ValueError, match="^aggregation function"):
        streamagg(seed=seed_vdf, ordered_on=ordered_on, agg=agg, store=store, key=key, by=by)


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
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {"sum": ("val", "sum")}
    with pytest.raises(ValueError, match="^provided key"):
        streamagg(seed=seed_vdf, ordered_on=ordered_on, agg=agg, store=store, key=key, by=by)


def test_exception_no_agg_in_keys(tmp_path):
    # Test error message when 'key' is a single key, without 'agg' parameter.
    date = "2020/01/01 "
    ts = DatetimeIndex([date + "08:00", date + "08:30"])
    ordered_on = "ts_order"
    val = range(1, len(ts) + 1)
    seed_pdf = pDataFrame({ordered_on: ts, "val": val})
    seed_path = os_path.join(tmp_path, "seed")
    fp_write(seed_path, seed_pdf, file_scheme="hive")
    # Setup oups parquet collection and key.
    store_path = os_path.join(tmp_path, "store")
    store = ParquetSet(store_path, Indexer)
    # Setup aggregation.
    with pytest.raises(ValueError, match="^not possible to use a single key"):
        streamagg(
            seed=seed_pdf,
            ordered_on=ordered_on,
            key=Indexer("agg_res"),
            store=store,
            by=Grouper(key=ordered_on, freq="1H", closed="left", label="left"),
        )
