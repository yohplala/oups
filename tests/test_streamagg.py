#!/usr/bin/env python3
"""
Created on Sun Mar 13 18:00:00 2022.

@author: yoh
"""
from os import path as os_path

import numpy as np
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Grouper
from pandas import Index as pIndex
from pandas import Timedelta
from pandas import Timestamp
from pandas import concat as pconcat
from vaex import concat as vconcat
from vaex import from_pandas

from oups import ParquetSet
from oups import streamagg
from oups import toplevel
from oups.streamagg import _get_streamagg_md


# from pandas.testing import assert_frame_equal
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
    (
        last_complete_index_res,
        binning_buffer_res,
        last_agg_row_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_complete_index_ref = pDataFrame({ordered_on: [ts[-2]]})
    assert last_complete_index_res.equals(last_complete_index_ref)
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 16}, index=pIndex([ts[-2]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref
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
    # Check 'last_complete_index' is last-but-one timestamp (because of
    # 'discard_last').
    (
        last_complete_index_res,
        binning_buffer_res,
        last_agg_row_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_complete_index_ref = pDataFrame({ordered_on: [ts[-2]]})
    assert last_complete_index_res.equals(last_complete_index_ref)
    binning_buffer_ref = {}
    assert binning_buffer_res == binning_buffer_ref
    last_agg_row_ref = pDataFrame(data={agg_col: 1}, index=pIndex([dti_ref[-1]], name=ordered_on))
    assert last_agg_row_res.equals(last_agg_row_ref)
    post_buffer_ref = {}
    assert post_buffer_res == post_buffer_ref


def test_parquet_seed_time_grouper_first_last_min_max_agg(tmp_path):
    # Test with parquet seed, time grouper and 'first', 'last', 'min', and
    # 'max' aggregation. No post, no discard_last.
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
    key = Indexer("seed")
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
    )
    # Test results
    ref_res = pconcat([seed_df, seed_df2, seed_df3]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    rec_res = store[key]
    assert rec_res.pdf.equals(ref_res)
    n_rows_res = [rg.num_rows for rg in rec_res.pf.row_groups]
    n_rows_ref = [5, 3, 4, 3, 4, 4, 4, 4, 3, 4, 4, 4, 4, 3, 4]
    assert n_rows_res == n_rows_ref


def test_vaex_seed_time_grouper_first_last_min_max_agg(tmp_path):
    # Test with vaex seed, time grouper and 'first', 'last', 'min', and
    # 'max' aggregation. No post, no discard_last. 'Stress test' with appending
    # new data twice.
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
    key = Indexer("seed")
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


def test_parquet_seed_duration_weighted_mean_from_post(tmp_path):
    # Test with parquet seed, time grouper and assess with 'post':
    #  - assess a 'duration' with 'first' and 'last' aggregation,
    #  - assess a 'weighted mean' by using 'sum' aggregation,
    #  - keep previous weighted mean in a specific column,
    #  - remove all columns from aggregation
    # Seed data
    # RGS: row groups, TS: 'ordered_on', VAL: values for 'sum' agg, BIN: bins
    # 1 hour binning
    # RG    TS   VAL WEIGHT ROW BIN LABEL | comments
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
    #  7  14:00  16       2  15   7 14:00 | no stitching, 1 aggrgated row.
    #     14:20  17       1               |
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
    key = Indexer("seed")
    # Setup aggregation.
    by = Grouper(key=ordered_on, freq="1H", closed="left", label="left")
    agg = {
        "first": ("ts", "first"),
        "last": ("ts", "last"),
        "sum_weight": ("weight", "sum"),
        "sum_weighted_val": ("weighted_val", "sum"),
    }

    # Setup 'post'.
    def post(agg_res: pDataFrame, isfrn: bool, post_buffer: dict):
        """Compute duration, weighted mean and keep track of data to buffer."""
        # Compute 'duration'.
        agg_res["last"] = (agg_res["last"] - agg_res["first"]).view("int64")
        # Compute 'weighted_mean'.
        agg_res["sum_weighted_val"] = agg_res["sum_weighted_val"] / agg_res["sum_weight"]
        # Keep weighted_mean from previous row and update 'post_buffer'.
        if post_buffer:
            if isfrn:
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
    )
    # Check resulting dataframe.
    # Get reference results, discarding last row, because of 'discard_last'.
    ref_res_agg = seed_df.iloc[:-1, :].groupby(by).agg(**agg).reset_index()
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
        last_complete_index_res,
        binning_buffer_res,
        last_agg_row_res,
        post_buffer_res,
    ) = _get_streamagg_md(store[key])
    last_complete_index_ref = pDataFrame({ordered_on: [ts[-2]]})
    assert last_complete_index_res.equals(last_complete_index_ref)
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
    )
    # Test results
    ref_res_agg = pconcat([seed_df, seed_df2]).iloc[:-1].groupby(by).agg(**agg).reset_index()
    ref_res_post = post(ref_res_agg, None, {})
    rec_res = store[key].pdf
    assert rec_res.equals(ref_res_post)


# WiP
# test with 'by' as callable, with 'buffer_binning' and without 'buffer_binning'.
# test with discard_last = False:
#   - 1st a streamagg that will be used as seed
#   - then a 2nd streamagg with discard_last = False


# Test ValueError when not discard_last and not last_seed_index in seed metadata.

# discard_last : seed_index_end correctly taken into account?


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

# Test error message agg func is not within above values min, max, sum, first, last
