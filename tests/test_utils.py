#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh
"""
import zipfile
from os import path as os_path

import pytest
from numpy import all as nall
from numpy import array
from pandas import DataFrame
from pandas import Grouper
from pandas import Timestamp

from oups.defines import DIR_SEP
from oups.utils import files_at_depth
from oups.utils import merge_sorted
from oups.utils import tcut

from . import TEST_DATA


def test_files_at_depth(tmp_path):
    fn = os_path.join(TEST_DATA, "dummy_store.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    basepath = os_path.join(tmp_path, "store")
    # Test with 'depth=2'.
    depth = 2
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ]
    )
    paths_ref = [
        (f"london.temperature{DIR_SEP}greenwich.summer", ["dataset.parquet"]),
        (f"london.temperature{DIR_SEP}westminster.winter", ["dummyfile.txt"]),
        (f"paris.temperature{DIR_SEP}bastille.summer", ["datasetfile1.parq", "datasetfile2.parq"]),
        (f"stockholm.pressure{DIR_SEP}skansen.fall", ["datasetfile.parquet"]),
    ]
    assert paths_files == paths_ref
    # Test with 'depth=2'.
    depth = 1
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = [
        (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), files) for path, files in paths_files
    ]
    assert paths_files == []
    # Test with 'depth=3'.
    depth = 3
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ]
    )
    paths_ref = [
        (f"paris.temperature{DIR_SEP}bastille.summer{DIR_SEP}forgottendir", ["forgottenfile.parq"]),
        (f"stockholm.pressure{DIR_SEP}flemings.spring{DIR_SEP}innerplace.morning", ["_metadata"]),
    ]
    assert paths_files == paths_ref


@pytest.mark.parametrize(
    "closed,label", [("left", "left"), ("right", "left"), ("right", "right"), ("left", "right")]
)
def test_tcut(closed, label):
    # Test data
    ts = [
        Timestamp("2022/03/01 09:00"),
        Timestamp("2022/03/01 10:00"),
        Timestamp("2022/03/01 10:30"),
        Timestamp("2022/03/01 12:00"),
        Timestamp("2022/03/01 15:00"),
        Timestamp("2022/03/01 19:00"),
    ]
    df = DataFrame({"a": range(len(ts)), "ts": ts})
    # Test closed=left/label=left.
    grouper = Grouper(
        key="ts", freq="2H", sort=False, origin="start_day", closed=closed, label=label
    )
    agg_ref = df.groupby(grouper)["a"].agg("first")
    ddf = df.copy()
    ddf["ts"] = tcut(df["ts"], grouper).astype("datetime64[ns]")
    agg_res = ddf.groupby(grouper)["a"].agg("first")
    assert agg_ref.equals(agg_res)


def test_merge_sorted_labels_only():
    # Test data
    ar1 = array([1, 5, 9, 15, 19, 20, 20], dtype="int64")
    ar2 = array([2, 4, 15, 15, 18, 20], dtype="int64")
    sorted_ars, sorted_idx_ar2 = merge_sorted((ar1, ar2))
    # Insertion indices:         1   2               6   7   8              12
    ref_sorted_ars = array([1, 2, 4, 5, 9, 15, 15, 15, 18, 19, 20, 20, 20])
    ref_sorted_idx_ar2 = array([1, 2, 6, 7, 8, 12])
    # Test
    assert nall(sorted_ars == ref_sorted_ars)
    assert nall(sorted_idx_ar2 == ref_sorted_idx_ar2)


def test_merge_sorted_labels_and_keys():
    # Test data
    ar1 = array([10, 5, 98, 12, 32, 2, 2], dtype="int64")
    ks1 = array([1, 5, 9, 15, 19, 20, 20], dtype="int64")
    ar2 = array([20, 4, 16, 17, 18, 20], dtype="int64")
    ks2 = array([2, 4, 15, 15, 18, 20], dtype="int64")
    sorted_ars, sorted_idx_ar2 = merge_sorted((ar1, ar2), (ks1, ks2))
    # Insertion indices:         1   2               6   7   8              12
    ref_sorted_ars = array([10, 20, 4, 5, 98, 12, 16, 17, 18, 32, 2, 2, 20])
    ref_sorted_idx_ar2 = array([1, 2, 6, 7, 8, 12])
    # Test
    assert nall(sorted_ars == ref_sorted_ars)
    assert nall(sorted_idx_ar2 == ref_sorted_idx_ar2)


def test_merge_sorted_exceptions_first():
    # Test data
    ar1 = array([10, 5], dtype="int64")
    ks1 = array([1], dtype="int64")
    ar2 = array([20], dtype="int64")
    ks2 = array([2], dtype="int64")
    with pytest.raises(
        ValueError, match="^not possible to have arrays of different length for first"
    ):
        merge_sorted((ar1, ar2), (ks1, ks2))


def test_merge_sorted_exceptions_second():
    # Test data
    ar1 = array([10], dtype="int64")
    ks1 = array([1], dtype="int64")
    ar2 = array([20, 5], dtype="int64")
    ks2 = array([2], dtype="int64")
    with pytest.raises(
        ValueError, match="^not possible to have arrays of different length for second"
    ):
        merge_sorted((ar1, ar2), (ks1, ks2))
