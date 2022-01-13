#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh
"""
import zipfile
from os import path as os_path

from oups.defines import DIR_SEP
from oups.utils import files_at_depth

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
