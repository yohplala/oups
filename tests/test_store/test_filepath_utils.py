#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
import zipfile
from os import path as os_path

from oups.defines import DIR_SEP
from oups.store.filepath_utils import files_at_depth

from .. import TEST_DATA


def test_files_at_depth(tmp_path):
    fn = os_path.join(TEST_DATA, "dummy_store.zip")
    with zipfile.ZipFile(fn, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    basepath = tmp_path / "store"
    # Test with 'depth=2'.
    depth = 2
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(str(path).rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    paths_ref2 = [
        (f"stockholm.pressure{DIR_SEP}flemings.spring", ["innerplace.morning_opdmd"]),
    ]
    assert paths_files == paths_ref2
    # Test with 'depth=2'.
    depth = 1
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(str(path).rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    paths_ref1 = [
        ("london.temperature", ["greenwich.summer_opdmd", "westminster.winter_dummy"]),
        ("paris.temperature", ["bastille.summer_opdmd"]),
        ("stockholm.pressure", ["skansen.fall_opdmd"]),
    ]
    assert paths_files == paths_ref1
    # Test with 'depth=3'.
    depth = 3
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted(
        [
            (DIR_SEP.join(path.rsplit(DIR_SEP, depth)[1:]), sorted(files))
            for path, files in paths_files
        ],
    )
    assert paths_files == []
