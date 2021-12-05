#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from os import path as os_path
import zipfile

from oups.defines import DIR_SEP
from oups.utils import files_at_depth


TEST_DATA = 'test-data'

def test_files_at_depth(tmp_path):
    fn = os_path.join(TEST_DATA, 'dummy_store.zip')
    with zipfile.ZipFile(fn, 'r') as zip_ref:
        zip_ref.extractall(tmp_path)
    basepath = os_path.join(tmp_path, 'store')
    # Test with 'depth=2'.
    depth=2
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted([(DIR_SEP.join(path.rsplit(DIR_SEP,depth)[1:]),
                           sorted(files)) for path, files in paths_files])
    paths_ref = [('london.temperature/greenwich.summer', ['dataset.parquet']),
                 ('london.temperature/westminster.winter', ['dummyfile.txt']),
                 ('paris.temperature/bastille.summer', ['datasetfile1.parq', 'datasetfile2.parq']),
                 ('stockholm.pressure/skansen.fall', ['datasetfile.parquet'])]
    assert paths_files == paths_ref
    # Test with 'depth=2'.
    depth=1
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = [(DIR_SEP.join(path.rsplit(DIR_SEP,depth)[1:]), files)
                   for path, files in paths_files]
    assert paths_files == []
    # Test with 'depth=3'.
    depth=3
    paths_files = files_at_depth(basepath, depth)
    # Trim head.
    paths_files = sorted([(DIR_SEP.join(path.rsplit(DIR_SEP,depth)[1:]),
                           sorted(files)) for path, files in paths_files])
    paths_ref = [('paris.temperature/bastille.summer/forgottendir', ['forgottenfile.parq']),
                 ('stockholm.pressure/flemings.spring/innerplace.morning', ['_metadata'])]
    assert paths_files == paths_ref
