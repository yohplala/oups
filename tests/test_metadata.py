#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh
"""
from pandas import DataFrame as pDataFrame

from oups.collection import ParquetSet
from oups.indexer import toplevel
from oups.writer import OUPS_METADATA
from oups.writer import OUPS_METADATA_KEY


# from os import path as os_path
# tmp_path = os_path.expanduser('~/Documents/code/data/oups')


@toplevel
class ShortIndexer:
    comp: str


def test_custom_metadata(tmp_path):
    # Write metadata in new dataset, and update them.
    # No oups specific metadata.
    # Step 1: write.
    pdf = pDataFrame({"a": [1], "b": [2]})
    metadata = {"md1": "step1", "md2": "step1", "md3": "step1"}
    store = ParquetSet(tmp_path, ShortIndexer)
    sidx = ShortIndexer("sidx")
    store[sidx] = {"metadata": metadata}, pdf
    # Retrieve metadata.
    md_rec = store[sidx].metadata
    # Remove 'pandas' key and item.
    del md_rec["pandas"]
    assert md_rec == metadata
    # Step 2: update.
    metadata = {"md1": None, "md2": "step2", "md4": "step2"}
    store[sidx] = {"metadata": metadata}, pdf
    # Retrieve metadata.
    md_rec = store[sidx].metadata
    md_ref = {"md2": "step2", "md3": "step1", "md4": "step2"}
    del md_rec["pandas"]
    assert md_rec == md_ref


def test_oups_metadata_wo_custom_metadata(tmp_path):
    # Write specific oups metadata in new dataset, and update them.
    # No user-defined metadata.
    # Step 1: write.
    pdf = pDataFrame({"a": [1], "b": [2]})
    metadata = {"md1": "step1", "md2": "step1", "md3": "step1"}
    store = ParquetSet(tmp_path, ShortIndexer)
    sidx = ShortIndexer("sidx")
    OUPS_METADATA[sidx] = metadata
    store[sidx] = pdf
    # Retrieve oups metadata.
    md_rec = store[sidx]._oups_metadata
    assert md_rec == metadata
    # Check 'OUPS_METADATA' has been emptied after being 'written'.
    assert not OUPS_METADATA
    # Step 2: update.
    metadata = {"md1": None, "md2": "step2", "md4": "step2"}
    OUPS_METADATA[sidx] = metadata
    store[sidx] = pdf
    # Retrieve oups metadata.
    md_rec = store[sidx]._oups_metadata
    md_ref = {"md2": "step2", "md3": "step1", "md4": "step2"}
    assert md_rec == md_ref
    assert not OUPS_METADATA


def test_oups_metadata_with_custom_metadata(tmp_path):
    # Write specific oups metadata in new dataset, and update them.
    # with user-defined metadata.
    # Step 1: write.
    pdf = pDataFrame({"a": [1], "b": [2]})
    store = ParquetSet(tmp_path, ShortIndexer)
    sidx = ShortIndexer("sidx")
    metadata = {"md1": "step1", "md2": "step1", "md3": "step1"}
    OUPS_METADATA[sidx] = metadata
    store[sidx] = {"metadata": metadata.copy()}, pdf
    # Retrieve oups metadata.
    md_rec = store[sidx]._oups_metadata
    assert md_rec == metadata
    # Check 'OUPS_METADATA' has been emptied after being 'written'.
    assert not OUPS_METADATA
    # Retrieve user-defined metadata.
    md_rec = store[sidx].metadata
    del md_rec["pandas"]
    assert OUPS_METADATA_KEY in md_rec
    del md_rec[OUPS_METADATA_KEY]
    assert md_rec == metadata
    # Step 2: update.
    metadata = {"md1": None, "md2": "step2", "md4": "step2"}
    OUPS_METADATA[sidx] = metadata
    store[sidx] = {"metadata": metadata.copy()}, pdf
    # Retrieve oups metadata.
    md_rec = store[sidx]._oups_metadata
    md_ref = {"md2": "step2", "md3": "step1", "md4": "step2"}
    assert md_rec == md_ref
    assert not OUPS_METADATA
    md_rec = store[sidx].metadata
    del md_rec["pandas"]
    assert OUPS_METADATA_KEY in md_rec
    del md_rec[OUPS_METADATA_KEY]
    assert md_rec == md_ref
