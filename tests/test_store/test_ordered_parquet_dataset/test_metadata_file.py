#!/usr/bin/env python3
"""
Created on Sat May 24 21:00:00 2025.

@author: yoh

"""
import pytest

from oups.defines import DIR_SEP
from oups.store.ordered_parquet_dataset.metadata_filename import OPDMD_EXTENSION
from oups.store.ordered_parquet_dataset.metadata_filename import get_md_basename
from oups.store.ordered_parquet_dataset.metadata_filename import get_md_filepath


def test_get_md_filepath(tmp_path):
    """
    Test the get_md_filepath function.
    """
    assert (
        get_md_filepath(f"{tmp_path}{DIR_SEP}test") == f"{tmp_path}{DIR_SEP}test{OPDMD_EXTENSION}"
    )


@pytest.mark.parametrize(
    "filepath, expected",
    [
        (f"test_path{DIR_SEP}test", None),
        (f"test{OPDMD_EXTENSION}", "test"),
        (f"test_path{DIR_SEP}test{OPDMD_EXTENSION}", "test"),
    ],
)
def test_get_md_basename(filepath, expected):
    """
    Test the get_md_basename function.
    """
    assert get_md_basename(filepath) == expected
