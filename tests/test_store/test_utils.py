#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

Test utils.
- Initialize path:
tmp_path = os_path.expanduser('~/Documents/code/data/oups')

"""
import zipfile
from os import path as os_path

import pytest
from fastparquet import write
from numpy import array
from numpy import array_equal
from numpy import cumsum
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import MultiIndex

from oups.store.defines import DIR_SEP
from oups.store.split_strategies import get_region_start_end_delta
from oups.store.utils import conform_cmidx
from oups.store.utils import files_at_depth

from .. import TEST_DATA


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
        ],
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
        ],
    )
    paths_ref = [
        (f"paris.temperature{DIR_SEP}bastille.summer{DIR_SEP}forgottendir", ["forgottenfile.parq"]),
        (f"stockholm.pressure{DIR_SEP}flemings.spring{DIR_SEP}innerplace.morning", ["_metadata"]),
    ]
    assert paths_files == paths_ref


def test_conform_cmidx(tmp_path):
    # Check first that fastparquet is unable to write a DataFrame with
    # identified shortcomings, then confirm it works with 'conform_cmidx()'.
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(TypeError, match="^Column name must be a string"):
        write(tmp_path, df, file_scheme="hive")
    # Check then one column name is not a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(ValueError, match="^\\('Column names must be multi-index,"):
        write(tmp_path, df, file_scheme="hive")
    df = DataFrame({("a", 1, "o"): [1]})
    df.columns.set_names(["oh"], level=[1], inplace=True)
    # Conform cmidx.
    conform_cmidx(df)
    assert df.columns == MultiIndex.from_tuples([("a", "1", "o")], names=["", "oh", ""])


@pytest.mark.parametrize(
    "test_id, values, indices, expected",
    [
        (
            "single_region_at_first",
            array([1, 2, 3, 4]),  # values, cumsum = [1,3,6,10]
            array([[0, 3]]),  # one region: indices 0-2
            array([6]),  # 1+2+3 = 6-0 = 6
        ),
        (
            "single_region_at_second",
            array([1, 2, 3, 4, 5]),  # values, cumsum = [1,3,6,10,15]
            array([[1, 4]]),  # one region: indices 1-3
            array([9]),  # 2+3+4 = 10-1 = 9
        ),
        (
            "multiple_overlapping_regions",
            array([1, 2, 3, 4, 5, 6]),  # values, cumsum = [1,3,6,10,15,21]
            array(
                [
                    [0, 2],  # region 1: indices 0-1
                    [2, 5],  # region 2: indices 2-4
                    [4, 6],  # region 3: indices 4-5
                ],
            ),
            array([3, 12, 11]),  # 3-0=3, 15-3=12, 21-10=11
        ),
        (
            "boolean_values",
            array([0, 1, 0, 1, 1, 0]),  # values, cumsum = [0,1,1,2,3,3]
            array(
                [
                    [1, 4],  # one region: indices 1-3
                    [2, 5],  # one region: indices 2-4
                ],
            ),
            array([2, 2]),  # 2-1=1, 3-1=2
        ),
    ],
)
def test_get_region_start_end_delta(
    test_id: str,
    values: NDArray,
    indices: NDArray,
    expected: NDArray,
) -> None:
    """
    Test get_region_start_end_delta function with various inputs.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    values : NDArray
        Input array of values.
    indices : NDArray
        Array of shape (n, 2) containing start and end indices of regions.
    expected : NDArray
        Expected output containing sums for each region.

    """
    m_values = cumsum(values)
    result = get_region_start_end_delta(m_values, indices)
    assert array_equal(result, expected)
