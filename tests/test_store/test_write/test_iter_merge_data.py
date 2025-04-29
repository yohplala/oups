#!/usr/bin/env python3
"""
Test cases for iter_data functions.

Tests cover chunk extraction, pandas DataFrame iteration, and parquet file iteration
with various configurations of distinct bounds and duplicate handling.

"""

import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from oups.store.write.iter_merge_data import iter_merge_data
from tests.test_store.conftest import create_parquet_file


@pytest.fixture
def sample_df():
    """
    Create a sample DataFrame for testing.
    """
    return DataFrame(
        {
            "ordered": [1, 2, 2, 2, 3, 4, 4, 5],
            "values": list("abcdefgh"),
        },
    )


@pytest.mark.parametrize(
    "df_data,pf_data,expected_chunks,max_row_group_size,distinct_bounds,duplicates_on",
    [
        # Case 1: DataFrame before ParquetFile (no overlap)
        (
            {"ordered": [1, 2], "values": ["a", "b"]},
            {"ordered": [3, 4, 5], "values": ["c", "d", "e"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
            2,
            False,
            None,
        ),
        # Case 2: DataFrame after ParquetFile (no overlap)
        (
            {"ordered": [4, 5], "values": ["d", "e"]},
            {"ordered": [1, 2, 3], "values": ["a", "b", "c"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
            2,
            False,
            None,
        ),
        # Case 3: DataFrame spans over ParquetFile
        (
            {"ordered": [1, 2, 4, 5], "values": ["a", "b", "d", "e"]},
            {"ordered": [2, 3, 4], "values": ["x", "c", "y"]},
            [
                DataFrame({"ordered": [1, 2, 2], "values": ["a", "x", "b"]}),
                DataFrame({"ordered": [3, 4, 4], "values": ["c", "y", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
            3,
            False,
            None,
        ),
        # Case 4: ParquetFile spans over DataFrame, wo distinct bounds
        (
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "y"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "z"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
            ],
            2,
            False,
            None,
        ),
        # Case 5: ParquetFile spans over DataFrame, with distinct bounds
        (
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            [
                DataFrame({"ordered": [1], "values": ["a"]}),
                DataFrame({"ordered": [2, 2], "values": ["x", "b"]}),
                DataFrame({"ordered": [3, 3], "values": ["y", "c"]}),
                DataFrame({"ordered": [4, 4], "values": ["z", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
            2,
            True,
            None,
        ),
        # Case 6: Remainder from dataframe, wo distinct bounds.
        (
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            {"ordered": [1, 2, 3, 4], "values": ["a", "x", "y", "z"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "y"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "z"]}),
                DataFrame({"ordered": [4], "values": ["d"]}),
            ],
            2,
            False,
            None,
        ),
        # Case 7: ParquetFile spans over DataFrame, drop duplicates.
        (
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
            2,
            True,
            "ordered",
        ),
        # Case 8: DataFrame with long chunk between 2 row groups.
        (
            {"ordered": [2, 3, 4, 5, 6, 7, 8], "values": ["b", "c", "d", "e", "f", "g", "h"]},
            {"ordered": [1, 2, 6], "values": ["a", "x", "y"]},
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "c"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
                DataFrame({"ordered": [6, 6], "values": ["y", "f"]}),
                DataFrame({"ordered": [7, 8], "values": ["g", "h"]}),
            ],
            2,
            False,
            None,
        ),
    ],
)
def test_iter_merge_data(
    df_data,
    pf_data,
    expected_chunks,
    max_row_group_size,
    distinct_bounds,
    duplicates_on,
    create_parquet_file=create_parquet_file,
):
    """
    Test iter_merge_data with various overlap scenarios.

    Parameters
    ----------
    df_data : dict
        Data for input DataFrame.
    pf_data : dict
        Data for ParquetFile.
    expected_chunks : list
        List of expected DataFrame chunks.
    max_row_group_size : int
        Maximum size for each chunk.
    create_parquet_file : callable
        Fixture to create temporary parquet files.

    """
    df = DataFrame(df_data)
    pf = create_parquet_file(DataFrame(pf_data), row_group_offsets=max_row_group_size)

    chunks = list(
        iter_merge_data(
            ordered_on="ordered",
            df=df,
            pf=pf,
            max_row_group_size=max_row_group_size,
            distinct_bounds=distinct_bounds,
            duplicates_on=duplicates_on,
        ),
    )

    assert len(chunks) == len(expected_chunks)
    for chunk, expected in zip(chunks, expected_chunks):
        assert_frame_equal(chunk.reset_index(drop=True), expected.reset_index(drop=True))


def test_iter_merge_data_empty_df(create_parquet_file=create_parquet_file):
    """
    Test handling of empty DataFrame input.
    """
    df = DataFrame({"ordered": [], "values": []})
    pf = create_parquet_file(
        DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
        row_group_offsets=2,
    )

    chunks = list(
        iter_merge_data(
            df,
            pf,
            ordered_on="ordered",
            max_row_group_size=2,
        ),
    )
    assert len(chunks) == 0


def test_iter_merge_data_empty_opd(create_parquet_file=create_parquet_file):
    """
    Test handling of empty DataFrame input.
    """
    df = DataFrame({"ordered": [], "values": []})
    pf = create_parquet_file(
        DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
        row_group_offsets=2,
    )

    chunks = list(
        iter_merge_data(
            df,
            pf,
            ordered_on="ordered",
            max_row_group_size=2,
        ),
    )
    assert len(chunks) == 0
