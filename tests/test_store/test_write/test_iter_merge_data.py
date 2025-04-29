#!/usr/bin/env python3
"""
Test cases for iter_data functions.

Tests cover chunk extraction, pandas DataFrame iteration, and parquet file iteration
with various configurations of distinct bounds and duplicate handling.

"""
import pytest
from numpy import array
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


def compute_split_sequence(series, max_size=2):
    """
    Helper function to compute split sequence for testing.
    """
    return list(range(0, len(series), max_size))


@pytest.mark.parametrize(
    "test_id,pf_data,df_data,max_row_group_size,duplicates_on,merge_sequences,expected_chunks",
    [
        (  # Case 1: DataFrame before ParquetFile (no overlap),
            # only 1 DataFrame row group.
            "df_before_pf_single_row_group",
            {"ordered": [3, 4, 5], "values": ["c", "d", "e"]},
            {"ordered": [1, 2], "values": ["a", "b"]},
            2,
            None,
            [(0, array([[0, 2]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
            ],
        ),
        (  # Case 2: DataFrame after ParquetFile (no overlap)
            "df_after_pf",
            {"ordered": [1, 2, 3], "values": ["a", "b", "c"]},
            {"ordered": [4, 5], "values": ["d", "e"]},
            2,
            None,
            [(0, array([[2, 3]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # Case 3: DataFrame spans over ParquetFile
            "df_spans_pf",
            {"ordered": [2, 3, 4], "values": ["x", "c", "y"]},
            {"ordered": [1, 2, 4, 5], "values": ["a", "b", "d", "e"]},
            3,
            None,
            [(0, array([[2, 3]]))],
            [
                DataFrame({"ordered": [1, 2, 2], "values": ["a", "x", "b"]}),
                DataFrame({"ordered": [3, 4, 4], "values": ["c", "y", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # Case 4: ParquetFile spans over DataFrame
            "pf_spans_df",
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            2,
            None,
            [(0, array([[2, 3], [4, 5]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "y"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "z"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
            ],
        ),
        (  # Case 5: Multiple merge sequences
            "multiple_merge_sequences",
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            2,
            None,
            [(0, array([[1, 2]])), (1, array([[3, 4]])), (2, array([[5, 6]]))],
            [
                DataFrame({"ordered": [1], "values": ["a"]}),
                DataFrame({"ordered": [2, 2], "values": ["x", "b"]}),
                DataFrame({"ordered": [3, 3], "values": ["y", "c"]}),
                DataFrame({"ordered": [4, 4], "values": ["z", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # Case 6: Remainder from dataframe, wo distinct bounds.
            "remainder_from_df",
            {"ordered": [1, 2, 3, 4], "values": ["a", "x", "y", "z"]},
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            2,
            None,
            [(0, array([[2, 3], [4, 5]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "y"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "z"]}),
                DataFrame({"ordered": [4], "values": ["d"]}),
            ],
        ),
        (  # Case 7: ParquetFile spans over DataFrame, drop duplicates
            "pf_spans_df_drop_duplicates",
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            2,
            "ordered",
            [(0, array([[2, 3], [4, 5]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # Case 8: DataFrame with long chunk between 2 row groups
            "df_long_chunk",
            {"ordered": [1, 2, 6], "values": ["a", "x", "y"]},
            {"ordered": [2, 3, 4, 5, 6, 7, 8], "values": ["b", "c", "d", "e", "f", "g", "h"]},
            2,
            None,
            [(0, array([[2, 3]])), (1, array([[4, 5]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "c"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
                DataFrame({"ordered": [6, 6], "values": ["y", "f"]}),
                DataFrame({"ordered": [7, 8], "values": ["g", "h"]}),
            ],
        ),
    ],
)
def test_iter_merge_data(
    test_id,
    pf_data,
    df_data,
    max_row_group_size,
    duplicates_on,
    merge_sequences,
    expected_chunks,
    tmp_path,
    create_parquet_file=create_parquet_file,
):
    """
    Test iter_merge_data with various overlap scenarios.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    pf_data : dict
        Data for ParquetFile.
    df_data : dict
        Data for input DataFrame.
    max_row_group_size : int
        Maximum size for each chunk.
    duplicates_on : str or None
        Column to check for duplicates.
    merge_sequences : list of tuples
        List of tuples containing merge sequences, where each tuple contains:
        - First element: Start index of the first row group in the merge sequence
        - Second element: Array of end indices (excluded) for row groups and DataFrame chunks
    expected_chunks : list
        List of expected DataFrame chunks.
    tmp_path : Path
        Temporary directory path provided by pytest.
    create_parquet_file : callable
        Fixture to create temporary parquet files.

    """
    df = DataFrame(df_data)
    opd = create_parquet_file(
        tmp_path=tmp_path,
        df=DataFrame(pf_data),
        row_group_offsets=max_row_group_size,
    )

    chunks = list(
        iter_merge_data(
            opd=opd,
            df=df,
            ordered_on="ordered",
            merge_sequences=merge_sequences,
            split_sequence=lambda x: compute_split_sequence(x, max_row_group_size),
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
    opd = create_parquet_file(
        DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
        row_group_offsets=2,
    )

    chunks = list(
        iter_merge_data(
            opd=opd,
            df=df,
            ordered_on="ordered",
            merge_sequences=[(0, array([2]))],
            split_sequence=lambda x: compute_split_sequence(x, 2),
        ),
    )
    assert len(chunks) == 0


def test_iter_merge_data_empty_opd(create_parquet_file=create_parquet_file):
    """
    Test handling of empty ParquetFile input.
    """
    df = DataFrame({"ordered": [1, 2], "values": ["a", "b"]})
    opd = create_parquet_file(
        DataFrame({"ordered": [], "values": []}),
        row_group_offsets=2,
    )

    chunks = list(
        iter_merge_data(
            opd=opd,
            df=df,
            ordered_on="ordered",
            merge_sequences=[(0, array([]))],
            split_sequence=lambda x: compute_split_sequence(x, 2),
        ),
    )
    assert len(chunks) == 0
