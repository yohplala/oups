#!/usr/bin/env python3
"""
Created on Tue Apr 29 22:00:00 2025.

Test cases for iter_data functions.

Tests cover chunk extraction, pandas DataFrame iteration, and parquet file
iteration with various configurations of distinct bounds and duplicate handling.

@author: yoh

"""
import pytest
from numpy import array
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from oups.store.write.iter_merge_split_data import iter_merge_split_data
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
    "test_id,pf_data,df_data,row_group_target_size,duplicates_on,merge_sequences,expected_chunks",
    [
        (  # DataFrame before OrderedParquetDataset, no overlap, wo duplicates_on.
            "df_before_pf_wo_duplicates_on",
            {"ordered": [3, 4, 5], "values": ["c", "d", "e"]},  # pf_data
            {"ordered": [1, 2, 2], "values": ["a", "b", "c"]},  # df_data
            2,  # row_group_target_size
            None,  # duplicates_on
            [(0, array([[0, 3]]))],  # merge_sequences - no pf row group
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [2], "values": ["c"]}),
            ],
        ),
        (  # DataFrame after OrderedParquetDataset, no overlap, wo duplicates_on.
            "df_after_pf_wo_duplicates_on",
            {"ordered": [1, 2, 3], "values": ["a", "b", "c"]},  # pf_data
            {"ordered": [4, 5, 6, 7], "values": ["d", "e", "f", "g"]},  # df_data
            2,  # row_group_target_size
            None,  # duplicates_on
            [(1, array([[2, 4]]))],  # merge_sequences - incl. last pf row group
            [
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5, 6], "values": ["e", "f"]}),
                DataFrame({"ordered": [7], "values": ["g"]}),
            ],
        ),
        (  # DataFrame spans over OrderedParquetDataset, wo duplicates_on.
            "df_spans_pf_wo_duplicates_on",
            {"ordered": [2, 3, 4], "values": ["x", "c", "y"]},  # pf_data
            {"ordered": [1, 2, 4, 5], "values": ["a", "b", "d", "e"]},  # df_data
            3,  # row_group_target_size
            None,  # duplicates_on
            [(0, array([[1, 3], [1, 4]]))],  # merge_sequences
            [
                DataFrame({"ordered": [1, 2, 2], "values": ["a", "x", "b"]}),
                DataFrame({"ordered": [3, 4, 4], "values": ["c", "y", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # OrderedParquetDataset spans over DataFrame, wo duplicates_on.
            "pf_spans_df_wo_duplicates_on",
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},  # pf_data
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},  # df_data
            2,  # row_group_target_size
            None,  # duplicates_on
            [(0, array([[1, 0], [2, 2], [3, 3]]))],  # merge_sequences
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "x"]}),
                DataFrame({"ordered": [2, 3], "values": ["b", "y"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "z"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
            ],
        ),
        (  # Multiple merge sequences, with duplicates_on.
            "multiple_merge_sequences_w_duplicates_on",
            {
                "ordered": [1, 2, 3, 4, 5, 7, 8, 9, 10],
                "values": ["a", "x", "y", "z", "e", "f", "g", "h", "i"],
            },  # pf_data
            {"ordered": [2, 6, 7, 11, 12], "values": ["b", "c", "d", "i", "j"]},  # df_data
            2,  # row_group_target_size
            "ordered",  # duplicates_on
            [(0, array([[1, 1]])), (2, array([[3, 3]])), (4, array([[5, 5]]))],
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [5, 6], "values": ["e", "c"]}),
                DataFrame({"ordered": [7], "values": ["d"]}),
                DataFrame({"ordered": [10, 11], "values": ["i", "i"]}),
                DataFrame({"ordered": [12], "values": ["j"]}),
            ],
        ),
        (  # OrderedParquetDataset spans over DataFrame, with duplicates_on.
            "pf_spans_df_w_duplicates_on",
            {"ordered": [1, 2, 3, 4, 5], "values": ["a", "x", "y", "z", "e"]},
            {"ordered": [2, 3, 4], "values": ["b", "c", "d"]},
            2,  # row_group_target_size
            "ordered",  # duplicates_on
            [(0, array([[1, 1], [2, 3], [3, 3]]))],  # merge_sequences
            [
                DataFrame({"ordered": [1, 2], "values": ["a", "b"]}),
                DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
                DataFrame({"ordered": [5], "values": ["e"]}),
            ],
        ),
        (  # DataFrame with long chunk between 2 row groups.
            "df_long_intermediate_chunk",
            {"ordered": [1, 2, 6], "values": ["a", "x", "y"]},
            {"ordered": [2, 3, 4, 5, 6, 7, 8], "values": ["b", "c", "d", "e", "f", "g", "h"]},
            2,  # row_group_target_size
            None,  # duplicates_on
            [(1, array([[1, 2], [1, 4]])), (2, array([[2, 6], [2, 7]]))],  # merge_sequences
            [
                DataFrame({"ordered": [2, 3], "values": ["b", "c"]}),
                DataFrame({"ordered": [4, 5], "values": ["d", "e"]}),
                DataFrame({"ordered": [6, 7], "values": ["f", "g"]}),
                DataFrame({"ordered": [8], "values": ["h"]}),
            ],
        ),
        (  # Empty DataFrame test case
            "empty_dataframe",
            {"ordered": [1, 2], "values": ["a", "b"]},  # pf_data
            {"ordered": [], "values": []},  # df_data
            2,  # row_group_target_size
            None,  # duplicates_on
            [(0, array([[1, 0]]))],  # merge_sequences
            [DataFrame({"ordered": [1, 2], "values": ["a", "b"]})],  # expected_chunks
        ),
        (  # Empty OrderedParquetDataset test case
            "empty_parquetfile",
            {"ordered": [], "values": []},  # pf_data
            {"ordered": [1, 2], "values": ["a", "b"]},  # df_data
            2,  # row_group_target_size
            None,  # duplicates_on
            [(0, array([[0, 2]]))],  # merge_sequences
            [DataFrame({"ordered": [1, 2], "values": ["a", "b"]})],  # expected_chunks
        ),
    ],
)
def test_iter_merge_data(
    test_id,
    pf_data,
    df_data,
    row_group_target_size,
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
        Data for OrderedParquetDataset.
    df_data : dict
        Data for input DataFrame.
    row_group_target_size : int
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
    pf_data = DataFrame(pf_data)
    ordered_parquet_dataset = create_parquet_file(
        tmp_path,
        df=pf_data,
        row_group_offsets=compute_split_sequence(pf_data.loc[:, "ordered"], row_group_target_size),
    )
    merge_iter = iter_merge_split_data(
        opd=ordered_parquet_dataset,
        ordered_on="ordered",
        df=df,
        merge_sequences=merge_sequences,
        split_sequence=lambda x: compute_split_sequence(x, row_group_target_size),
        duplicates_on=duplicates_on,
    )
    chunks = list(merge_iter)
    assert len(chunks) == len(expected_chunks)
    for chunk, expected in zip(chunks, expected_chunks):
        assert_frame_equal(chunk.reset_index(drop=True), expected.reset_index(drop=True))
