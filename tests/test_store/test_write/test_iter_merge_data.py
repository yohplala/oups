#!/usr/bin/env python3
"""
Test cases for iter_data functions.

Tests cover chunk extraction, pandas DataFrame iteration, and parquet file iteration
with various configurations of distinct bounds and duplicate handling.

"""

from typing import Iterable

import pytest
from pandas import DataFrame
from pandas import concat
from pandas.testing import assert_frame_equal

from oups.store.write.iter_merge_data import _get_next_chunk
from oups.store.write.iter_merge_data import _iter_df
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
    "start_idx,size,distinct_bounds,expected_end_idx",
    [
        # Case 1: No distinct bounds - simply takes size=3 rows
        (0, 3, False, 3),
        # Case 2: Distinct bounds with fewer duplicates than size
        # Should only take value=1 to avoid splitting value=2
        (0, 3, True, 1),
        # Case 3: Distinct bounds with more duplicates than size
        # Starting at first 2, should take all three 2s
        (1, 2, True, 4),
        # Case 4: Starting mid-sequence with distinct bounds
        # Starting at first 4, should take both 4s
        (5, 2, True, 7),
        # Case 5: End of dataframe
        (6, 3, False, 8),
        # Case 6: Distinct bounds at end of dataframe
        (7, 2, True, 8),
    ],
)
def test_get_next_chunk(
    sample_df,
    start_idx,
    size,
    distinct_bounds,
    expected_end_idx,
):
    """
    Test _get_next_chunk function with different boundary conditions.

    Parameters
    ----------
    sample_df : DataFrame
        Test DataFrame fixture.
    start_idx : int
        Starting index for chunk extraction.
    size : int
        Maximum chunk size.
    distinct_bounds : bool
        Whether to respect value boundaries.
    expected_end_idx : int
        Expected ending index.

    """
    chunk, end_idx = _get_next_chunk(
        df=sample_df,
        start_idx=start_idx,
        size=size,
        distinct_bounds=distinct_bounds,
        ordered_on="ordered",
    )
    assert_frame_equal(chunk, sample_df.iloc[start_idx:expected_end_idx])
    assert end_idx == expected_end_idx


def yield_all(iterator: Iterable[DataFrame]) -> Iterable[DataFrame]:
    """
    Yield all chunks from an iterator, including any remainder.
    """
    remainder = yield from iterator
    if remainder is not None:
        yield remainder


@pytest.mark.parametrize(
    "start_df, duplicates_on, yield_remainder",
    [
        (None, None, True),  # Basic case, no remainder
        (DataFrame({"ordered": [0], "values": ["z"]}), None, True),  # With start_df
        (
            DataFrame({"ordered": [-1, 0], "values": ["w", "z"]}),
            "ordered",
            False,
        ),  # With start_df
        (
            DataFrame({"ordered": [-1, 0, 1], "values": ["w", "z", "u"]}),
            "ordered",
            False,
        ),  # With start_df
        (None, "ordered", True),  # With duplicates
        (None, False, True),  # Return remainder
    ],
)
def test_iter_df(
    sample_df,
    start_df,
    duplicates_on,
    yield_remainder,
):
    """
    Test _iter_df with various configurations.

    Parameters
    ----------
    sample_df : DataFrame
        Test DataFrame fixture.
    start_df : DataFrame or None
        Optional starter DataFrame.
    duplicates_on : str or None
        Column to check for duplicates.
    yield_last : bool
        Whether to yield the last chunk.

    """
    row_group_size = 3
    iterator = _iter_df(
        ordered_on="ordered",
        max_row_group_size=row_group_size,
        df=[start_df, sample_df] if start_df is not None else sample_df,
        distinct_bounds=bool(duplicates_on),
        duplicates_on=duplicates_on,
        yield_remainder=yield_remainder,
    )

    if start_df is not None:
        expected = concat([start_df, sample_df], ignore_index=True)
    else:
        expected = sample_df

    if duplicates_on:
        expected = expected.drop_duplicates(duplicates_on, keep="last", ignore_index=True)
    has_remainder = len(expected) % row_group_size > 0

    # Collect all chunks
    if yield_remainder:
        all_chunks = list(iterator)
        yielded_chunks = all_chunks
    else:
        all_chunks = list(yield_all(iterator))
        # Do a 2nd time to check only yielded chunks.
        iterator2 = _iter_df(
            ordered_on="ordered",
            max_row_group_size=row_group_size,
            df=[start_df, sample_df] if start_df is not None else sample_df,
            distinct_bounds=bool(duplicates_on),
            duplicates_on=duplicates_on,
            yield_remainder=yield_remainder,
        )
        yielded_chunks = list(iterator2)

    # Verify chunk sizes
    complete_chunks = all_chunks[:-1] if has_remainder else all_chunks
    for chunk in complete_chunks:
        assert len(chunk) == row_group_size

    # Verify yielded data
    result = concat(yielded_chunks, ignore_index=True)

    if yield_remainder:
        assert_frame_equal(result, expected)
    else:
        # When not yielding last chunk, expected data should exclude remainder
        expected_without_remainder = expected.iloc[
            : (len(expected) // row_group_size) * row_group_size
        ]
        assert_frame_equal(result, expected_without_remainder)


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


@pytest.mark.parametrize(
    "duplicates_on,distinct_bounds,error_expected",
    [
        (None, False, False),  # No duplicates check
        ("ordered", True, False),  # Valid duplicates check
        ("ordered", False, True),  # Invalid: duplicates without distinct bounds
        ("invalid_col", True, True),  # Invalid column name
    ],
)
def test_iter_merge_data_exceptions(
    duplicates_on,
    distinct_bounds,
    error_expected,
    create_parquet_file=create_parquet_file,
):
    """
    Test validation in iter_merge_data.

    Parameters
    ----------
    duplicates_on : str or None
        Column to check for duplicates.
    distinct_bounds : bool
        Whether to use distinct bounds.
    error_expected : bool
        Whether an error is expected.
    create_parquet_file : callable
        Fixture to create temporary parquet files.

    """
    df = DataFrame({"ordered": [1, 2], "values": ["a", "b"]})
    pf = create_parquet_file(
        DataFrame({"ordered": [3, 4], "values": ["c", "d"]}),
        row_group_offsets=2,
    )

    if error_expected:
        with pytest.raises(ValueError):
            list(
                iter_merge_data(
                    df,
                    pf,
                    ordered_on="ordered",
                    max_row_group_size=2,
                    distinct_bounds=distinct_bounds,
                    duplicates_on=duplicates_on,
                ),
            )
    else:
        chunks = list(
            iter_merge_data(
                df,
                pf,
                ordered_on="ordered",
                max_row_group_size=2,
                distinct_bounds=distinct_bounds,
                duplicates_on=duplicates_on,
            ),
        )
        assert len(chunks) > 0


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
