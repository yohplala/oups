#!/usr/bin/env python3
"""
Test cases for iter_dataframe private functions.

Tests cover chunk extraction, pandas DataFrame iteration, and parquet file iteration
with various configurations of distinct bounds and duplicate handling.

"""

import pytest
from fastparquet import ParquetFile
from fastparquet import write
from pandas import DataFrame
from pandas import concat
from pandas.testing import assert_frame_equal

from oups.store.iter_data import _get_next_chunk
from oups.store.iter_data import _iter_pandas_dataframe
from oups.store.iter_data import _iter_resized_parquet_file


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
        sample_df,
        start_idx,
        size,
        "ordered",
        distinct_bounds,
    )
    assert_frame_equal(chunk, sample_df.iloc[start_idx:expected_end_idx])
    assert end_idx == expected_end_idx


@pytest.mark.parametrize(
    "start_df, duplicates_on, yield_remainder",
    [
        (None, None, True),  # Basic case, no remainder
        (DataFrame({"ordered": [0], "values": ["z"]}), None, True),  # With start_df
        (DataFrame({"ordered": [-1, 0], "values": ["w", "z"]}), "ordered", False),  # With start_df
        (
            DataFrame({"ordered": [-1, 0, 1], "values": ["w", "z", "u"]}),
            "ordered",
            False,
        ),  # With start_df
        (None, "ordered", True),  # With duplicates
        (None, None, False),  # Return remainder
    ],
)
def test_iter_pandas_dataframe(
    sample_df,
    start_df,
    duplicates_on,
    yield_remainder,
):
    """
    Test _iter_pandas_dataframe with various configurations.

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
    iterator = _iter_pandas_dataframe(
        sample_df,
        row_group_size,
        "ordered",
        start_df=start_df.copy(deep=True) if start_df is not None else None,
        distinct_bounds=bool(duplicates_on),
        duplicates_on=duplicates_on,
        yield_remainder=yield_remainder,
    )

    expected = (
        concat([start_df, sample_df], ignore_index=True) if start_df is not None else sample_df
    )
    if duplicates_on:
        expected = expected.drop_duplicates(duplicates_on, keep="last", ignore_index=True)
    has_remainder = len(expected) % row_group_size > 0

    # Collect all chunks
    if yield_remainder:
        all_chunks = list(iterator)
        yielded_chunks = all_chunks
    else:

        def yield_all():
            remainder = yield from iterator
            if remainder is not None:
                yield remainder

        all_chunks = list(yield_all())
        # Do a 2nd time to check only yielded chunks.
        iterator2 = _iter_pandas_dataframe(
            sample_df,
            row_group_size,
            "ordered",
            start_df=start_df.copy(deep=True) if start_df is not None else None,
            distinct_bounds=bool(duplicates_on),
            duplicates_on=duplicates_on,
            yield_remainder=yield_remainder,
        )
        yielded_chunks = list(iterator2)
        print("yielded_chunks")
        print(yielded_chunks)

    print("all_chunks")
    print(all_chunks)
    complete_chunks = all_chunks[:-1] if has_remainder else all_chunks

    # Verify chunk sizes
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
        print("expected_without_remainder")
        print(expected_without_remainder)
        print("result")
        print(result)
        assert_frame_equal(result, expected_without_remainder)


@pytest.fixture
def create_parquet_file(tmp_path):
    """
    Create a temporary parquet file for testing.
    """

    def _create_parquet(df):
        path = f"{tmp_path}/test.parquet"
        write(path, df, row_group_offsets=[0, 3, 6], file_scheme="hive")
        return ParquetFile(path)

    return _create_parquet


@pytest.mark.parametrize(
    "start_df,yield_remainder,has_remainder",
    [
        (None, True, False),  # Basic case
        (DataFrame({"ordered": [0], "values": ["z"]}), True, False),  # With start_df
        (None, False, True),  # Return remainder
    ],
)
def test_iter_resized_parquet_file(
    sample_df,
    create_parquet_file,
    start_df,
    yield_remainder,
    has_remainder,
):
    """
    Test _iter_resized_parquet_file with various configurations.

    Parameters
    ----------
    sample_df : DataFrame
        Test DataFrame fixture.
    create_parquet_file : callable
        Fixture to create temporary parquet files.
    start_df : DataFrame or None
        Optional starter DataFrame.
    yield_remainder : bool
        Whether to yield the last chunk.
    has_remainder : bool
        Whether a remainder is expected.

    """
    pf = create_parquet_file(sample_df)
    row_group_size = 3

    iterator = _iter_resized_parquet_file(
        pf,
        row_group_size,
        "ordered",
        start_df=start_df,
        distinct_bounds=False,
        yield_remainder=yield_remainder,
    )

    # Collect all chunks
    chunks = []
    returned_remainder = None
    for chunk in iterator:
        chunks.append(chunk)

    # Get remainder if it was returned
    if not yield_remainder:
        returned_remainder = iterator.send(None)

    # Verify chunk sizes
    for chunk in chunks[:-1]:  # All but last chunk
        assert len(chunk) <= row_group_size

    # Verify total data
    result = concat(chunks, ignore_index=True)
    expected = (
        concat([start_df, sample_df], ignore_index=True) if start_df is not None else sample_df
    )

    if not yield_remainder:
        # When not yielding last chunk, expected data should exclude remainder
        expected_without_remainder = expected.iloc[
            : (len(expected) // row_group_size) * row_group_size
        ]
        n_expected_rows = len(expected_without_remainder)
        expected_remainder = expected.iloc[n_expected_rows:]
        assert returned_remainder is not None
        assert_frame_equal(returned_remainder, expected_remainder)
        assert_frame_equal(result, expected_without_remainder)
    else:
        assert returned_remainder is None
        assert_frame_equal(result, expected)
