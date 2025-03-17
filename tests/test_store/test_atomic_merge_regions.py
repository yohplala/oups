#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""

from typing import List

import pytest
from numpy import array
from numpy import array_equal
from numpy.typing import NDArray
from pandas import Series

from oups.store.atomic_merge_regions import HAS_DF_CHUNK
from oups.store.atomic_merge_regions import HAS_ROW_GROUP
from oups.store.atomic_merge_regions import compute_atomic_merge_regions


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, drop_duplicates, expected",
    [
        (
            "no_gaps_rg_df",
            [10, 20, 30],  # rg_mins
            [15, 25, 35],  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 2]),
                "rg_idx_end_excl": array([1, 2, 3]),
                "df_idx_end_excl": array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_CHUNK: array([True, True, True]),
            },
        ),
        (
            "gap_at_start_df_leading_rg",
            [20, 30],  # rg_mins
            [25, 35],  # rg_maxs
            Series([5, 22, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 0, 1]),
                "rg_idx_end_excl": array([0, 1, 2]),
                "df_idx_end_excl": array([1, 2, 3]),
                HAS_ROW_GROUP: array([False, True, True]),
                HAS_DF_CHUNK: array([True, True, True]),
            },
        ),
        (
            "gap_in_middle_df_not_overlapping_rg",
            [10, 30],  # rg_mins
            [15, 35],  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 1]),
                "rg_idx_end_excl": array([1, 1, 2]),
                "df_idx_end_excl": array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, False, True]),
                HAS_DF_CHUNK: array([True, True, True]),
            },
        ),
        (
            "gap_at_end_df_trailing_rg",
            [10, 20],  # rg_mins
            [15, 25],  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 2]),
                "rg_idx_end_excl": array([1, 2, 2]),
                "df_idx_end_excl": array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, True, False]),
                HAS_DF_CHUNK: array([True, True, True]),
            },
        ),
        (
            "gap_at_start_rg_leading_df",
            [0, 20, 30],  # rg_mins
            [5, 23, 33],  # rg_maxs
            Series([22, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 2]),
                "rg_idx_end_excl": array([1, 2, 3]),
                "df_idx_end_excl": array([0, 1, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_CHUNK: array([False, True, True]),
            },
        ),
        (
            "gap_in_middle_rg_not_overlapping_df",
            [0, 10, 30],  # rg_mins
            [5, 15, 35],  # rg_maxs
            Series([2, 32]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 2]),
                "rg_idx_end_excl": array([1, 2, 3]),
                "df_idx_end_excl": array([1, 1, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_CHUNK: array([True, False, True]),
            },
        ),
        (
            "gap_at_end_rg_trailing_df",
            [10, 20, 30],  # rg_mins
            [15, 25, 35],  # rg_maxs
            Series([12, 22]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 1, 2]),
                "rg_idx_end_excl": array([1, 2, 3]),
                "df_idx_end_excl": array([1, 2, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_CHUNK: array([True, True, False]),
            },
        ),
        (
            "multiple_gaps_df_not_overlapping_rg",
            [20, 40, 43],  # rg_mins
            [25, 42, 45],  # rg_maxs
            Series([5, 22, 32, 41, 42, 46, 52]),  # df_ordered_on
            True,
            {
                "rg_idx_start": array([0, 0, 1, 1, 2, 3]),
                "rg_idx_end_excl": array([0, 1, 1, 2, 3, 3]),
                "df_idx_end_excl": array([1, 2, 3, 5, 5, 7]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_CHUNK: array([True, True, True, True, False, True]),
            },
        ),
        (
            "no_drop_duplicates_with_gap_with_overlapping_rg",
            [20, 40, 43],  # rg_mins, 43 overlaps with previous rg max
            [25, 43, 45],  # rg_maxs
            Series([5, 22, 32, 43, 46, 52]),  # df_ordered_on - 43 is duplicate
            False,  # don't drop duplicates - 43 expected to fall in last rg
            {
                "rg_idx_start": array([0, 0, 1, 1, 2, 3]),
                "rg_idx_end_excl": array([0, 1, 1, 2, 3, 3]),
                "df_idx_end_excl": array([1, 2, 3, 3, 4, 6]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_CHUNK: array([True, True, True, False, True, True]),
            },
        ),
        (
            "no_drop_duplicates_with_gap_wo_overlapping_rg",
            [10, 20],  # rg_mins
            [15, 25],  # rg_maxs
            Series([15, 16, 17, 22, 32]),  # df_ordered_on - note 15 is duplicate
            False,
            {
                "rg_idx_start": array([0, 1, 1, 2]),
                "rg_idx_end_excl": array([1, 1, 2, 2]),
                "df_idx_end_excl": array([0, 3, 4, 5]),
                HAS_ROW_GROUP: array([True, False, True, False]),
                HAS_DF_CHUNK: array([False, True, True, True]),
            },
        ),
    ],
)
def test_compute_atomic_merge_regions(
    test_id: str,
    rg_mins: List,
    rg_maxs: List,
    df_ordered_on: Series,
    drop_duplicates: bool,
    expected: NDArray,
) -> None:
    """
    Test _get_atomic_merge_regions with various scenarios.
    """
    amrs_prop = compute_atomic_merge_regions(
        rg_mins,
        rg_maxs,
        df_ordered_on,
        drop_duplicates,
    )
    # Check structured array fields
    assert array_equal(amrs_prop["rg_idx_start"], expected["rg_idx_start"])
    assert array_equal(amrs_prop["rg_idx_end_excl"], expected["rg_idx_end_excl"])
    assert array_equal(amrs_prop["df_idx_end_excl"], expected["df_idx_end_excl"])
    assert array_equal(amrs_prop[HAS_ROW_GROUP], expected[HAS_ROW_GROUP])
    assert array_equal(amrs_prop[HAS_DF_CHUNK], expected[HAS_DF_CHUNK])
