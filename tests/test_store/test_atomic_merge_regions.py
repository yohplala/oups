#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""

from typing import List

import pytest
from numpy import array
from numpy import bool_
from numpy import int_
from numpy import ones
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import Series

from oups.store.atomic_merge_regions import HAS_DF_CHUNK
from oups.store.atomic_merge_regions import HAS_ROW_GROUP
from oups.store.atomic_merge_regions import NRowsSplitStrategy
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
    assert_array_equal(amrs_prop["rg_idx_start"], expected["rg_idx_start"])
    assert_array_equal(amrs_prop["rg_idx_end_excl"], expected["rg_idx_end_excl"])
    assert_array_equal(amrs_prop["df_idx_end_excl"], expected["df_idx_end_excl"])
    assert_array_equal(amrs_prop[HAS_ROW_GROUP], expected[HAS_ROW_GROUP])
    assert_array_equal(amrs_prop[HAS_DF_CHUNK], expected[HAS_DF_CHUNK])


def test_nrows_split_strategy_likely_meets_target_size():
    """
    Test NRowsSplitStrategy.likely_meets_target_size property.

    Tests various scenarios:
    1. AMR with only row group
    2. AMR with only DataFrame chunk
    3. AMR with both row group and DataFrame chunk
    4. AMR that's too small
    5. AMR that's too large

    """
    target_size = 100  # min size: 80
    # Create mock amrs_info with 5 regions:
    # 1. RG only (90 rows) - meets target
    # 2. DF only (85 rows) - meets target
    # 3. Both RG (50) and DF (40) - meets target due to potential duplicates
    # 4. Both RG (30) and DF (30) - too small
    # 5. Both RG (120) and DF (0) - too large
    # 6. Both RG (30) and DF (80) - too large
    amrs_info = ones(
        6,
        dtype=[
            ("rg_idx_start", int_),
            ("rg_idx_end_excl", int_),
            ("df_idx_end_excl", int_),
            (HAS_ROW_GROUP, bool_),
            (HAS_DF_CHUNK, bool_),
        ],
    )
    # Set which regions have row groups and DataFrame chunks
    amrs_info[HAS_ROW_GROUP] = array([True, False, True, True, True, True])
    amrs_info[HAS_DF_CHUNK] = array([False, True, True, True, False, True])
    # Set up DataFrame chunk sizes through df_idx_end_excl
    amrs_info["df_idx_end_excl"] = array([0, 85, 125, 155, 155, 235])
    # Create row group sizes array
    rgs_n_rows = array([90, 50, 30, 120, 30])
    # Initialize strategy
    strategy = NRowsSplitStrategy(
        amrs_info=amrs_info,
        rgs_n_rows=rgs_n_rows,
        df_n_rows=sum(amrs_info["df_idx_end_excl"]),
        row_group_target_size=target_size,
        max_n_irgs=1,
    )
    # Get result
    result = strategy.likely_meets_target_size
    # Expected results:
    # 1. True  - RG only with 90 rows (between min_size and target_size)
    # 2. True  - DF only with 85 rows (between min_size and target_size)
    # 3. True  - Combined RG(50) + DF(40) could meet target after deduplication
    # 4. False - Combined RG(30) + DF(30) too small even without deduplication
    # 5. False - RG only with 120 rows (above target_size)
    # 6. False - Combined RG(30) + DF(80) could be too large
    expected = array([True, True, True, False, False, False])
    assert_array_equal(result, expected)
