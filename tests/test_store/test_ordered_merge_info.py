#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import List

import pytest
from numpy import array
from numpy import array_equal
from numpy import empty
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.ordered_merge_info import compute_ordered_merge_plan
from oups.store.ordered_merge_info import get_region_indices_of_same_values
from oups.store.ordered_merge_info import get_region_indices_of_true_values
from oups.store.split_strategies import NRowsSplitStrategy
from oups.store.split_strategies import TimePeriodSplitStrategy
from tests.test_store.conftest import create_parquet_file


REF_D = "2020/01/01 "


@pytest.mark.parametrize(
    "mask, expected",
    [
        (  # Case 1: Entire mask is single region.
            array([True, True, True]),
            array([[0, 3]]),
        ),
        (  # Case 2: A mask with two regions.
            array([False, True, True, False, True]),
            array([[1, 3], [4, 5]]),
        ),
        (  # Case 3: no region.
            array([False, False, False]),
            empty((0, 2), dtype=int),
        ),
        (  # Case 4: One region at start, one region at end.
            array([True, False, True]),
            array([[0, 1], [2, 3]]),
        ),
    ],
)
def test_get_region_indices_of_true_values(mask: NDArray, expected: NDArray) -> None:
    """
    Test get_region_indices_of_true_values to verify that it correctly returns the start
    and end indices (as [start, end) pairs) for each region of contiguous True values in
    a boolean array.
    """
    result = get_region_indices_of_true_values(mask)
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "ints, expected",
    [
        (  # Case 1: 2 regions of same values.
            array([1, 1, 2]),
            array([[0, 2], [2, 3]]),
        ),
        (  # Case 2: 1 region of same values.
            array([1, 1, 1]),
            array([[0, 3]]),
        ),
        (  # Case 3: 3 region of one value.
            array([1, 2, 3]),
            array([[0, 1], [1, 2], [2, 3]]),
        ),
    ],
)
def test_get_region_indices_of_same_values(ints: NDArray, expected: NDArray) -> None:
    """
    Test get_region_indices_of_same_values to verify that it correctly returns the start
    and end indices (as [start, end) pairs) for each region made of same values in an
    array of int.
    """
    result = get_region_indices_of_same_values(ints)
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "test_id, rg_n_rows, df_n_rows, df_idx_tmrg_starts, df_idx_tmrg_ends_excl, "
    "row_group_size_target, irgs_allowed, indices_enlarged, indices_overlap, "
    "expected",
    [
        (
            "no_fragmentation_risk",
            array([50, 50, 50]),  # rg_n_rows
            100,  # df_n_rows
            array([0, 10, 20]),  # df_idx_tmrg_starts
            array([5, 15, 25]),  # df_idx_tmrg_ends_excl
            200,  # min_size = 0.8*row_group_size_target = 160
            True,  # irgs_allowed
            array([[0, 3]]),  # one enlarged region covering all row groups
            array(
                [
                    [0, 2],  # first overlap region: rgs 0-1
                    [1, 3],
                ],
            ),  # second overlap region: rgs 1-2
            array([False]),  # total rows 1 = 100 (rg) + 10 (df) < min_size
            # total rows 2 = 50 (rg) + 5 (df) < min_size
        ),
        (
            "single_fragmentation_risk",
            array([100, 100, 50]),  # rg_n_rows
            100,  # df_n_rows
            array([0, 10, 20]),  # df_idx_tmrg_starts
            array([5, 15, 25]),  # df_idx_tmrg_ends_excl
            150,  # min_size = 0.8*row_group_size_target = 120
            True,  # irgs_allowed
            array([[0, 3]]),  # one enlarged region covering all row groups
            array([[0, 3]]),  # one overlap region: rgs 0-1
            array([True]),  # total rows = 250 (rg) + 10 (df) > min_size
        ),
        (
            "multiple_regions_mixed_risks",
            array([100, 50, 100, 50]),  # rg_n_rows
            100,  # df_n_rows
            array([0, 10, 20, 30]),  # df_idx_tmrg_starts
            array([5, 15, 25, 35]),  # df_idx_tmrg_ends_excl
            150,  # min_size
            True,  # irgs_allowed
            array(
                [
                    [0, 2],  # region 1: indices 0-1
                    [2, 4],  # region 2: indices 2-3
                ],
            ),
            array(
                [
                    [0, 2],  # overlap 1: indices 0-1
                    [2, 4],  # overlap 2: indices 2-3
                ],
            ),
            array([True, True]),  # both regions exceed min_size
        ),
        (
            "overlapping_regions",
            array([50, 100, 100, 50]),  # rg_n_rows
            100,  # df_n_rows
            array([0, 10, 20, 30]),  # df_idx_tmrg_starts
            array([5, 15, 25, 35]),  # df_idx_tmrg_ends_excl
            200,  # min_size
            True,  # irgs_allowed
            array(
                [
                    [0, 3],  # region 1: indices 0-2
                    [1, 4],  # region 2: indices 1-3
                ],
            ),
            array(
                [
                    [0, 2],  # overlap 1: indices 0-1
                    [1, 3],  # overlap 2: indices 1-2
                    [2, 4],  # overlap 3: indices 2-3
                ],
            ),
            array([True, True]),  # both regions have risky overlaps
        ),
    ],
)
def test_NRowsPattern_get_fragmentation_risk(
    test_id: str,
    rg_n_rows: NDArray,
    df_n_rows: int,
    df_idx_tmrg_starts: NDArray,
    df_idx_tmrg_ends_excl: NDArray,
    row_group_size_target: int,
    irgs_allowed: bool,
    indices_enlarged: NDArray,
    indices_overlap: NDArray,
    expected: NDArray,
) -> None:
    """
    Test get_fragmentation_risk method with various inputs.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_n_rows : NDArray
        Number of rows in each row group.
    df_idx_tmrg_starts : NDArray
        Start indices in DataFrame for each row group overlap.
    df_idx_tmrg_ends_excl : NDArray
        End indices (exclusive) in DataFrame for each row group overlap.
    row_group_size_target : int
        Target size for row groups.
    irgs_allowed : bool
        Whether incomplete row groups are allowed.
    indices_enlarged : NDArray
        Array of shape (n, 2) containing start and end indices of enlarged
        regions.
    indices_overlap : NDArray
        Array of shape (m, 2) containing start and end indices of overlap regions.
    expected : NDArray
        Expected output indicating which enlarged regions have fragmentation risk.

    """
    pattern = NRowsSplitStrategy(
        rg_n_rows=rg_n_rows,
        df_n_rows=df_n_rows,
        df_idx_tmrg_starts=df_idx_tmrg_starts,
        df_idx_tmrg_ends_excl=df_idx_tmrg_ends_excl,
        row_group_size_target=row_group_size_target,
        irgs_allowed=irgs_allowed,
    )
    result = pattern.get_fragmentation_risk(
        indices_overlap=indices_overlap,
        indices_enlarged=indices_enlarged,
    )
    assert array_equal(result, expected)


@pytest.mark.parametrize(
    "test_id, rg_n_rows, row_group_size_target, df_idx_tmrg_starts, df_idx_tmrg_ends_excl, expected",
    [
        (
            "complete_rgs_split_regions",
            # Row groups with varying sizes
            # 0    1    2   3   4   5    6    7
            array([100, 100, 50, 50, 50, 100, 100, 100]),  # rg_n_rows
            100,  # row_group_size_target
            # DataFrame indices showing where it overlaps with row groups
            #      0  1   2   3   4   5   6   7
            array([0, 14, 14, 20, 34, 34, 34, 50]),  # df_idx_tmrg_starts
            array([1, 15, 15, 29, 35, 35, 35, 55]),  # df_idx_tmrg_ends_excl
            {
                # Complete row groups (rg_n_rows >= 80) at indices 0,1,5,6,7
                # and non-overlapping at indices 0,1,2,4,5,6
                # split the merge regions at indices 0,1,5
                "splits_merge_regions_starts": array([0, 1, 5]),
                "splits_merge_regions_ends_excl": array([1, 2, 7]),
            },
        ),
        (
            "no_complete_rgs",
            # All row groups incomplete
            [50, 40, 30, 20],  # rg_n_rows
            100,  # row_group_size_target
            array([0, 10, 20, 30]),  # df_idx_tmrg_starts
            array([5, 15, 25, 35]),  # df_idx_tmrg_ends_excl
            {
                # No complete row groups, so no splits
                "splits_merge_regions_starts": array([0]),
                "splits_merge_regions_ends_excl": array([4]),
            },
        ),
        (
            "all_complete_rgs",
            # All row groups complete and non-overlapping
            [100, 100, 100],  # rg_n_rows
            100,  # row_group_size_target
            array([0, 100, 200]),  # df_idx_tmrg_starts
            array([99, 199, 299]),  # df_idx_tmrg_ends_excl
            {
                # Each row group forms its own region
                "splits_merge_regions_starts": array([0, 1, 2]),
                "splits_merge_regions_ends_excl": array([1, 2, 3]),
            },
        ),
    ],
)
def test_consolidate_merge_plan(
    test_id: str,
    rg_n_rows: List[int],
    row_group_size_target: int,
    df_idx_tmrg_starts: NDArray,
    df_idx_tmrg_ends_excl: NDArray,
    expected: dict,
) -> None:
    """
    Test _rgst_as_int_merge_region_map function with various inputs.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_n_rows : List[int]
        Number of rows in each row group.
    row_group_size_target : int
        Target size for row groups.
    df_idx_tmrg_starts : NDArray
        Start indices in DataFrame for each row group overlap.
    df_idx_tmrg_ends_excl : NDArray
        End indices (exclusive) in DataFrame for each row group overlap.
    expected : dict
        Expected output containing split region indices.

    """
    # Act
    splits_starts, splits_ends = NRowsSplitStrategy.consolidate_merge_plan(
        rg_n_rows=rg_n_rows,
        row_group_size_target=row_group_size_target,
        df_idx_tmrg_starts=df_idx_tmrg_starts,
        df_idx_tmrg_ends_excl=df_idx_tmrg_ends_excl,
    )

    # Assert
    assert array_equal(splits_starts, expected["splits_merge_regions_starts"])
    assert array_equal(splits_ends, expected["splits_merge_regions_ends_excl"])


@pytest.mark.parametrize(
    "test_id, rg_maxs, row_group_period, df_ordered_on, expected",
    [
        (
            # Row groups before DataFrame.
            # pr     : 2h           4h           6h      8h
            # rg maxs:    2:30 3:00
            # df     :                 4:10 5:10    6:10
            "rg_leading_df_ending",
            # rg maxs
            [
                Timestamp(f"{REF_D}02:30"),
                Timestamp(f"{REF_D}03:00"),
            ],
            "2h",  # 2-hour periods
            # df
            Series(
                [
                    Timestamp(f"{REF_D}04:10"),
                    Timestamp(f"{REF_D}05:10"),
                    Timestamp(f"{REF_D}06:10"),
                ],
            ),
            # expected
            {
                "rg_idx_tmrg_ends_excl": [2, 2, 2],
                "df_idx_period_ends_excl": [0, 2, 3],
            },
        ),
        (
            # Row groups after DataFrame, with overlap.
            # pr     : 2h           4h        6h        8h
            # rg maxs:              4:00      6:00 7:59
            # df     :    2:10 3:10      4:10
            "df_leading_rg_ending",
            # rg maxs
            [
                Timestamp(f"{REF_D}04:00"),
                Timestamp(f"{REF_D}06:00"),
                Timestamp(f"{REF_D}07:59"),
            ],
            "2h",  # 2-hour periods
            # df
            Series(
                [
                    Timestamp(f"{REF_D}02:10"),
                    Timestamp(f"{REF_D}03:10"),
                    Timestamp(f"{REF_D}04:10"),
                ],
            ),
            # expected
            {
                "rg_idx_tmrg_ends_excl": [0, 1, 3],
                "df_idx_period_ends_excl": [2, 3, 3],
            },
        ),
        (
            # Consolidating row groups and DataFrame chunks when separated by
            #  several periods.
            # pr     : 2h           4h  6h             8h
            # rg maxs:                  6:00      7:59
            # df     :    2:10 3:10     6:00 6:40
            "consolidation_of_empty_period",
            # rg maxs
            [
                Timestamp(f"{REF_D}06:00"),
                Timestamp(f"{REF_D}07:59"),
            ],
            "2h",  # 2-hour periods
            # df
            Series(
                [
                    Timestamp(f"{REF_D}02:10"),
                    Timestamp(f"{REF_D}03:10"),
                    Timestamp(f"{REF_D}06:00"),
                    Timestamp(f"{REF_D}06:40"),
                ],
            ),
            # expected
            {
                "rg_idx_tmrg_ends_excl": [0, 2],
                "df_idx_period_ends_excl": [2, 4],
            },
        ),
        (
            # pr     : 8h        10h            12h       14h               16h
            # rg maxs: 8:00 9:00                          14:00       15:00
            # df     :              10:30 11:45                 14:15
            "start_on_lower_bound",
            # Row groups with gaps between periods
            [
                Timestamp(f"{REF_D}08:00"),
                Timestamp(f"{REF_D}09:00"),
                Timestamp(f"{REF_D}14:00"),
                Timestamp(f"{REF_D}15:00"),
            ],
            "2h",  # 2-hour periods
            Series(
                [
                    Timestamp(f"{REF_D}10:30"),
                    Timestamp(f"{REF_D}11:45"),
                    Timestamp(f"{REF_D}14:15"),
                ],
            ),
            {
                "rg_idx_tmrg_ends_excl": [2, 2, 4],
                "df_idx_period_ends_excl": [0, 2, 3],
            },
        ),
    ],
)
def test_rgst_as_str__merge_plan(
    test_id: str,
    rg_maxs: NDArray,
    row_group_period: str,
    df_ordered_on: Series,
    expected: dict,
) -> None:
    """
    Test _rgst_as_str__merge_plan function with various inputs.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_maxs : NDArray
        Maximum timestamps for each row group.
    row_group_period : str
        Period string (e.g., "2h" for 2-hour periods).
    df_ordered_on : Series
        Series containing ordered timestamps.
    expected : dict
        Expected output containing row group and DataFrame chunk indices.

    """
    # Act
    rg_idx_ends, df_idx_ends = TimePeriodSplitStrategy._rgst_as_str__merge_plan(
        rg_maxs=array(rg_maxs),
        row_group_period=row_group_period,
        df_ordered_on=df_ordered_on,
    )

    # Assert
    assert rg_idx_ends == expected["rg_idx_tmrg_ends_excl"]
    assert df_idx_ends == expected["df_idx_period_ends_excl"]


@pytest.mark.parametrize(
    (
        "test_id, df_data, pf_data, row_group_offsets, row_group_size_target, drop_duplicates, max_n_irgs, expected"
    ),
    [
        # 1/ Adding data at complete tail, testing 'drop_duplicates'.
        # 'max_n_irgs' is never triggered.
        (
            # Max row group size as int.
            # Writing after pf data, no incomplete row group.
            # rg:  0      1
            # pf: [0,1], [2,3]
            # df:               [3]
            "new_rg_simple_append_int",
            [3],
            [0, 1, 2, 3],
            [0, 2],  # row_group_offsets
            2,  # row_group_size_target | no irgs to merge with
            False,  # drop_duplicates | should not merge with preceding rg
            2,  # max_n_irgs | no irgs to rewrite
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Writing after pf data, no incomplete row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [12h10]
            "new_rg_simple_append_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}12:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size_target | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as int.
            # Writing at end of pf data, merging with incomplete row group.
            # rg:  0        1        2
            # pf: [0,1,2], [6,7,8], [9]
            # df:                   [9]
            "drop_duplicates_merge_tail_int",
            [9],
            [0, 1, 2, 6, 7, 8, 9],
            [0, 3, 6],  # row_group_offsets
            3,  # row_group_size_target | should not merge irg
            True,  # drop_duplicates | should merge with irg
            2,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data, not merging with incomplete row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [10h10]
            "no_drop_duplicates_simple_append_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size_target | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data, merging with incomplete row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:              [10h10]
            "drop_duplicates_merge_tail_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size_target | should not merge irg
            True,  # drop_duplicates | should merge with irg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values on boundary.
            # Writing after pf data, not merging with incomplete row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:                      [10h00]
            "no_drop_duplicates_simple_append_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}08:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size_target | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values on boundary.
            # Writing after pf data, merging with incomplete row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:              [10h00]
            "drop_duplicates_merge_tail_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size_target | should not merge irg
            True,  # drop_duplicates | should merge with irg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # Writing after pf data, incomplete row group to merge.
            # rg:  0        1        2    3
            # pf: [0,1,2], [3,4,5], [6], [7],
            # df:                             [8]
            "last_row_group_exceeded_merge_tail_int",
            [8],
            range(8),
            [0, 3, 6, 7],  # row_group_offsets
            3,  # row_group_size_target | should merge irgs
            False,  # drop_duplicates | should not merge with preceding rg
            4,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # Writing after pf data, incomplete row group should be merged.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [13h00]
            "last_row_group_exceeded_merge_tail_timestamp",
            [Timestamp(f"{REF_D}13:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target | new period, should merge irgs
            True,  # drop_duplicates | no duplicates to drop
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 2/ Adding data right at the start.
        (
            # Max row group size as int.
            # df at the start of pf data.
            # rg:         0       1       2    3
            # pf:        [2, 6], [7, 8], [9], [10]
            # df: [0,1]
            "no_duplicates_insert_at_start_new_rg_int",
            [0, 1],
            [2, 6, 7, 8, 9, 10],
            [0, 2, 4, 5],  # row_group_offsets
            2,  # row_group_size_target | df enough to make complete rg, should merge.
            True,  # no duplicates to drop
            2,  # max_n_irgs | not triggered
            {
                "chunk_counter": [2],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # df at the start of pf data.
            # rg:       0       1       2    3
            # pf:      [2, 6], [7, 8], [9], [10]
            # df: [0]
            "no_duplicates_insert_at_start_no_new_rg_int",
            [0],
            [2, 6, 7, 8, 9, 10],
            [0, 2, 4, 5],  # row_group_offsets
            2,  # row_group_size_target | df not enough to make complete rg, should not merge.
            True,  # no duplicates to drop
            2,  # max_n_irgs | not triggered
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # df at the start of pf data.
            # df is not overlapping with existing row groups.
            # rg:           0            1        2
            # pf:          [8h00,9h00], [12h00], [13h00]
            # df:  [7h30]
            "no_duplicates_insert_at_start_new_rg_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}7:30")],
            [
                Timestamp(f"{REF_D}08:00"),
                Timestamp(f"{REF_D}09:00"),
                Timestamp(f"{REF_D}12:00"),
                Timestamp(f"{REF_D}14:00"),
            ],
            [0, 2, 3],
            "2h",  # row_group_size_target | no rg in same period to merge with
            True,  # no duplicates to drop
            2,  # max_n_irgs | should rewrite tail
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # df at the start of pf data.
            # df is overlapping with existing row groups.
            # rg:            0            1        2
            # pf:           [8h10,9h10], [12h10], [13h10]
            # df:           [8h00]
            "no_duplicates_insert_at_start_no_new_rg_timestamp_on_boundary",
            [Timestamp(f"{REF_D}8:00")],
            [
                Timestamp(f"{REF_D}08:10"),
                Timestamp(f"{REF_D}09:10"),
                Timestamp(f"{REF_D}12:10"),
                Timestamp(f"{REF_D}14:10"),
            ],
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target | should merge with rg in same period
            True,  # no duplicates to drop
            2,  # max_n_irgs | should rewrite tail
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 3/ Adding data at complete end, testing 'max_n_irgs'.
        (
            # Max row group size as int
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_irgs_not_reached_simple_append_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as int
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_irgs_reached_tail_rewrite_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            2,  # max_n_irgs | should rewrite tail
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_irgs_not_reached_simple_append_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_irgs_reached_tail_rewrite_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target | should not merge with irg.
            False,  # drop_duplicates | should not merge with preceding rg
            2,  # max_n_irgs | should rewrite tail
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_irgs_none_simple_append_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],
            4,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            None,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_irgs_none_simple_append_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            None,  # max_n_irgs
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        # 4/ Adding data just before last incomplete row groups.
        (
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups.
            # rg:  0        1                    2
            # pf: [0,1,2], [6,7,8],             [11]
            # df:                   [8, 9, 10]
            "insert_before_irgs_simple_append_int",
            [8, 9, 10],
            [0, 1, 2, 6, 7, 8, 11],
            [0, 3, 6],
            3,  # row_group_size_target | no df remainder to merge with next rg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [3],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups
            # rg:  0        1                2
            # pf: [0,1,2], [6,7,8],         [11]
            # df:                   [8, 9]
            "insert_before_irgs_tail_rewrite_int",
            [8, 9],
            [0, 1, 2, 6, 7, 8, 11],
            [0, 3, 6],
            3,  # row_group_size_target | df remainder to merge with next rg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [0, 2],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Incomplete row groups at the end of pf data.
            # rg:  0        1           2
            # pf: [0,1,2], [6,7,8],    [10]
            # df:              [8, 9]
            "insert_before_irgs_drop_duplicates_no_tail_rewrite_int",
            [8, 9],
            [0, 1, 2, 6, 7, 8, 10],
            [0, 3, 6],
            3,  # row_group_size_target | because df merge with previous df,
            # df remainder should not merge with next rg
            True,  # drop_duplicates | merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                # Other acceptable solution:
                # [0, 1, 1, 2]
                "chunk_counter": [0, 2],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  11 (3.c) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Tail is rewritten because with df, 'max_n_irgs' is reached.
            # rg:  0        1        2
            # pf: [0,1,2], [6,7,8], [10]
            # df:              [8]
            "insert_before_irgs_drop_duplicates_tail_rewrite_int",
            [8],
            [0, 1, 2, 6, 7, 8, 10],
            [0, 3, 6],
            3,  # row_group_size_target | should not rewrite tail
            True,  # drop_duplicates | merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Incomplete row groups at the end of pf data.
            # Write of last row group is triggered
            # rg:  0         1                  2
            # pf: [0,1,2,3],[6,7,8,8],         [10]
            # df:                      [8, 9]
            "insert_before_irgs_tail_rewrite_int",
            [8, 9],
            [0, 1, 2, 3, 6, 7, 8, 8, 10],
            [0, 4, 8],
            4,  # row_group_size_target | should merge with next rg.
            False,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [0, 2],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  13 (3.c) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # 'max_n_irgs' reached to rewrite all tail.
            # row grps:  0            1                 2
            # pf: [8h00,9h00], [10h00],          [13h00]
            # df:                       [12h00]
            "insert_timestamp_max_n_irgs_tail_rewrite",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}12:00")]}),
            DataFrame(
                {
                    "ordered_on": [
                        Timestamp(f"{REF_D}08:00"),
                        Timestamp(f"{REF_D}09:00"),
                        Timestamp(f"{REF_D}10:00"),
                        Timestamp(f"{REF_D}13:00"),
                    ],
                },
            ),
            [0, 2, 3],
            "2h",  # row_group_size_target | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  14 (3.d) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # df is not directly connected to existing values in row
            # groups, so tail is not rewritten.
            # row grps:  0            1                 2
            # pf: [8h00,9h00], [10h00],          [13h00]
            # df:                       [11h00]
            "insert_timestamp_disconnected_no_rewrite",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame(
                {
                    "ordered_on": [
                        Timestamp(f"{REF_D}08:00"),
                        Timestamp(f"{REF_D}09:00"),
                        Timestamp(f"{REF_D}10:00"),
                        Timestamp(f"{REF_D}13:00"),
                    ],
                },
            ),
            [0, 2, 3],
            "2h",  # row_group_size_target | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  15 (3.e) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # df is not overlapping with existing row groups.
            # It should be added.
            # row grps:  0                   1        2
            # pf: [8h00,9h00],        [12h00], [14h00]
            # df:             [10h30]
            "insert_timestamp_non_overlapping",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}10:30")]}),
            DataFrame(
                {
                    "ordered_on": [
                        Timestamp(f"{REF_D}08:00"),
                        Timestamp(f"{REF_D}09:00"),
                        Timestamp(f"{REF_D}12:00"),
                        Timestamp(f"{REF_D}14:00"),
                    ],
                },
            ),
            [0, 2, 3],
            "2h",  # row_group_size_target | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        # 5/ Adding data in the middle of pf data.
        (
            # Test  16 (4.a) /
            # Max row group size as int | df within pf data.
            # Writing in-between pf data, with incomplete row groups at
            # the end of pf data.
            # row grps:  0      1            2       3    4
            # pf: [0,1], [2,      6], [7, 8], [9], [10]
            # df:        [2, 3, 4]
            "insert_middle_with_incomplete_rgs",
            DataFrame({"ordered_on": [2, 3, 4]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9, 10]}),
            [0, 2, 4, 6, 7],
            2,  # row_group_size_target | should rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            (1, 2, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  17 (4.b) /
            # Max row group size as int | df within pf data.
            # Writing in-between pf data, with incomplete row groups at
            # the end of pf data.
            # row grps:  0           1        2
            # pf: [0,1,2],    [6,7,8],[9]
            # df:         [3]
            "insert_middle_single_value",
            DataFrame({"ordered_on": [3]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9]}),
            [0, 3, 6],
            3,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates
            2,  # max_n_irgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  18 (4.c) /
            # Max row group size as int | df within pf data.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # One-but last row group is complete, but because df is
            # overlapping with it, it has to be rewritten.
            # By choice, the rewrite does not propagate till the end.
            # row grps:  0        1              2             3
            # pf: [0,1,2], [6,7,8],       [10, 11, 12], [13]
            # df:                  [9, 10]
            "insert_middle_partial_rewrite",
            DataFrame({"ordered_on": [9, 10]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10, 11, 12, 13]}),
            [0, 3, 6, 9],
            3,  # row_group_size_target | should not rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should not rewrite tail
            (2, 3, True),
        ),
        (
            # Test  19 (4.d) /
            # Max row group size as pandas freqstr | df within pf data.
            # Writing in-between pf data, with incomplete row groups at
            # the end of pf data.
            # row grps:  0        1           2           3      4
            # pf: [8h,9h], [10h, 11h], [12h, 13h], [14h], [15h]
            # df:               [11h]
            "insert_timestamp_middle_with_incomplete_rgs",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame({"ordered_on": date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=8)}),
            [0, 2, 4, 6, 7],
            "2h",  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            (1, 2, True),
        ),
        (
            # Test  20 (4.e) /
            # Max row group size as pandas freqstr | df within pf data.
            # Writing in-between pf data, with incomplete row groups at
            # the end of pf data.
            # row grps:  0          1           2           3      4
            # pf: [8h,9h],   [10h, 11h], [12h, 13h], [14h], [15h]
            # df:        [9h]
            "insert_timestamp_middle_no_rewrite",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}9:00")]}),
            DataFrame({"ordered_on": date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=8)}),
            [0, 2, 4, 6, 7],
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            2,  # max_n_irgs
            (None, None, True),
        ),
        # Do "island" cases
        # pf:         [0, 1]                [7, 9]                [ 15, 16]
        # df1                       [4, 5            11, 12]  # should have df_head, df_tail, no merge?
        # df2                                [ 8, 11, 12]     # should have merge + df_tail?
        # df3                       [4, 5      8]             # should have df_head + merge?
        # df4                       [4, 5]   # here df not to be merged with following row group
        # df5                       [4, 5, 6]   # here, should be merged
        # df6                                         + same with row_group_size_target as str
        # plenty of test with max_n_irgs set to 0 to check merging of last row groups
        # up to reach 4 x row_group_size_target
        # Do case when it is not possible to reach it, but showing all available
        # row groups are merge together nonetheless.
        # max_n_irgs
        # test with impossibility to have complete row groups with subset to merge
        # then check left, right, and all available row groups to merge
        #   pf:  rg1, rg2, rg3, rg4, rg5
        #   df:            df1
        #   will merge with only right or only left
        #
        #   pf:  rg1, rg2, rg3
        #   df:       df1
        #   will merge with right and left
        # test with freqstr, with several empty periods, to make sure the empty periods are
        #  not in the output
        # test with max_n_irgs set to 0, to make sure the last chunk is
        #  large enough to ensure complete row groups (calculation of "min_size"
        #  in "consolidate_merge_plan")
    ],
)
def test_compute_ordered_merge_plan(
    test_id,
    df_data,
    pf_data,
    row_group_offsets,
    row_group_size_target,
    drop_duplicates,
    max_n_irgs,
    expected,
    tmp_path,
):
    df = DataFrame({"ordered": df_data})
    pf_data = DataFrame({"ordered": pf_data})
    pf = create_parquet_file(tmp_path, pf_data, row_group_offsets=row_group_offsets)
    chunk_counter, sort_rgs_after_write = compute_ordered_merge_plan(
        df=df,
        pf=pf,
        ordered_on="ordered",
        row_group_size_target=row_group_size_target,
        drop_duplicates=drop_duplicates,
        max_n_irgs=max_n_irgs,
    )

    assert array_equal(chunk_counter, expected["chunk_counter"])
    assert sort_rgs_after_write == expected["sort_rgs_after_write"]


"""
        (
            "pf_leading_df_with_df_tail",
            [4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6],
            {
                "has_pf_head": True,
                "has_df_head": False,
                "has_tmrgs": True,
                "has_pf_tail": False,
                "has_df_tail": True,
                "df_idx_overlap_start": 0,
                "df_idx_overlap_end_excl": 3,
                "rg_idx_overlap_start": 1,
                "rg_idx_overlap_end_excl": 3,
                "df_idx_rg_starts": [0, 0, 1],
                "df_idx_rg_ends_excl": [0, 1, 3],
            },
            2,
        ),
        (
            "pf_leading_df_with_1_row_overlap",
            [4, 5, 6, 7],
            [1, 2, 3, 4],
            {
                "has_pf_head": True,
                "has_df_head": False,
                "has_tmrgs": True,
                "has_pf_tail": False,
                "has_df_tail": True,
                "df_idx_overlap_start": 0,
                "df_idx_overlap_end_excl": 1,
                "rg_idx_overlap_start": 1,
                "rg_idx_overlap_end_excl": 2,
                "df_idx_rg_starts": [0, 0],
                "df_idx_rg_ends_excl": [0, 1],
            },
            2,
        ),
        (
            "df_leading_pf_wo_df_tail",
            [1, 2, 3, 4, 5, 9],
            [4, 5, 6, 7, 8, 9],
            {
                "has_pf_head": False,
                "has_df_head": True,
                "has_tmrgs": True,
                "has_pf_tail": False,
                "has_df_tail": False,
                "df_idx_overlap_start": 3,
                "df_idx_overlap_end_excl": 6,
                "rg_idx_overlap_start": 0,
                "rg_idx_overlap_end_excl": 3,
                "df_idx_rg_starts": [3, 5, 5],
                "df_idx_rg_ends_excl": [5, 5, 6],
            },
            2,
        ),
        (
            "df_leading_pf_with_1_row_overlap",
            [1, 2, 3, 4, 5],
            [5, 6, 7, 8, 9],
            {
                "has_pf_head": False,
                "has_df_head": True,
                "has_tmrgs": True,
                "has_pf_tail": True,
                "has_df_tail": False,
                "df_idx_overlap_start": 4,
                "df_idx_overlap_end_excl": 5,
                "rg_idx_overlap_start": 0,
                "rg_idx_overlap_end_excl": 1,
                "df_idx_rg_starts": [4, 5, 5],
                "df_idx_rg_ends_excl": [5, 5, 5],
            },
            2,
        ),
        (
            "df_leading_by_row_group_size_sharp",
            [1, 2, 3, 4, 5, 6],
            [3, 5, 6, 7, 8, 9],
            {
                "has_pf_head": False,
                "has_df_head": True,
                "has_tmrgs": True,
                "has_pf_tail": True,
                "has_df_tail": False,
                "df_idx_overlap_start": 2,
                "df_idx_overlap_end_excl": 6,
                "rg_idx_overlap_start": 0,
                "rg_idx_overlap_end_excl": 2,
                "df_idx_rg_starts": [2, 5, 6],
                "df_idx_rg_ends_excl": [5, 6, 6],
            },
            2,
        ),
        (
            "pf_ending_after_dataframe",
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8],
            {
                "has_pf_head": False,
                "has_df_head": False,
                "has_tmrgs": True,
                "has_pf_tail": True,
                "has_df_tail": False,
                "df_idx_overlap_start": 0,
                "df_idx_overlap_end_excl": 5,
                "rg_idx_overlap_start": 0,
                "rg_idx_overlap_end_excl": 3,
                "df_idx_rg_starts": [0, 2, 4, 5],
                "df_idx_rg_ends_excl": [2, 4, 5, 5],
            },
            2,
        ),
        (
            "df_ending_after_pf_with_large_tail",
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6],
            {
                "has_pf_head": False,
                "has_df_head": False,
                "has_tmrgs": True,
                "has_pf_tail": False,
                "has_df_tail": True,
                "df_idx_overlap_start": 0,
                "df_idx_overlap_end_excl": 6,
                "rg_idx_overlap_start": 0,
                "rg_idx_overlap_end_excl": 3,
                "df_idx_rg_starts": [0, 2, 4],
                "df_idx_rg_ends_excl": [2, 4, 6],
            },
            2,
        ),
        (
            "no_overlap_pf_before_df",
            [7, 8, 9],
            [1, 2, 3, 4],
            {
                "has_pf_head": True,
                "has_df_head": False,
                "has_tmrgs": False,
                "has_pf_tail": False,
                "has_df_tail": True,
                "df_idx_overlap_start": None,
                "df_idx_overlap_end_excl": None,
                "rg_idx_overlap_start": None,
                "rg_idx_overlap_end_excl": None,
                "df_idx_rg_starts": [0, 0],
                "df_idx_rg_ends_excl": [0, 0],
            },
            2,
        ),
        (
            "no_overlap_df_before_pf",
            [1, 2, 3, 4],
            [7, 8, 9],
            {
                "has_pf_head": False,
                "has_df_head": True,
                "has_tmrgs": False,
                "has_pf_tail": True,
                "has_df_tail": False,
                "df_idx_overlap_start": None,
                "df_idx_overlap_end_excl": None,
                "rg_idx_overlap_start": None,
                "rg_idx_overlap_end_excl": None,
                "df_idx_rg_starts": [4, 4],
                "df_idx_rg_ends_excl": [4, 4],
            },
            2,
        ),"""
