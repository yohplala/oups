#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import Dict

import pytest
from numpy import array
from numpy import array_equal
from numpy import bool_
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import empty
from numpy import int_
from numpy import zeros
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.ordered_atomic_regions import NRowsMergeSplitStrategy
from oups.store.ordered_atomic_regions import OARMergeSplitStrategy
from oups.store.ordered_atomic_regions import TimePeriodMergeSplitStrategy
from oups.store.ordered_atomic_regions import get_region_indices_of_same_values
from oups.store.ordered_atomic_regions import get_region_indices_of_true_values
from oups.store.ordered_atomic_regions import get_region_start_end_delta
from oups.store.ordered_atomic_regions import set_true_in_regions
from tests.test_store.conftest import create_parquet_file


MIN = "min"
MAX = "max"
RG_IDX_START = "rg_idx_start"
RG_IDX_END_EXCL = "rg_idx_end_excl"
DF_IDX_END_EXCL = "df_idx_end_excl"
HAS_ROW_GROUP = "has_row_group"
HAS_DF_OVERLAP = "has_df_overlap"
RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS = "rg_idx_ends_excl_not_to_use_as_split_points"
REF_D = "2020/01/01 "


class TestOARMergeSplitStrategy(OARMergeSplitStrategy):
    """
    Concrete implementation for testing purposes.
    """

    def specialized_init(self, **kwargs):
        raise NotImplementedError("Test implementation only")

    def specialized_compute_merge_sequences(self):
        raise NotImplementedError("Test implementation only")

    def row_group_offsets(self, chunk: DataFrame, is_last_chunk: bool):
        raise NotImplementedError("Test implementation only")


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
    "length, regions, expected",
    [
        (  # Case 1: Single region at the start
            5,  # length
            array([[0, 2]]),  # regions
            array([True, True, False, False, False]),  # expected
        ),
        (  # Case 2: Single region in the middle
            5,
            array([[1, 3]]),
            array([False, True, True, False, False]),
        ),
        (  # Case 3: Single region at the end
            5,
            array([[3, 5]]),
            array([False, False, False, True, True]),
        ),
        (  # Case 4: Multiple non-overlapping regions
            7,
            array([[0, 2], [4, 6]]),
            array([True, True, False, False, True, True, False]),
        ),
        (  # Case 5: Empty regions
            3,
            array([], dtype=int).reshape(0, 2),  # empty regions
            array([False, False, False]),
        ),
        (  # Case 6: Region covering entire length
            4,
            array([[0, 4]]),
            array([True, True, True, True]),
        ),
    ],
)
def test_set_true_in_regions(length: int, regions: NDArray[int_], expected: NDArray[bool_]) -> None:
    """
    Test set_true_in_regions function to verify it correctly sets True values in
    specified regions.

    Parameters
    ----------
    length : int
        Length of the output array.
    regions : NDArray[np_int]
        2D array of shape (n, 2) containing [start, end) index pairs.
    expected : NDArray[np_bool]
        Expected boolean array with True values in specified regions.

    """
    result = set_true_in_regions(length=length, regions=regions)
    assert array_equal(result, expected)


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


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, drop_duplicates, expected",
    [
        (
            "no_gaps_rg_df",
            array([10, 20, 30]),  # rg_mins
            array([15, 25, 35]),  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 2]),
                RG_IDX_END_EXCL: array([1, 2, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_OVERLAP: array([True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_at_start_df_leading_rg",
            array([20, 30]),  # rg_mins
            array([25, 35]),  # rg_maxs
            Series([5, 22, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 0, 1]),
                RG_IDX_END_EXCL: array([0, 1, 2]),
                DF_IDX_END_EXCL: array([1, 2, 3]),
                HAS_ROW_GROUP: array([False, True, True]),
                HAS_DF_OVERLAP: array([True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_in_middle_df_not_overlapping_rg",
            array([10, 30]),  # rg_mins
            array([15, 35]),  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 1]),
                RG_IDX_END_EXCL: array([1, 1, 2]),
                DF_IDX_END_EXCL: array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, False, True]),
                HAS_DF_OVERLAP: array([True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_at_end_df_trailing_rg",
            array([10, 20]),  # rg_mins
            array([15, 25]),  # rg_maxs
            Series([12, 22, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 2]),
                RG_IDX_END_EXCL: array([1, 2, 2]),
                DF_IDX_END_EXCL: array([1, 2, 3]),
                HAS_ROW_GROUP: array([True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_at_start_rg_leading_df",
            array([0, 20, 30]),  # rg_mins
            array([5, 23, 33]),  # rg_maxs
            Series([22, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 2]),
                RG_IDX_END_EXCL: array([1, 2, 3]),
                DF_IDX_END_EXCL: array([0, 1, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_OVERLAP: array([False, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_in_middle_rg_not_overlapping_df",
            array([0, 10, 30]),  # rg_mins
            array([5, 15, 35]),  # rg_maxs
            Series([2, 32]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 2]),
                RG_IDX_END_EXCL: array([1, 2, 3]),
                DF_IDX_END_EXCL: array([1, 1, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_OVERLAP: array([True, False, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "gap_at_end_rg_trailing_df",
            array([10, 20, 30]),  # rg_mins
            array([15, 25, 35]),  # rg_maxs
            Series([12, 22]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 1, 2]),
                RG_IDX_END_EXCL: array([1, 2, 3]),
                DF_IDX_END_EXCL: array([1, 2, 2]),
                HAS_ROW_GROUP: array([True, True, True]),
                HAS_DF_OVERLAP: array([True, True, False]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "multiple_gaps_df_not_overlapping_rg",
            array([20, 40, 43]),  # rg_mins
            array([25, 42, 45]),  # rg_maxs
            Series([5, 22, 32, 41, 42, 46, 52]),  # df_ordered_on
            True,  # drop duplicates
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 5, 5, 7]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, True, False, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "no_drop_duplicates_with_gap_with_overlapping_rg",
            array([20, 40, 43]),  # rg_mins, 43 overlaps with previous rg max
            array([25, 43, 45]),  # rg_maxs
            Series([5, 22, 32, 43, 46, 52]),  # df_ordered_on - 43 is duplicate
            False,  # don't drop duplicates - 43 expected to fall in last rg
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 3, 4, 6]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, False, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "no_drop_duplicates_rg_same_min_max",
            array([20, 40, 43]),  # rg_mins, 43 overlaps with previous rg max
            array([25, 43, 43]),  # rg_maxs
            Series([5, 22, 32, 43, 46, 52]),  # df_ordered_on - 43 is duplicate
            False,  # don't drop duplicates - 43 expected to fall after last rg
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 3, 3, 6]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, False, False, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "drop_duplicates_rg_same_min_max",
            array([20, 40, 43]),  # rg_mins, 43 overlaps with previous rg max
            array([25, 43, 43]),  # rg_maxs
            Series([5, 22, 32, 43, 46, 52]),  # df_ordered_on - 43 is duplicate
            True,  # drop duplicates - 43 expected to fall after last rg
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 3, 4, 6]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([2]),
            },
        ),
        (
            "drop_duplicates_rg_same_min_max_df_same_value",
            array([20, 40, 43]),  # rg_mins, 43 overlaps with previous rg max
            array([25, 43, 43]),  # rg_maxs
            Series([5, 22, 32, 43, 43, 52]),  # df_ordered_on - 43 is duplicate
            True,  # drop duplicates - 43 expected to fall in 1st rg with 43
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 3, 5, 6]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([2]),
            },
        ),
        (
            "drop_duplicates_successive_rgs_same_min_max_df_same_value",
            array([40, 43]),  # rg_mins, 43 overlaps with previous rg max
            array([43, 45]),  # rg_maxs
            Series([22, 32, 43, 43, 44, 46]),  # df_ordered_on - 43 is duplicate
            True,  # drop duplicates - 43 expected to fall after last rg
            {
                RG_IDX_START: array([0, 0, 1, 2]),
                RG_IDX_END_EXCL: array([0, 1, 2, 2]),
                DF_IDX_END_EXCL: array([2, 2, 5, 6]),
                HAS_ROW_GROUP: array([False, True, True, False]),
                HAS_DF_OVERLAP: array([True, True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([1]),
            },
        ),
        (
            "drop_duplicates_successive_rgs_same_min_max_no_df_overlap",
            array([40, 43, 43, 50]),  # rg_mins, 43 overlaps with previous rg max
            array([43, 43, 45, 50]),  # rg_maxs
            Series([22, 32, 46, 50]),  # df_ordered_on - no 43
            True,  # drop duplicates - 43 expected to fall after last rg
            {
                RG_IDX_START: array([0, 0, 1, 2, 3, 3]),
                RG_IDX_END_EXCL: array([0, 1, 2, 3, 3, 4]),
                DF_IDX_END_EXCL: array([2, 2, 2, 2, 3, 4]),
                HAS_ROW_GROUP: array([False, True, True, True, False, True]),
                HAS_DF_OVERLAP: array([True, False, False, False, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "no_drop_duplicates_with_gap_wo_overlapping_rg",
            array([10, 20]),  # rg_mins
            array([15, 25]),  # rg_maxs
            Series([15, 16, 17, 22, 32]),  # df_ordered_on - note 15 is duplicate
            False,
            {
                RG_IDX_START: array([0, 1, 1, 2]),
                RG_IDX_END_EXCL: array([1, 1, 2, 2]),
                DF_IDX_END_EXCL: array([0, 3, 4, 5]),
                HAS_ROW_GROUP: array([True, False, True, False]),
                HAS_DF_OVERLAP: array([False, True, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
        (
            "with_timestamps",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   01/01,   15/01,                 , spans Jan
            #   2,         Feb,                     01/02,   15/02, spans Feb
            #   3,         Mar,   01/03,   15/03,                 , spans Mar
            #   4,         Apr,                     01/04,   15/04, spans Apr
            #   5,         May,   08/05,   12/05,   08/05,   10/05, spans May
            array(
                [Timestamp("2024-01-01"), Timestamp("2024-03-01"), Timestamp("2024-05-08")],
            ),  # rg_mins
            array(
                [Timestamp("2024-01-15"), Timestamp("2024-03-15"), Timestamp("2024-05-12")],
            ),  # rg_maxs
            Series(
                [
                    Timestamp("2024-02-01"),
                    Timestamp("2024-02-15"),
                    Timestamp("2024-04-01"),
                    Timestamp("2024-04-15"),
                    Timestamp("2024-05-08"),
                    Timestamp("2024-05-10"),
                ],
            ),
            False,
            {
                RG_IDX_START: array([0, 1, 1, 2, 2]),
                RG_IDX_END_EXCL: array([1, 1, 2, 2, 3]),
                DF_IDX_END_EXCL: array([0, 2, 2, 4, 6]),
                HAS_ROW_GROUP: array([True, False, True, False, True]),
                HAS_DF_OVERLAP: array([False, True, False, True, True]),
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
            },
        ),
    ],
)
def test_OARSplitStrategy_init(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: Series,
    drop_duplicates: bool,
    expected: NDArray,
) -> None:
    """
    Test OARSplitStrategy with various scenarios.
    """
    oars_prop = TestOARMergeSplitStrategy(
        rg_mins,
        rg_maxs,
        df_ordered_on,
        drop_duplicates,
    )
    # Check array fields
    assert_array_equal(oars_prop.oars_rg_idx_starts, expected[RG_IDX_START])
    assert_array_equal(oars_prop.oars_cmpt_idx_ends_excl[:, 0], expected[RG_IDX_END_EXCL].T)
    assert_array_equal(oars_prop.oars_cmpt_idx_ends_excl[:, 1], expected[DF_IDX_END_EXCL].T)
    assert_array_equal(oars_prop.oars_has_row_group, expected[HAS_ROW_GROUP])
    assert_array_equal(oars_prop.oars_has_df_overlap, expected[HAS_DF_OVERLAP])
    assert_array_equal(
        oars_prop.rg_idx_ends_excl_not_to_use_as_split_points,
        expected[RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS],
    )


@pytest.mark.parametrize(
    "test_id, rg_ordered_on_mins, rg_ordered_on_maxs, df_ordered_on, expected_error",
    [
        (
            "different_lengths",
            Series([Timestamp("2024-01-01"), Timestamp("2024-01-02")]).to_numpy(),  # rg_mins
            Series([Timestamp("2024-01-01")]).to_numpy(),  # rg_maxs - different length
            Series([Timestamp("2024-01-01"), Timestamp("2024-01-02")]),  # df_ordered_on
            "^rg_ordered_on_mins and rg_ordered_on_maxs",
        ),
        (
            "overlapping_row_groups",
            Series([Timestamp("2024-01-01"), Timestamp("2024-01-02")]).to_numpy(),  # rg_mins
            Series(
                [Timestamp("2024-01-03"), Timestamp("2024-01-04")],
            ).to_numpy(),  # rg_maxs - overlaps with next rg_min
            Series([Timestamp("2024-01-01"), Timestamp("2024-01-02")]),  # df_ordered_on
            "^row groups must not overlap",
        ),
        (
            "unsorted_df",
            Series([Timestamp("2024-01-01"), Timestamp("2024-01-03")]).to_numpy(),  # rg_mins
            Series([Timestamp("2024-01-02"), Timestamp("2024-01-04")]).to_numpy(),  # rg_maxs
            Series([Timestamp("2024-01-02"), Timestamp("2024-01-01")]),  # df_ordered_on - unsorted
            "^'df_ordered_on' must be sorted in ascending order.",
        ),
    ],
)
def test_OARSplitStrategy_validation(
    test_id: str,
    rg_ordered_on_mins: array,
    rg_ordered_on_maxs: array,
    df_ordered_on: Series,
    expected_error: Exception,
) -> None:
    """
    Test input validation in OARSplitStrategy.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_ordered_on_mins : array
        Array of minimum values for row groups.
    rg_ordered_on_maxs : array
        Array of maximum values for row groups.
    df_ordered_on : Series
        Series of ordered values.
    expected_error : Exception
        Expected error to be raised.

    """
    with pytest.raises(ValueError, match=expected_error):
        TestOARMergeSplitStrategy(
            rg_ordered_on_mins=rg_ordered_on_mins,
            rg_ordered_on_maxs=rg_ordered_on_maxs,
            df_ordered_on=df_ordered_on,
            drop_duplicates=True,
        )


@pytest.mark.parametrize(
    "test_id, oars_has_df_overlap, oars_likely_on_target_size, max_n_off_target_rgs, expected",
    [
        (  # Contiguous OARs with DataFrame chunk.
            # No need to enlarge since neighbors OARs are on target size.
            "contiguous_dfcs_no_off_target",
            array([False, True, True, False]),  # Has DataFrame chunk
            array([True, False, True, True]),  # On target size
            3,  # max_n_off_target_rgs - is not triggered
            array([[1, 3]]),  # Single region
        ),
        (  # First EMR has enough neighbor off-target OARs.
            # There is a second one potential EMR without DataFrame chunk.
            # The 3rd has not enough neighbor off-target OARs to be enlarged.
            "enlarging_based_on_off_target_oars",
            # OARS:   0,     1,     2,    3,     4,     5,     6,    7,     8
            # EMRs:  [0,                 4),                     [7,8)
            array([True, False, False, True, False, False, False, True, False]),
            array([False, False, False, False, True, False, True, False, False]),
            3,  # max_n_off_target_rgs
            array([[7, 8], [0, 4]]),  # Two regions: [0-1) and [2-4)
        ),
        (  # Confirm EMR because of likely on target OAR with DataFrame chunk.
            "enlarging_because_adding_likely_on_target_oar_with_dfc",
            array([False, True, False, False]),  # Has DataFrame chunk
            array([True, True, False, True]),  # On target size
            3,  # max_n_off_target_rgs
            array([[1, 3]]),  # Adding 3rd OAR in EMR.
        ),
        (  # Alternating regions with and without DataFrame chunks.
            # Should only merge regions with DataFrame chunks.
            "alternating_regions",
            array([True, False, True, False]),  # Alternating DataFrame chunks
            array([True, True, True, True]),  # All on target size
            2,  # max_n_off_target_rgs
            array([[0, 1], [2, 3]]),  # Two separate regions
        ),
        (  # Multiple off-target regions between DataFrame chunks
            # Should merge if number of off-target regions exceeds max_n_off_target_rgs
            "multiple_off_target_between_chunks",
            array([False, False, False, True]),  # DataFrame chunks at ends
            array([False, False, False, False]),  # All off target
            1,  # max_n_off_target_rgs
            array([[0, 4]]),  # Single region covering all
        ),
        (  # No regions with DataFrame chunks. Should return empty array
            "no_df_chunks",
            array([False, False, False]),  # No DataFrame chunks
            array([True, True, True]),  # All on target size
            1,  # max_n_off_target_rgs
            array([], dtype=int).reshape(0, 2),  # Empty array
        ),
        (  # Single off-target region with DataFrame chunk
            # Should return single region
            "single_off_target_with_df_chunk",
            array([True]),  # Has DataFrame chunk
            array([False]),  # Off target
            1,  # max_n_off_target_rgs
            array([[0, 1]]),  # Single region
        ),
        (  # Complex pattern with varying conditions
            # Tests multiple conditions in one test
            "mixed_pattern",
            array([True, False, False, False, True, False]),  # DataFrame chunks at 0, 4
            array([True, False, True, False, False, True]),  # Likely on target
            2,  # max_n_off_target_rgs
            array([[4, 5], [0, 2]]),
        ),
        (  # 'max_n_off_target' is None
            "max_n_off_target_is_none",
            array([True, False, False, False, True]),  # DataFrame chunks at 0, 4
            array([True, False, True, False, False]),  # Likely on target
            None,  # max_n_off_target_rgs
            array([[0, 1], [4, 5]]),
        ),
        (  # 'max_n_off_target' is 0
            "max_n_off_target_is_0",
            array([True, False, False, False, True]),  # DataFrame chunks at 0, 4
            array([False, False, True, False, False]),  # Likely on target
            0,  # max_n_off_target_rgs
            array([[0, 2], [3, 5]]),
        ),
    ],
)
def test_compute_merge_regions_start_ends_excl(
    test_id: str,
    oars_has_df_overlap: NDArray[bool_],
    oars_likely_on_target_size: NDArray[bool_],
    max_n_off_target_rgs: int,
    expected: NDArray[int_],
) -> None:
    """
    Test compute_merge_regions_start_ends_excl function with various inputs.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    oars_has_df_overlap : NDArray[bool_]
        Boolean array indicating if each atomic region has a DataFrame chunk.
    oars_likely_on_target_size : NDArray[bool_]
        Boolean array indicating if each atomic region is likely to be on target size.
    max_n_off_target_rgs : int
        Maximum number of off-target row groups allowed.
    expected : NDArray[int_]
        Expected output containing start and end indices for enlarged merge regions.

    """
    split_strat = TestOARMergeSplitStrategy(
        rg_ordered_on_mins=zeros(1, dtype=int_),
        rg_ordered_on_maxs=zeros(1, dtype=int_),
        df_ordered_on=Series([1]),
        drop_duplicates=True,
    )
    split_strat.oars_has_df_overlap = oars_has_df_overlap
    split_strat.oars_likely_on_target_size = oars_likely_on_target_size
    split_strat.n_oars = len(oars_has_df_overlap)
    split_strat.compute_merge_regions_start_ends_excl(
        max_n_off_target_rgs=max_n_off_target_rgs,
    )
    assert array_equal(split_strat.oar_idx_mrs_starts_ends_excl, expected)


def test_nrows_oars_likely_on_target_size():
    """
    Test initialization and 'oars_likely_on_target_size' method.
    """
    target_size = 100  # min size: 80
    # Create mock oars_desc with 5 regions:
    # 1. RG only (90 rows) - on target
    # 2. DF only (85 rows) - on target
    # 3. Both RG (50) and DF (40) - on target
    # 4. Both RG (30) and DF (30) - too small
    # 5. Both RG (120) and DF (0) - too large and only row group.
    # 6. Both RG (30) and DF (80) - too large but accepted since it has a dfc.
    # Set which regions have row groups and DataFrame chunks
    dummy_rg_idx = array([0, 1, 2, 3, 4, 5])
    oars_has_row_group = array([True, False, True, True, True, True])
    # Set up DataFrame chunk sizes through df_idx_end_excl
    oars_df_idx_ends_excl = array([0, 85, 125, 155, 155, 235])
    # Create row group sizes array
    rgs_n_rows = array([90, 50, 30, 120, 30])
    # Initialize strategy
    strategy = NRowsMergeSplitStrategy.from_oars_desc(
        oars_rg_idx_starts=dummy_rg_idx,
        oars_cmpt_idx_ends_excl=column_stack((dummy_rg_idx, oars_df_idx_ends_excl)),
        oars_has_row_group=oars_has_row_group,
        oars_has_df_overlap=diff(oars_df_idx_ends_excl, prepend=0).astype(bool),
        rg_idx_ends_excl_not_to_use_as_split_points=None,
        drop_duplicates=True,
        rgs_n_rows=rgs_n_rows,
        row_group_target_size=target_size,
    )
    # Expected results for oars_max_n_rows:
    # rg_n_rows:       [90,  0, 50, 30, 120,  30]
    # df_n_rows:       [ 0, 85, 40, 30,   0,  80]
    # oars_max_n_rows: [90, 85, 90, 60, 120, 110]
    assert_array_equal(strategy.oars_max_n_rows, array([90, 85, 90, 60, 120, 110]))
    # Expected results for likely_on_target_size:
    # 1. True  - RG only with 90 rows (between min_size and target_size)
    # 2. True  - DF only with 85 rows (between min_size and target_size)
    # 3. True  - Combined RG(50) + DF(40) could be on target
    # 4. False - Combined RG(30) + DF(30) too small
    # 5. False - RG only with 120 rows (above target_size)
    # 6. True - Combined RG(30) + DF(80) oversized but with a RG
    result = strategy.oars_likely_on_target_size
    expected = array([True, True, True, False, False, True])
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "test_id, drop_duplicates, oars_desc_dict, rgs_n_rows, row_group_target_size, oar_idx_mrs_starts_ends_excl, expected",
    [
        (
            "single_sequence_encompassing_all_oars_drop_duplicates",
            # row_group_target_size : 100
            # rgs_n_rows    :  [  50,   50,      ,   50,    50]
            # has_row_group :  [True, True, False, True,  True]
            # dfc_ends_excl :  [  10,   25,    60,   75,      ]
            # has_df_overlap:  [True, True,  True, True, False]
            True,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([10, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
            },
            array([50, 50, 50, 50]),  # rg_n_rows
            100,  # row_group_target_size
            array([[0, 5]]),  # merge region with all OARs
            {
                "oars_min_n_rows": array([50, 50, 35, 50, 50]),
                "oars_merge_sequences": [
                    (0, array([[2, 25], [4, 75]])),  # single sequence
                ],
            },
        ),
        (
            "single_sequence_encompassing_all_oars_wo_drop_duplicates",
            # This test case check that last row group in sequence is correctly
            # added to merge region.
            # row_group_target_size : 100
            # rgs_n_rows    :  [  50,   50,      ,   50,    50]
            # has_row_group :  [True, True, False, True,  True]
            # dfc_ends_excl :  [  10,   25,    60,   75,      ]
            # has_df_overlap:  [True, True,  True, True, False]
            False,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([10, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
            },
            array([50, 50, 50, 50]),  # rg_n_rows
            100,  # row_group_target_size
            array([[0, 5]]),  # merge region with all OARs
            {
                "oars_min_n_rows": array([60, 65, 35, 65, 50]),
                "oars_merge_sequences": [
                    (0, array([[2, 25], [4, 75]])),  # single sequence
                ],
            },
        ),
        (
            "small_single_sequence_drop_duplicates",
            # row_group_target_size : 60
            # rgs_n_rows    :  [   50,   50,      ,   50,    50]
            # has_row_group :  [ True, True, False, True,  True]
            # dfc_ends_excl :  [    0,   25,    60,   75,      ]
            # has_df_overlap:  [False, True,  True, True, False]
            True,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([0, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
            },
            array([50, 50, 50, 50]),  # rg_n_rows
            60,  # row_group_target_size
            array([[1, 4]]),  # merge region contains 3 OARs
            {
                "oars_min_n_rows": array([50, 50, 35, 50, 50]),
                "oars_merge_sequences": [
                    (1, array([[2, 60], [3, 75]])),  # single sequence
                ],
            },
        ),
        (
            "small_single_sequence_wo_drop_duplicates",
            # row_group_target_size : 35
            # rgs_n_rows    :  [   10,   10,      ,   30,    10]
            # has_row_group :  [ True, True, False, True,  True]
            # dfc_ends_excl :  [    0,   25,    60,   75,      ]
            # has_df_overlap:  [False, True,  True, True, False]
            False,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([0, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
            },
            array([10, 10, 30, 10]),  # rg_n_rows
            35,  # row_group_target_size
            array([[1, 4]]),  # merge region contains 3 OARs
            {
                "oars_min_n_rows": array([10, 35, 35, 45, 10]),
                "oars_merge_sequences": [
                    (1, array([[2, 25], [2, 60], [3, 75]])),  # single sequence
                ],
            },
        ),
        (
            "multiple_sequences_drop_duplicates",
            # row_group_target_size : 45
            # rgs_n_rows    :  [   50,   50,      ,   10,    60,   20,   20,             ]
            # has_row_group :  [ True, True, False, True,  True, True, True, False, False]
            # dfc_ends_excl :  [    0,   25,    60,   75,      ,   90,  120,   165,   190]
            # has_df_overlap:  [False, True,  True, True, False, True, True,  True,  True]
            True,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3, 4, 5, 6, 6]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4, 5, 6, 6, 6]),
                DF_IDX_END_EXCL: array([0, 25, 60, 75, 75, 90, 120, 165, 190]),
                HAS_ROW_GROUP: array([True, True, False, True, True, True, True, False, False]),
            },
            array([50, 50, 10, 60, 20, 20]),  # rg_n_rows
            45,  # row_group_target_size
            array([[1, 4], [5, 9]]),  # merge region
            {
                "oars_min_n_rows": array([50, 50, 35, 15, 60, 20, 30, 45, 25]),
                "oars_merge_sequences": [
                    (1, array([[2, 25], [3, 75]])),
                    (4, array([[6, 120], [6, 190]])),
                ],
            },
        ),
        (
            "multiple_sequences_wo_drop_duplicates",
            # row_group_target_size : 45
            # rgs_n_rows    :  [   50,   50,      ,   10,    60,   20,   20,             ]
            # has_row_group :  [ True, True, False, True,  True, True, True, False, False]
            # dfc_ends_excl :  [    0,   25,    60,   75,      ,   90,  120,   165,   210]
            # has_df_overlap:  [False, True,  True, True, False, True, True,  True,  True]
            False,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3, 4, 5, 6, 6]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4, 5, 6, 6, 6]),
                DF_IDX_END_EXCL: array([0, 25, 60, 75, 75, 90, 120, 165, 210]),
                HAS_ROW_GROUP: array([True, True, False, True, True, True, True, False, False]),
            },
            array([50, 50, 10, 60, 20, 20]),  # rg_n_rows
            45,  # row_group_target_size
            array([[1, 4], [5, 9]]),  # merge region
            {
                "oars_min_n_rows": array([50, 75, 35, 25, 60, 35, 50, 45, 45]),
                "oars_merge_sequences": [
                    (1, array([[2, 25], [2, 60], [3, 75]])),
                    (4, array([[6, 120], [6, 165], [6, 210]])),
                ],
            },
        ),
        (
            "small_row_group_target_size",
            # row_group_target_size : 10
            # rgs_n_rows    :  [  50,   50,      ,   50,    50]
            # has_row_group :  [True, True, False, True,  True]
            # dfc_ends_excl :  [  10,   25,    60,   75,      ]
            # has_df_overlap:  [True, True,  True, True, False]
            True,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([0, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
            },
            array([50, 50, 50, 50]),  # rg_n_rows
            10,  # row_group_target_size
            array([[1, 4]]),  # merge region with all OARs
            {
                "oars_min_n_rows": array([50, 50, 35, 50, 50]),
                "oars_merge_sequences": [
                    (1, array([[2, 25], [2, 60], [3, 75]])),  # single sequence
                ],
            },
        ),
    ],
)
def test_nrows_specialized_compute_merge_sequences(
    test_id: str,
    drop_duplicates: bool,
    oars_desc_dict: Dict[str, NDArray],
    rgs_n_rows: NDArray,
    row_group_target_size: int,
    oar_idx_mrs_starts_ends_excl: NDArray,
    expected: Dict,
) -> None:
    """
    Test 'specialized_compute_merge_sequences' method.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    drop_duplicates : bool
        Whether to drop duplicates between row groups and DataFrame.
    oars_desc_dict : Dict[str, NDArray]
        Dictionary containing the oars_desc array.
    rg_n_rows : NDArray
        Array of shape (n) containing the number of rows in each row group.
    row_group_target_size : int
        Target number of rows above which a new row group should be created.
    oar_idx_mrs_starts_ends_excl : NDArray
        Array of shape (n, 2) containing start and end indices (excluded)
        for each merge region to be consolidated.
    expected : List[Tuple[int, NDArray]]
        List of expected tuples, where each tuple contains:
        - int: Start index of the first row group in the merge sequence
        - NDArray: Array of shape (m, 2) containing end indices (excluded) for
          row groups and DataFrame chunks in the merge sequence

    """
    # Initialize strategy with target size of 100 rows
    strategy = NRowsMergeSplitStrategy.from_oars_desc(
        oars_rg_idx_starts=oars_desc_dict[RG_IDX_START],
        oars_cmpt_idx_ends_excl=column_stack(
            (oars_desc_dict[RG_IDX_END_EXCL], oars_desc_dict[DF_IDX_END_EXCL]),
        ),
        oars_has_row_group=oars_desc_dict[HAS_ROW_GROUP],
        oars_has_df_overlap=diff(oars_desc_dict[DF_IDX_END_EXCL], prepend=0).astype(bool),
        rg_idx_ends_excl_not_to_use_as_split_points=None,
        drop_duplicates=drop_duplicates,
        rgs_n_rows=rgs_n_rows,
        row_group_target_size=row_group_target_size,
    )
    assert_array_equal(strategy.oars_min_n_rows, expected["oars_min_n_rows"])
    # Test specialized_compute_merge_sequences.
    strategy.oar_idx_mrs_starts_ends_excl = oar_idx_mrs_starts_ends_excl
    result = strategy.specialized_compute_merge_sequences()
    # Check.
    assert len(result) == len(expected["oars_merge_sequences"])
    for (result_rg_start, result_cmpt_ends_excl), (
        expected_rg_start,
        expected_cmpt_ends_excl,
    ) in zip(result, expected["oars_merge_sequences"]):
        assert result_rg_start == expected_rg_start
        assert_array_equal(result_cmpt_ends_excl, expected_cmpt_ends_excl)


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, row_group_target_size, drop_duplicates, rgs_n_rows, max_n_off_target, expected",
    [
        (
            "complex_case_with_drop_duplicates",
            # row_group_target_size : 3
            # rg_idx        :     0,  1,  2,  3,  4,  5,  6,          7
            # rg_mins       :  [  1,  3,  8, 11, 12  14, 14,     19]
            # rg_maxs       :  [  3,  6, 10, 11, 14, 14, 14,         22]
            # rgs_n_rows    :  [  3,  3,  3,  1,  3,  3,  1,          2]
            # df_ordered_on :  [                         14, 18, 19, 20, 26]
            # df_idx        :                             0,  1,  2,  3,  4
            #
            # rg_idx_ends_excl_not_to_use_as_split_points :  [5, 6]
            # merge regions :  [(3, [[7, 2], [8, 5]])]
            array([1, 3, 8, 11, 12, 14, 14, 19]),  # rg_mins
            array([3, 6, 10, 11, 14, 14, 14, 22]),  # rg_maxs
            Series([14, 18, 19, 20, 26]),  # df_ordered_on
            3,  # row_group_target_size
            True,  # drop_duplicates
            [3, 3, 3, 1, 3, 3, 1, 2],  # rgs_n_rows
            1,  # max_n_off_target
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([5, 6]),
                "oars_merge_sequences": [
                    (3, array([[7, 2], [8, 5]])),
                ],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Writing after pf data, no off target size row group.
            # rg:  0      1
            # pf: [0,1], [2,3]
            # df:               [3]
            "new_rg_simple_append",
            array([0, 2]),  # rg_mins
            array([1, 3]),  # rg_maxs
            Series([3]),  # df_ordered_on
            2,  # row_group_target_size | no incomplete rgs to merge with
            False,  # drop_duplicates | should not merge with preceding rg
            [2, 2],  # rgs_n_rows
            2,  # max_n_off_target_rgs | no incomplete rgs to rewrite
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Writing at end of pf data, merging with off target size row group.
            # rg:  0        1        2
            # pf: [0,1,2], [6,7,8], [9]
            # df:                   [9]
            "drop_duplicates_merge_tail",
            array([0, 6, 9]),  # rg_mins
            array([2, 8, 9]),  # rg_maxs
            Series([9]),  # df_ordered_on
            3,  # row_group_target_size | should not merge incomplete rg
            True,  # drop_duplicates | should merge with incomplete rg
            [3, 3, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df within pf data.
            # Writing at end of pf data, with off target size row groups at
            # the end of pf data.
            # One-but last row group is on target size, but because df is
            # overlapping with it, it has to be rewritten.
            # By choice, the rewrite does not propagate till the end.
            # rg:  0        1          2             3
            # pf: [0,1,2], [6,7,8],   [10, 11, 12], [13]
            # df:                  [9, 10]
            "insert_middle_partial_rewrite",
            array([0, 6, 10, 13]),  # rg_mins
            array([2, 8, 12, 13]),  # rg_maxs
            Series([9, 10]),  # df_ordered_on
            3,  # row_group_target_size | should not rewrite tail
            True,  # drop_duplicates
            [3, 3, 3, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[4, 2]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Writing after pf data, off target size row group to merge.
            # rg:  0        1        2    3
            # pf: [0,1,2], [3,4,5], [6], [7],
            # df:                             [8]
            "last_row_group_exceeded_merge_tail",
            array([0, 3, 6, 7]),  # rg_mins
            array([2, 5, 6, 7]),  # rg_maxs
            Series([8]),  # df_ordered_on
            3,  # row_group_size | should merge irgs
            False,  # drop_duplicates | should not merge with preceding rg
            [3, 3, 1, 1],  # rgs_n_rows
            4,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(4, array([[4, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df at the start of pf data.
            # rg:         0       1       2    3
            # pf:        [2, 6], [7, 8], [9], [10]
            # df: [0,1]
            "no_duplicates_insert_at_start_new_rg",
            array([2, 7, 9, 10]),  # rg_mins
            array([6, 8, 9, 10]),  # rg_maxs
            Series([0, 1]),  # df_ordered_on
            2,  # row_group_size | df enough to make on target size rg, should merge.
            True,  # no duplicates to drop
            [2, 2, 1, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | not triggered
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[0, 2]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # df at the start of pf data.
            # rg:       0       1       2    3
            # pf:      [2, 6], [7, 8], [9], [10]
            # df: [0]
            "no_duplicates_insert_at_start_no_new_rg",
            array([2, 7, 9, 10]),  # rg_mins
            array([6, 8, 9, 10]),  # rg_maxs
            Series([0]),  # df_ordered_on
            2,  # row_group_size | df not enough to make on target size rg, should not merge.
            True,  # no duplicates to drop
            [2, 2, 1, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | not triggered
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[0, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_off_target_rgs_not_reached_simple_append",
            array([0, 7, 11, 12]),  # rg_mins
            array([6, 10, 11, 12]),  # rg_maxs
            Series([12]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 1, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(4, array([[4, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df within pf data.
            # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # rg:  0      1            2       3    4
            # pf: [0,1], [2,      6], [7, 8], [9], [10]
            # df:        [2, 3, 4]
            "insert_middle_with_off_target_rgs",
            array([0, 2, 7, 9, 10]),  # rg_mins
            array([1, 6, 8, 9, 10]),  # rg_maxs
            Series([2, 3, 4]),  # df_ordered_on
            2,  # row_group_size | should rewrite tail
            True,  # drop_duplicates
            [2, 2, 2, 1, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[2, 3]]))],
                "sort_rgs_after_write": True,  # bool: need to sort rgs after write
            },
        ),
        (
            # df within pf data.
            # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # rg:  0           1        2
            # pf: [0,1,2],    [6,7,8],[9]
            # df:         [3]
            "insert_middle_single_value",
            array([0, 6, 9]),  # rg_mins
            array([2, 8, 9]),  # rg_maxs
            Series([3]),  # df_ordered_on
            3,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates
            [3, 3, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[1, 1]]))],
                "sort_rgs_after_write": True,  # bool: need to sort rgs after write
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_off_target_rgs_reached_tail_rewrite",
            array([0, 7, 11, 12]),  # rg_mins
            array([6, 10, 11, 12]),  # rg_maxs
            Series([12]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 1, 1],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[4, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups.
            # rg:  0        1                    2
            # pf: [0,1,2], [6,7,8],             [11]
            # df:                   [8, 9, 10]
            "insert_before_incomplete_rgs_simple_append",
            array([0, 6, 11]),  # rg_mins
            array([2, 8, 11]),  # rg_maxs
            Series([8, 9, 10]),  # df_ordered_on
            3,  # row_group_size | full df written, triggers tail rwrite
            False,  # drop_duplicates | should not merge with preceding rg
            [3, 3, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[3, 3]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups
            # rg:  0          1               2
            # pf: [0,1,2,2], [6,7,8,8],      [11]
            # df:                      [8,9]
            "insert_before_incomplete_rgs_no_tail_rewrite",
            array([0, 6, 11]),  # rg_mins
            array([2, 8, 11]),  # rg_maxs
            Series([8, 9]),  # df_ordered_on
            4,  # row_group_target_size | df remainder not to merge with next rg
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 2]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # df connected to off target size rgs.
            # A rg reaching target size is written, this triggers a full rewrite.
            # Because what remains to be written does not reach target size, a
            # single load is planned, to load row groups 1 and 2
            # Incomplete row groups at the end of pf data.
            # rg:  0        1           2
            # pf: [0,1,2], [6,7,8],    [10]
            # df:              [8, 9]
            "insert_before_incomplete_rgs_drop_duplicates_tail_rewrite_1",
            array([0, 6, 10]),  # rg_mins
            array([2, 8, 10]),  # rg_maxs
            Series([8, 9]),  # df_ordered_on
            3,  # row_group_size | because df merge with previous rg,
            # df remainder should not merge with next rg
            True,  # drop_duplicates | merge with preceding rg
            [3, 3, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[3, 2]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups at
            # the end of pf data.
            # Tail is rewritten because a 'full' row group is written.
            # rg:  0        1        2
            # pf: [0,1,2], [6,7,8], [10]
            # df:              [8]
            "insert_before_incomplete_rgs_drop_duplicates_tail_rewrite_2",
            array([0, 6, 10]),  # rg_mins
            array([2, 8, 10]),  # rg_maxs
            Series([8]),  # df_ordered_on
            3,  # row_group_size | should not rewrite tail
            True,  # drop_duplicates | merge with preceding rg
            [3, 3, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # df connected to off target size rgs.
            # Incomplete row groups at the end of pf data.
            # Write of last row group is triggered
            # rg:  0         1                  2
            # pf: [0,1,2,3],[6,7,8,8],         [10]
            # df:                      [8, 9]
            "insert_before_incomplete_rgs_tail_rewrite",
            array([0, 6, 10]),  # rg_mins
            array([3, 8, 10]),  # rg_maxs
            Series([8, 9]),  # df_ordered_on
            4,  # row_group_size | should not merge with next rg.
            False,  # drop_duplicates
            [4, 4, 1],  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 2]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_off_target_rgs_none_simple_append",
            array([0, 7, 11, 12]),  # rg_mins
            array([6, 10, 11, 12]),  # rg_maxs
            Series([12]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 1, 1],  # rgs_n_rows
            None,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(4, array([[4, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
    ],
)
def test_nrows_integration_compute_merge_sequences(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: NDArray,
    row_group_target_size: int,
    drop_duplicates: bool,
    rgs_n_rows: NDArray,
    max_n_off_target: int,
    expected: Dict,
) -> None:
    """
    Integration test for 'compute_merge_sequences' method.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_mins : NDArray
        Array of shape (n) containing the minimum values of the row groups.
    rg_maxs : NDArray
        Array of shape (n) containing the maximum values of the row groups.
    df_ordered_on : NDArray
        Array of shape (m) containing the values of the DataFrame to be ordered on.
    row_group_target_size : int
        Target number of rows above which a new row group should be created.
    drop_duplicates : bool
        Whether to drop duplicates between row groups and DataFrame.
    rgs_n_rows : NDArray
        Array of shape (n) containing the number of rows in each row group.
    max_n_off_target : int
        Maximum number of off-target row groups allowed.
    expected : Dict
        Dictionary containing the expected results.

    """
    # Initialize strategy.
    strategy = NRowsMergeSplitStrategy(
        rg_ordered_on_mins=rg_mins,
        rg_ordered_on_maxs=rg_maxs,
        df_ordered_on=df_ordered_on,
        rgs_n_rows=rgs_n_rows,
        row_group_target_size=row_group_target_size,
        drop_duplicates=drop_duplicates,
    )
    if expected[RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS] is not None:
        assert_array_equal(
            strategy.rg_idx_ends_excl_not_to_use_as_split_points,
            expected[RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS],
        )
    else:
        assert strategy.rg_idx_ends_excl_not_to_use_as_split_points is None
    # Compute merge sequences.
    strategy.compute_merge_sequences(
        max_n_off_target_rgs=max_n_off_target,
    )
    # Check.
    assert strategy.sort_rgs_after_write == expected["sort_rgs_after_write"]
    assert len(strategy.filtered_merge_sequences) == len(expected["oars_merge_sequences"])
    for (result_rg_idx_start, result_cmpt_ends_excl), (
        expected_rg_idx_start,
        expected_cmpt_ends_excl,
    ) in zip(strategy.filtered_merge_sequences, expected["oars_merge_sequences"]):
        assert result_rg_idx_start == expected_rg_idx_start
        assert_array_equal(result_cmpt_ends_excl, expected_cmpt_ends_excl)


@pytest.mark.parametrize(
    "df_size,target_size,expected_offsets",
    [
        # DataFrame larger than target size, no remainder.
        (100, 20, [0, 20, 40, 60, 80]),
        # DataFrame smaller than target size.
        (15, 20, [0]),
        # DataFrame larger than target size, with remainder.
        (45, 20, [0, 20, 40]),
        # Single row DataFrame.
        (1, 20, [0]),
    ],
)
def test_nrows_row_group_offsets(df_size, target_size, expected_offsets):
    # Create test data
    # Initialize strategy
    strategy = NRowsMergeSplitStrategy(
        rg_ordered_on_mins=Series([0]),  # dummy value
        rg_ordered_on_maxs=Series([1]),  # dummy value
        df_ordered_on=Series([0]),  # dummy value
        rgs_n_rows=Series([1]),  # dummy value
        row_group_target_size=target_size,
    )
    # Get offsets
    offsets = strategy.row_group_offsets(df_ordered_on=Series(range(df_size)))
    # Verify results
    assert offsets == expected_offsets


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, expected",
    [
        (
            "two_rgs_in_period",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Dec,   15/12,   19/12,                 , False (2 ARMs in period)
            #   2,         Dec,   20/12,   28/12,                 , False (2 ARMs in period)
            #   3,         Mar,                     15/03,   15/03, True (one DFc in period)
            array([Timestamp("2023-12-15"), Timestamp("2023-12-20")]),  # rg_mins
            array([Timestamp("2023-12-19"), Timestamp("2023-12-28")]),  # rg_maxs
            Series([Timestamp("2024-03-15")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2023-12-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-04-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [
                            Timestamp("2023-12-15"),
                            Timestamp("2023-12-20"),
                            Timestamp("2024-03-15"),
                        ],
                        "maxs": [
                            Timestamp("2023-12-19"),
                            Timestamp("2023-12-28"),
                            Timestamp("2024-03-15"),
                        ],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, False, True]),
            },
        ),
        (
            "rg_spans_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Jan,   01/01,                          , False (RG spans several periods)
            #   1,         Feb,            01/02,                 , same OAR as above
            #   2,         Mar,                     15/03,   15/03, True (one DFc in period)
            array([Timestamp("2024-01-01")]),  # rg_mins, value on edge
            array([Timestamp("2024-02-01")]),  # rg_maxs, value on edge
            Series([Timestamp("2024-03-15")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-01-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-04-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-01-01"), Timestamp("2024-03-15")],
                        "maxs": [Timestamp("2024-02-01"), Timestamp("2024-03-15")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, True]),
            },
        ),
        (
            "rg_and_dfc_in_same_oar",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Mar,   15/03,   16/03,   15/03,   15/03, True (both RG & DFc in period)
            array([Timestamp("2024-03-15")]),  # rg_mins
            array([Timestamp("2024-03-16")]),  # rg_maxs
            Series([Timestamp("2024-03-15")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-03-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-04-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-03-15")],
                        "maxs": [Timestamp("2024-03-16")],
                    },
                ).to_numpy(),
                "likely_on_target": array([True]),
            },
        ),
        (
            "dfc_spans_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Apr,                     17/04,        , True (DFc spans several periods)
            #   1,         May,                              03/05,
            #   2,         Jun,   10/06,   15/06,                 , True
            array([Timestamp("2024-06-10")]),  # rg_mins
            array([Timestamp("2024-06-15")]),  # rg_maxs
            Series([Timestamp("2024-04-17"), Timestamp("2024-05-03")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-04-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-07-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-04-17"), Timestamp("2024-06-10")],
                        "maxs": [Timestamp("2024-05-03"), Timestamp("2024-06-15")],
                    },
                ).to_numpy(),
                "likely_on_target": array([True, True]),
            },
        ),
        (
            "rg_and_dfc_in_same_period",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Jun,   10/06,   15/06,                 , False (both RG & DFc in period)
            #   2,         Jun,                     16/06,   18/06, False (both RG & DFc in period)
            array([Timestamp("2024-06-10")]),  # rg_mins
            array([Timestamp("2024-06-15")]),  # rg_maxs
            Series([Timestamp("2024-06-16"), Timestamp("2024-06-18")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-06-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-07-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-06-10"), Timestamp("2024-06-16")],
                        "maxs": [Timestamp("2024-06-15"), Timestamp("2024-06-18")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, False]),
            },
        ),
        (
            "dfc_ends_in period_with_a_rg",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Jul,                     15/07,        , False (DFc spans several periods)
            #   1,         Aug,                              10/08,
            #   2,         Aug,   15/08,   17/08,                 , False (both RG & DFc in period)
            array([Timestamp("2024-08-15")]),  # rg_mins
            array([Timestamp("2024-08-17")]),  # rg_maxs
            Series([Timestamp("2024-07-15"), Timestamp("2024-08-10")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-07-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-09-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-07-15"), Timestamp("2024-08-15")],
                        "maxs": [Timestamp("2024-08-10"), Timestamp("2024-08-17")],
                    },
                ).to_numpy(),
                "likely_on_target": array([True, False]),
            },
        ),
        (
            "rg_ends_in_period_with_a_dfc",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Sep,   11/09,                          , False (RG spans several periods)
            #   1,         Oct,            15/10,
            #   2,         Oct,                     16/10,   18/10, False (both RG & DFc in period)
            array([Timestamp("2024-09-11")]),  # rg_mins
            array([Timestamp("2024-10-15")]),  # rg_maxs
            Series([Timestamp("2024-10-16"), Timestamp("2024-10-18")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-09-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-11-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-09-11"), Timestamp("2024-10-16")],
                        "maxs": [Timestamp("2024-10-15"), Timestamp("2024-10-18")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, False]),
            },
        ),
        (
            "rg_starts_in_period_with_a_dfc",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Nov,                     15/11,   17/11, False (both RG & DFc in period)
            #   2,         Nov,   18/11,                          , False (RG spans several periods)
            #   2,         Dec,            05/12,
            array([Timestamp("2024-11-18")]),  # rg_mins
            array([Timestamp("2024-12-05")]),  # rg_maxs
            Series([Timestamp("2024-11-15"), Timestamp("2024-11-17")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-11-01"),  # Floor of earliest timestamp
                    end=Timestamp("2025-01-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-11-15"), Timestamp("2024-11-18")],
                        "maxs": [Timestamp("2024-11-17"), Timestamp("2024-12-05")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, False]),
            },
        ),
        (
            "dfc_starts_in_period_with_a_rg",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #  14,         Jan,   01/01,   04/01,                 , False (both RG & DFc in period)
            #  15,         Jan,                     16/01,        , False (DFc spans several periods)
            #  15,         Feb,                              01/02,
            array([Timestamp("2024-01-01")]),  # rg_mins
            array([Timestamp("2024-01-04")]),  # rg_maxs
            Series([Timestamp("2024-01-16"), Timestamp("2024-02-01")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-01-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-03-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-01-01"), Timestamp("2024-01-16")],
                        "maxs": [Timestamp("2024-01-04"), Timestamp("2024-02-01")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, True]),
            },
        ),
        (
            "rg_and_dfc_ok",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Mar,   01/03,   02/03,                 , True (single RG in period)
            #   2,         Apr,                     01/04,   30/04, True (single DFc in period)
            array([Timestamp("2024-03-01")]),  # rg_mins
            array([Timestamp("2024-03-02")]),  # rg_maxs
            Series([Timestamp("2024-04-01"), Timestamp("2024-04-30")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-03-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-05-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-03-01"), Timestamp("2024-04-01")],
                        "maxs": [Timestamp("2024-03-02"), Timestamp("2024-04-30")],
                    },
                ).to_numpy(),
                "likely_on_target": array([True, True]),
            },
        ),
        (
            "rg_and_dfc_end_in_period_with_a_dfc",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         May,   01/05,            01/05,        , False (both RG & DFC)
            #   1,         Jun,            02/06,            01/06,
            #   2,         Jun,                     04/06,   30/06, False (past RG & DFC ending in period)
            array([Timestamp("2024-05-01")]),  # rg_mins
            array([Timestamp("2024-06-02")]),  # rg_maxs
            Series(
                [
                    Timestamp("2024-05-01"),
                    Timestamp("2024-06-01"),
                    Timestamp("2024-06-04"),
                    Timestamp("2024-06-30"),
                ],
            ),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-05-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-07-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-05-01"), Timestamp("2024-06-04")],
                        "maxs": [Timestamp("2024-06-02"), Timestamp("2024-06-30")],
                    },
                ).to_numpy(),
                "likely_on_target": array([True, False]),
            },
        ),
        (
            "rg_and_dfc_start_in_period_with_a_rg",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Apr,   01/04,   28/04,                 , False (next RG & DFC start in period)
            #   2,         May,   29/04,            01/05,        , False (both RG & DFC)
            #   2,         Jun,            02/06,            01/06,
            array([Timestamp("2024-04-01"), Timestamp("2024-04-29")]),  # rg_mins
            array([Timestamp("2024-04-28"), Timestamp("2024-06-02")]),  # rg_maxs
            Series([Timestamp("2024-05-01"), Timestamp("2024-06-01")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-04-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-07-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-04-01"), Timestamp("2024-04-29")],
                        "maxs": [Timestamp("2024-04-28"), Timestamp("2024-06-02")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, True]),
            },
        ),
        (
            "rg_end_and_dfc_start_in_same_period",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max, meets target size
            #   1,         Apr,   01/04,                          , False (rg spans 2 periods)
            #   1,         May,            15/05,
            #   2,         May,                     18/05,        , False (dfc spans 2 periods)
            #   2,         Jun,                              01/06,
            array([Timestamp("2024-04-01")]),  # rg_mins
            array([Timestamp("2024-05-15")]),  # rg_maxs
            Series([Timestamp("2024-05-18"), Timestamp("2024-06-01")]),  # df_ordered_on
            {
                "period_bounds": date_range(
                    start=Timestamp("2024-04-01"),  # Floor of earliest timestamp
                    end=Timestamp("2024-07-01"),  # Ceil of latest timestamp
                    freq="MS",
                ),
                "oars_mins_maxs": DataFrame(
                    {
                        "mins": [Timestamp("2024-04-01"), Timestamp("2024-05-18")],
                        "maxs": [Timestamp("2024-05-15"), Timestamp("2024-06-01")],
                    },
                ).to_numpy(),
                "likely_on_target": array([False, True]),
            },
        ),
    ],
)
def test_time_period_oars_likely_on_target_size(
    test_id,
    rg_mins,
    rg_maxs,
    df_ordered_on,
    expected,
):
    """
    Test initialization and oars_likely_on_target_size.
    """
    time_period = "MS"
    # Initialize strategy
    strategy = TimePeriodMergeSplitStrategy(
        rg_ordered_on_mins=rg_mins,
        rg_ordered_on_maxs=rg_maxs,
        df_ordered_on=df_ordered_on,
        drop_duplicates=False,
        row_group_time_period=time_period,
    )
    # Test period_bounds
    assert_array_equal(strategy.period_bounds, expected["period_bounds"])
    # Test oars_mins_maxs
    assert_array_equal(strategy.oars_mins_maxs, expected["oars_mins_maxs"])
    # Test likely_on_target_size
    assert_array_equal(strategy.oars_likely_on_target_size, expected["likely_on_target"])


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, time_period, oar_idx_mrs_starts_ends_excl, expected",
    [
        (
            "single_sequence_single_period",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   01/01,   15/01,                 , single period
            #   2,         Jan,                     16/01,   31/01, single period
            array([Timestamp("2024-01-01")]),  # rg_mins
            array([Timestamp("2024-01-15")]),  # rg_maxs
            Series([Timestamp("2024-01-16"), Timestamp("2024-01-31")]),  # df_ordered_on
            "MS",  # monthly periods
            array([[0, 2]]),  # merge region with all OARs
            {
                "oars_merge_sequences": [
                    (0, array([[1, 2]])),  # single sequence
                ],
            },
        ),
        (
            "single_sequence_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   01/01,   15/01,                 , spans Jan
            #   2,         Feb,                     01/02,   15/02, spans Feb
            array([Timestamp("2024-01-01")]),  # rg_mins
            array([Timestamp("2024-01-15")]),  # rg_maxs
            Series([Timestamp("2024-02-01"), Timestamp("2024-02-15")]),  # df_ordered_on
            "MS",  # monthly periods
            array([[0, 2]]),  # merge region with all OARs
            {
                "oars_merge_sequences": [
                    (0, array([[1, 0], [1, 2]])),  # single sequence
                ],
            },
        ),
        (
            "multiple_sequences_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   01/01,   15/01,                 , spans Jan
            #   2,         Feb,                     01/02,   15/02, spans Feb
            #   3,         Mar,   01/03,   15/03,                 , spans Mar
            #   4,         Apr,                     01/04,   15/04, spans Apr
            #   5,         May,   08/05,   12/05,   08/05,   10/05, spans May
            array(
                [Timestamp("2024-01-01"), Timestamp("2024-03-01"), Timestamp("2024-05-08")],
            ),  # rg_mins
            array(
                [Timestamp("2024-01-15"), Timestamp("2024-03-15"), Timestamp("2024-05-12")],
            ),  # rg_maxs
            Series(
                [
                    Timestamp("2024-02-01"),
                    Timestamp("2024-02-15"),
                    Timestamp("2024-04-01"),
                    Timestamp("2024-04-15"),
                    Timestamp("2024-05-08"),
                    Timestamp("2024-05-10"),
                ],
            ),  # df_ordered_on
            "MS",  # monthly periods
            array([[0, 2], [3, 5]]),  # two merge regions
            {
                "oars_merge_sequences": [
                    (0, array([[1, 0], [1, 2]])),  # first sequence
                    (2, array([[2, 4], [3, 6]])),  # second sequence
                ],
            },
        ),
        (
            "rg_dfc_span_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   15/01,                          , spans Jan-Feb
            #   1,         Feb,            15/02,
            #   2,         Mar,                     01/03,        , spans Mar-Apr
            #   2,         Apr,                              01/04
            #   3,         Apr,   02/04,   15/04,   02/04,   15/04, spans Apr
            #   4,         Jun,   01/06,   15/06,                 , spans Jun
            array(
                [Timestamp("2024-01-15"), Timestamp("2024-04-02"), Timestamp("2024-06-01")],
            ),  # rg_mins
            array(
                [Timestamp("2024-02-15"), Timestamp("2024-04-15"), Timestamp("2024-06-15")],
            ),  # rg_maxs
            Series(
                [
                    Timestamp("2024-03-01"),
                    Timestamp("2024-04-01"),
                    Timestamp("2024-04-02"),
                    Timestamp("2024-04-15"),
                ],
            ),  # df_ordered_on
            "MS",  # monthly periods
            array([[0, 3]]),  # merge region with 3 OARs
            {
                "oars_merge_sequences": [
                    (0, array([[1, 0], [1, 2], [2, 3]])),  # single sequence
                ],
            },
        ),
        (
            "dfc_spans_multiple_periods",
            # OAR, Time period, RGs min, RGs max, DFc min, DFc max
            #   1,         Jan,   01/01,   15/01,                 , spans Jan
            #   2,         Feb,                    01/02,         , spans Feb-Mar
            #   2,         Mar,                            15/03,
            array([Timestamp("2024-01-01")]),  # rg_mins
            array([Timestamp("2024-01-15")]),  # rg_maxs
            Series([Timestamp("2024-02-01"), Timestamp("2024-03-15")]),  # df_ordered_on
            "MS",  # monthly periods
            array([[0, 2]]),  # merge region with all OARs
            {
                "oars_merge_sequences": [
                    (0, array([[1, 0], [1, 2]])),  # single sequence
                ],
            },
        ),
    ],
)
def test_time_period_specialized_compute_merge_sequences(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: Series,
    time_period: str,
    oar_idx_mrs_starts_ends_excl: NDArray,
    expected: Dict,
) -> None:
    """
    Test 'specialized_compute_merge_sequences' method.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    rg_mins : NDArray
        Array of minimum values for row groups.
    rg_maxs : NDArray
        Array of maximum values for row groups.
    df_ordered_on : Series
        Series of ordered values.
    oars_desc_dict : Dict[str, NDArray]
        Dictionary containing the oars_desc array.
    time_period : str
        Time period for row groups.
    oar_idx_mrs_starts_ends_excl : NDArray
        Array of shape (n, 2) containing start and end indices (excluded)
        for each merge region to be consolidated.
    expected : Dict
        Dictionary containing expected results.

    """
    # Initialize strategy.
    strategy = TimePeriodMergeSplitStrategy(
        rg_ordered_on_mins=rg_mins,
        rg_ordered_on_maxs=rg_maxs,
        df_ordered_on=df_ordered_on,
        drop_duplicates=False,
        row_group_time_period=time_period,
    )
    # Test specialized_compute_merge_sequences.
    strategy.oar_idx_mrs_starts_ends_excl = oar_idx_mrs_starts_ends_excl
    result = strategy.specialized_compute_merge_sequences()
    # Check
    for (result_rg_start, result_cmpt_ends_excl), (
        expected_rg_start,
        expected_cmpt_ends_excl,
    ) in zip(result, expected["oars_merge_sequences"]):
        assert result_rg_start == expected_rg_start
        assert_array_equal(result_cmpt_ends_excl, expected_cmpt_ends_excl)


@pytest.mark.parametrize(
    "test_id,df_dates,target_period,expected_offsets",
    [
        (
            "monthly_periods",
            Series(
                [
                    Timestamp("2024-01-01"),
                    Timestamp("2024-01-01 12:00"),
                    Timestamp("2024-02-01"),
                    Timestamp("2024-02-02"),
                    Timestamp("2024-02-29"),
                    Timestamp("2024-03-01"),
                ],
            ),
            "MS",  # Month Start
            [0, 2, 5],
        ),
        (
            "monthly_periods_different_start",
            Series(
                [
                    Timestamp("2023-12-31"),
                    Timestamp("2024-01-01 12:00"),
                    Timestamp("2024-02-01"),
                    Timestamp("2024-02-02"),
                    Timestamp("2024-02-29"),
                ],
            ),
            "MS",  # Month Start
            [0, 1, 2],
        ),
        (
            "daily_periods_with_remainder",
            date_range(start="2024-01-01 12:00", periods=2, freq="D"),  # 2 days starting at noon
            "D",  # Daily
            [0, 1],  # End of first day
        ),
        (
            "single_day",
            Series([Timestamp("2024-01-01")]),  # Single day
            "D",  # Daily
            [0],  # No splits needed
        ),
        (
            "hourly_periods",
            date_range(start="2024-01-01", periods=3, freq="h"),  # 3 hours
            "h",  # Hourly
            [0, 1, 2],  # End of first and second hour
        ),
        (
            "sparse_df_ordered_on",
            Series([Timestamp("2024-01-01 12:00"), Timestamp("2024-06-01 12:00")]),
            "D",  # Daily
            [0, 1],  # End of first day
        ),
    ],
)
def test_time_period_row_group_offsets(test_id, df_dates, target_period, expected_offsets):
    # Initialize strategy
    strategy = TimePeriodMergeSplitStrategy(
        rg_ordered_on_mins=Series([Timestamp("2024/01/01 04:00:00")]),  # dummy value
        rg_ordered_on_maxs=Series([Timestamp("2024/01/05 14:00:00")]),  # dummy value
        df_ordered_on=Series(Timestamp("2024/01/06 04:00:00")),  # dummy value
        row_group_time_period=target_period,
    )
    # Get offsets
    offsets = strategy.row_group_offsets(df_ordered_on=Series(df_dates))
    # Verify results
    assert offsets == expected_offsets


@pytest.mark.parametrize(
    (
        "test_id, df_data, pf_data, row_group_offsets, row_group_size, drop_duplicates, max_n_off_target_rgs, expected"
    ),
    [
        # 1/ Adding data at complete tail, testing 'drop_duplicates'.
        # 'max_n_off_target_rgs' is never triggered.
        (
            # Max row group size as freqstr.
            # Writing after pf data, no off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [12h10]
            "new_rg_simple_append_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}12:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data, not merging with off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [10h10]
            "no_drop_duplicates_simple_append_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data, merging with off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:              [10h10]
            "drop_duplicates_merge_tail_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size | should not merge irg
            True,  # drop_duplicates | should merge with irg
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values on boundary.
            # Writing after pf data, not merging with off target size row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:                      [10h00]
            "no_drop_duplicates_simple_append_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}08:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size | should not merge irg
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # Values on boundary.
            # Writing after pf data, merging with off target size row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:              [10h00]
            "drop_duplicates_merge_tail_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size | should not merge irg
            True,  # drop_duplicates | should merge with irg
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr.
            # Writing after pf data, off target size row group should be merged.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [13h00]
            "last_row_group_exceeded_merge_tail_timestamp",
            [Timestamp(f"{REF_D}13:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size | new period, should merge irgs
            True,  # drop_duplicates | no duplicates to drop
            3,  # max_n_off_target_rgs | should not rewrite irg
            {
                "merge_plan": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 2/ Adding data right at the start.
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
            "2h",  # row_group_size | no rg in same period to merge with
            True,  # no duplicates to drop
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                "merge_plan": [1],
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
            "2h",  # row_group_size | should merge with rg in same period
            True,  # no duplicates to drop
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                "merge_plan": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 3/ Adding data at complete end, testing 'max_n_off_target_rgs'.
        (
            # Max row group size as freqstr.
            # df connected to off target size rgs.
            # Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_not_reached_simple_append_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            3,  # max_n_off_targetrgs | should not rewrite tail
            {
                "merge_plan": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Max row group size as freqstr.
            # df connected to off target size rgs.
            # Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_reached_tail_rewrite_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size | should not merge with irg.
            False,  # drop_duplicates | should not merge with preceding rg
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                "merge_plan": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Max row group size as freqstr
            # df connected to off target size rgs.
            # Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_none_simple_append_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size
            False,  # drop_duplicates
            None,  # max_n_off_target_rgs
            {
                "merge_plan": [1],
                "sort_rgs_after_write": False,
            },
        ),
        # 4/ Adding data just before last off target size row groups.
        (
            # Test  13 (3.c) /
            # Max row group size as int | df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups at
            # the end of pf data.
            # 'max_n_off_target_rgs' reached to rewrite all tail.
            # row grps:  0            1                 2
            # pf: [8h00,9h00], [10h00],          [13h00]
            # df:                       [12h00]
            "insert_timestamp_max_n_off_target_rgs_tail_rewrite",
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
            "2h",  # row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_off_target_rgs | should rewrite tail
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  14 (3.d) /
            # Max row group size as int | df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups at
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
            "2h",  # row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_off_target_rgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  15 (3.e) /
            # Max row group size as int | df connected to off target size rgs.
            # Writing at end of pf data, with off target size row groups at
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
            "2h",  # row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_n_off_target_rgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        # 5/ Adding data in the middle of pf data.
        (
            # Test  19 (4.d) /
            # Max row group size as pandas freqstr | df within pf data.
            # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # row grps:  0        1           2           3      4
            # pf: [8h,9h], [10h, 11h], [12h, 13h], [14h], [15h]
            # df:               [11h]
            "insert_timestamp_middle_with_off_target_rgs",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame({"ordered_on": date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=8)}),
            [0, 2, 4, 6, 7],
            "2h",  # row_group_size
            True,  # drop_duplicates
            2,  # max_n_off_target_rgs | should rewrite tail
            (1, 2, True),
        ),
        (
            # Test  20 (4.e) /
            # Max row group size as pandas freqstr | df within pf data.
            # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # row grps:  0          1           2           3      4
            # pf: [8h,9h],   [10h, 11h], [12h, 13h], [14h], [15h]
            # df:        [9h]
            "insert_timestamp_middle_no_rewrite",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}9:00")]}),
            DataFrame({"ordered_on": date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=8)}),
            [0, 2, 4, 6, 7],
            "2h",  # row_group_size
            False,  # drop_duplicates
            2,  # max_n_off_target_rgs
            (None, None, True),
        ),
        # Do "island" cases
        # pf:         [0, 1]                [7, 9]                [ 15, 16]
        # df1                       [4, 5            11, 12]  # should have df_head, df_tail, no merge?
        # df2                                [ 8, 11, 12]     # should have merge + df_tail?
        # df3                       [4, 5      8]             # should have df_head + merge?
        # df4                       [4, 5]   # here df not to be merged with following row group
        # df5                       [4, 5, 6]   # here, should be merged
        # df6                                         + same with row_group_size as str
        # test with freqstr, with several empty periods, to make sure the empty periods are
        # not in the output
    ],
)
def test_compute_ordered_merge_plan(
    test_id,
    df_data,
    pf_data,
    row_group_offsets,
    row_group_size,
    drop_duplicates,
    max_n_off_target_rgs,
    expected,
    tmp_path,
):
    df = DataFrame({"ordered_on": df_data})
    pf_data = DataFrame({"ordered_on": pf_data})
    pf = create_parquet_file(tmp_path, pf_data, row_group_offsets=row_group_offsets)
    pf_statistics = pf.statistics
    rg_ordered_on_mins = array(pf_statistics[MIN]["ordered_on"])
    rg_ordered_on_maxs = array(pf_statistics[MAX]["ordered_on"])
    df_ordered_on = df.loc[:, "ordered_on"]
    if isinstance(row_group_size, str):
        split_strat = TimePeriodMergeSplitStrategy(
            rg_ordered_on_mins=rg_ordered_on_mins,
            rg_ordered_on_maxs=rg_ordered_on_maxs,
            df_ordered_on=df_ordered_on,
            row_group_time_period=row_group_size,
        )
    else:
        split_strat = NRowsMergeSplitStrategy(
            rg_ordered_on_mins=rg_ordered_on_mins,
            rg_ordered_on_maxs=rg_ordered_on_maxs,
            df_ordered_on=df_ordered_on,
            rgs_n_rows=[rg.num_rows for rg in pf.row_groups],
            row_group_target_size=row_group_size,
            drop_duplicates=drop_duplicates,
        )
    split_strat.compute_merge_regions_start_ends_excl(
        max_n_off_target_rgs=max_n_off_target_rgs,
    )
    merge_plan = split_strat.partition_merge_regions()

    print("merge_plan")
    print(merge_plan)
    print("len(pf)")
    print(len(pf))
    for ms_idx, expected_merge_sequence in enumerate(expected["merge_plan"]):
        assert merge_plan[ms_idx][0] == expected_merge_sequence[0]
        assert array_equal(merge_plan[ms_idx][1], expected_merge_sequence[1])
    sort_rgs_after_write = len(merge_plan) > 1 or merge_plan[0][0] < len(pf)
    assert sort_rgs_after_write == expected["sort_rgs_after_write"]
