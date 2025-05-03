#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import List

import pytest
from numpy import array
from numpy import array_equal
from numpy import bool_
from numpy import cumsum
from numpy import empty
from numpy import int_
from numpy import zeros
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import Series
from pandas import Timestamp

from oups.store.write.merge_split_strategies.base import OARMergeSplitStrategy
from oups.store.write.merge_split_strategies.base import get_region_indices_of_true_values
from oups.store.write.merge_split_strategies.base import get_region_start_end_delta
from oups.store.write.merge_split_strategies.base import set_true_in_regions


RG_IDX_START = "rg_idx_start"
RG_IDX_END_EXCL = "rg_idx_end_excl"
DF_IDX_END_EXCL = "df_idx_end_excl"
HAS_ROW_GROUP = "has_row_group"
HAS_DF_OVERLAP = "has_df_overlap"
RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS = "rg_idx_ends_excl_not_to_use_as_split_points"


class ConcreteOARMergeSplitStrategy(OARMergeSplitStrategy):
    """
    Concrete implementation for testing purposes.
    """

    def _specialized_init(self, **kwargs):
        raise NotImplementedError("Test implementation only")

    def _specialized_compute_merge_sequences(self):
        raise NotImplementedError("Test implementation only")

    def compute_split_sequence(self, df_ordered_on: Series) -> List[int]:
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
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([2]),
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
def test_OARMergeSplitStrategy_init(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: Series,
    drop_duplicates: bool,
    expected: NDArray,
) -> None:
    """
    Test OARMergeSplitStrategy with various scenarios.
    """
    oars_prop = ConcreteOARMergeSplitStrategy(
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
def test_OARMergeSplitStrategy_validation(
    test_id: str,
    rg_ordered_on_mins: array,
    rg_ordered_on_maxs: array,
    df_ordered_on: Series,
    expected_error: Exception,
) -> None:
    """
    Test input validation in OARMergeSplitStrategy.

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
        ConcreteOARMergeSplitStrategy(
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
            array([[0, 4], [7, 8]]),  # Two regions: [0-1) and [2-4)
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
            array([[0, 2], [4, 5]]),
        ),
        (  # 'max_n_off_target' is None
            "max_n_off_target_is_none",
            array([True, False, False, False, True]),  # DataFrame chunks at 0, 4
            array([True, False, True, False, False]),  # Likely on target
            None,  # max_n_off_target_rgs
            array([[0, 1], [4, 5]]),
        ),
        (  # 'max_n_off_target' is 1
            "max_n_off_target_is_1",
            array([True, False, False, False, True]),  # DataFrame chunks at 0, 4
            array([False, False, True, False, False]),  # Likely on target
            1,  # max_n_off_target_rgs
            array([[0, 2], [3, 5]]),
        ),
        (  # 'max_n_off_target' is 1
            "max_n_off_target_reached_where_no_df_chunk_overlap",
            array([True, False, False, False, False, True]),  # DataFrame chunks at 0, 4
            array([False, True, False, False, True, False]),  # Likely on target
            1,  # max_n_off_target_rgs
            array([[0, 1], [5, 6]]),
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
    split_strat = ConcreteOARMergeSplitStrategy(
        rg_ordered_on_mins=zeros(1, dtype=int_),
        rg_ordered_on_maxs=zeros(1, dtype=int_),
        df_ordered_on=Series([1]),
        drop_duplicates=True,
    )
    split_strat.oars_has_df_overlap = oars_has_df_overlap
    split_strat.oars_likely_on_target_size = oars_likely_on_target_size
    split_strat.n_oars = len(oars_has_df_overlap)
    split_strat._compute_merge_regions_start_ends_excl(
        max_n_off_target_rgs=max_n_off_target_rgs,
    )
    assert array_equal(split_strat.oar_idx_mrs_starts_ends_excl, expected)
