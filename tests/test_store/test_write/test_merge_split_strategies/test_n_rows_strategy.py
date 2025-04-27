#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import Dict

import pytest
from numpy import array
from numpy import column_stack
from numpy import diff
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import Series

from oups.store.write.merge_split_strategies import NRowsMergeSplitStrategy


RG_IDX_START = "rg_idx_start"
RG_IDX_END_EXCL = "rg_idx_end_excl"
DF_IDX_END_EXCL = "df_idx_end_excl"
HAS_ROW_GROUP = "has_row_group"
RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS = "rg_idx_ends_excl_not_to_use_as_split_points"


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
        (  # This test case check that last row group in sequence is correctly
            # added to merge region.
            "single_sequence_encompassing_all_oars_wo_drop_duplicates",
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
    result = strategy._specialized_compute_merge_sequences()
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
        (  # Islands case 1.
            # rg:  0         1          2           3
            # pf: [0,xx,6], [6,xx,10], [11,xx,12], [13]
            # df:   [3,4,                           13,14]
            "islands_case_with_drop_duplicates_1",
            array([0, 6, 11, 13]),  # rg_mins
            array([6, 10, 12, 13]),  # rg_maxs
            Series([3, 4, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            True,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 4, 1],  # rgs_n_rows
            1,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[1, 2]])), (3, array([[4, 4]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # Islands case 2.
            # rg:  0         1          2           3
            # pf: [0,xx,6], [6,xx,10], [11,xx,12], [13]
            # df:    [3,     6,                        13, 14]
            "islands_case_with_drop_duplicates_2",
            array([0, 6, 11, 13]),  # rg_mins
            array([6, 10, 12, 13]),  # rg_maxs
            Series([3, 6, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            True,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 4, 1],  # rgs_n_rows
            1,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([1]),
                "oars_merge_sequences": [(0, array([[2, 2]])), (3, array([[4, 4]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # Islands case 3.
            # Need to load rgs 0 & 1 at same time to sort data.
            # rg:  0         1          2           3
            # pf: [0,xx,6], [6,xx,10], [11,xx,12], [13]
            # df:    [3,      6,                       13, 14]
            "islands_case_no_drop_duplicates_3",
            array([0, 6, 11, 13]),  # rg_mins
            array([6, 10, 12, 13]),  # rg_maxs
            Series([3, 6, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 4, 1],  # rgs_n_rows
            1,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([1]),
                "oars_merge_sequences": [(0, array([[2, 2]])), (3, array([[4, 4]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # Islands case 4.
            # Need to load rg 1 only, df data coming after.
            # rg:  0         1          2           3
            # pf: [0,xx,6], [6,xx,10], [11,xx,12], [13]
            # df:            [6,7,                     13, 14]
            "islands_case_no_drop_duplicates_4",
            array([0, 6, 11, 13]),  # rg_mins
            array([6, 10, 12, 13]),  # rg_maxs
            Series([6, 7, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            [4, 4, 4, 1],  # rgs_n_rows
            1,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([1]),
                "oars_merge_sequences": [(1, array([[2, 2]])), (3, array([[4, 4]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # rg:  0     1     2     3     4     5     6     7
            # pf: [0,1],[1,1],[2,3],[4,5],[5,6],[6,7],[8,9],[10,11]
            # df:           [1                             9,      13,14]
            "many_only_off_target_rgs_no_drop_duplicates",
            array([0, 1, 2, 4, 5, 6, 8, 10]),  # rg_mins
            array([1, 1, 3, 5, 6, 7, 9, 11]),  # rg_maxs
            Series([1, 9, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should rewrite all rgs
            False,  # drop_duplicates | should not merge with preceding rg
            [2, 2, 2, 2, 2, 2, 2, 2],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite all rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[2, 0], [4, 1], [6, 1], [7, 2], [8, 4]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # rg:  0     1     2     3     4     5     6     7
            # pf: [0,1],[1,1],[2,3],[4,5],[5,6],[6,7],[8,9],[10,11]
            # df: [0,      1,                            9,       13,14]
            "many_only_off_target_rgs_with_drop_duplicates",
            array([0, 1, 2, 4, 5, 6, 8, 10]),  # rg_mins
            array([1, 1, 3, 5, 6, 7, 9, 11]),  # rg_maxs
            Series([1, 9, 13, 14]),  # df_ordered_on
            4,  # row_group_size | should rewrite all rgs
            True,  # drop_duplicates | should not merge with preceding rg
            [2, 2, 2, 2, 2, 2, 2, 2],  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite all rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: array([1]),
                "oars_merge_sequences": [(0, array([[2, 1], [4, 1], [6, 1], [8, 4]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Writing after pf data, no off target size row group.
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
        (  # Writing at end of pf data, merging with off target size row group.
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
        (  # df within pf data.
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
        (  # Writing after pf data, off target size row group to merge.
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
        (  # df at the start of pf data.
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
        (  # df at the start of pf data.
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
        (  # df connected to off target size rgs.
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
        (  # df within pf data.
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
        (  # df within pf data.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
        (  # df connected to off target size rgs.
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
def test_nrows_compute_split_sequence(df_size, target_size, expected_offsets):
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
    offsets = strategy.compute_split_sequence(df_ordered_on=Series(range(df_size)))
    # Verify results
    assert offsets == expected_offsets
