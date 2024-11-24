#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
import pytest
from numpy import array_equal
from pandas import DataFrame
from pandas import Timestamp
from pandas import date_range

from oups.store.ordered_merge_info import OrderedMergeInfo
from tests.test_store.conftest import create_parquet_file


REF_D = "2020/01/01 "


@pytest.mark.parametrize(
    (
        "test_id, df_data, pf_data, row_group_offsets, row_group_size_target, drop_duplicates, max_n_irgs, expected"
    ),
    [
        # 1/ Adding data at complete tail, testing 'drop_duplicates'.
        # 'max_n_irgs' is never triggered.
        (
            # Test  0 (0.a.a) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing after pf data, no incomplete row groups.
            # rg:  0      1
            # pf: [0,1], [2,3]
            # df:             [3]
            "no_drop_duplicates_simple_append",
            [3],
            [0, 1, 2, 3],
            [0, 2],  # row_group_offsets
            2,  # row_group_size_target
            False,  # drop_duplicates
            2,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": False,
                "has_df_tail": True,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": False,
            },
        ),
        (
            # Test  1 (0.a.b) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at the end
            # of pf data.
            # rg:  0        1        2
            # pf: [0,1,2], [6,7,8], [9]
            # df:                   [9]
            "drop_duplicates_merge_tail",
            [9],
            [0, 1, 2, 6, 7, 8, 9],
            [0, 3, 6],  # row_group_offsets
            3,  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 2,
                "rg_idx_merge_end_excl": 3,
                "df_idx_tmrg_starts": [0],
                "df_idx_tmrg_ends_excl": [1],
                "tmrg_n_rows": [1],
                "n_tmrgs": 1,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  2 (0.b.a) /
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                     [10h10]
            "no_drop_duplicates_simple_append_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size_target | not triggered
            False,  # drop_duplicates
            3,  # max_n_irgs | not triggered
            {
                "has_df_head": False,
                "has_overlap": False,
                "has_df_tail": True,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": False,
            },
        ),
        (
            # Test  3 (0.b.b) /
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:              [10h10]
            "drop_duplicates_merge_tail_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}10:10")],
            date_range(Timestamp(f"{REF_D}08:10"), freq="1h", periods=3),
            [0, 2],
            "2h",  # row_group_size_target | not triggered
            True,  # drop_duplicates
            3,  # max_n_irgs | not triggered
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 1,
                "rg_idx_merge_end_excl": 2,
                "df_idx_tmrg_starts": [0],
                "df_idx_tmrg_ends_excl": [1],
                "tmrg_n_rows": [1],
                "n_tmrgs": 1,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  4 (0.b.c) /
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:                     [10h00]
            "no_drop_duplicates_simple_append_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}08:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            3,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": False,
                "has_df_tail": True,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": False,
            },
        ),
        (
            # Test  5 (0.b.d) /
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:              [10h00]
            "drop_duplicates_merge_tail_timestamp_on_boundary",
            [Timestamp(f"{REF_D}10:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=3),
            [0, 2],  # row_group_offsets
            "2h",  # row_group_size_target
            True,  # drop_duplicates
            3,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 1,
                "rg_idx_merge_end_excl": 2,
                "df_idx_tmrg_starts": [0],
                "df_idx_tmrg_ends_excl": [1],
                "tmrg_n_rows": [1],
                "n_tmrgs": 1,
                "sort_rgs_after_rewrite": True,
            },
        ),
        # 2/ Adding data right at the start.
        (
            # Test  21 (5.a) /
            # Max row group size as int.
            # df at the start of pf data.
            # rg:       0       1       2    3
            # pf:      [2, 6], [7, 8], [9], [10]
            # df: [0,1]
            "drop_duplicates_insert_at_start_new_rg",
            [0, 1],
            [2, 6, 7, 8, 9, 10],
            [0, 2, 4, 5],  # row_group_offsets
            2,  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | not triggered
            {
                "has_df_head": True,
                "has_overlap": False,
                "has_df_tail": False,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  21 (5.a) /
            # Max row group size as int.
            # df at the start of pf data.
            # rg:       0       1       2    3
            # pf:      [2, 6], [7, 8], [9], [10]
            # df: [0]
            "drop_duplicates_insert_at_start_no_new_rg",
            [0],
            [2, 6, 7, 8, 9, 10],
            [0, 2, 4, 5],  # row_group_offsets
            2,  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | not triggered
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 0,
                "rg_idx_merge_end_excl": 1,
                "df_idx_tmrg_starts": [0],
                "df_idx_tmrg_ends_excl": [1],
                "tmrg_n_rows": [2],
                "n_tmrgs": 1,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  22 (5.b) /
            # Max row group size as freqstr.
            # df at the very start.
            # df is not overlapping with existing row groups.
            # rg:            0            1        2
            # pf:           [8h00,9h00], [12h00], [13h00]
            # df:  [7h30]
            "drop_duplicates_insert_at_start_new_rg_timestamp_not_on_boundary",
            [Timestamp(f"{REF_D}7:30")],
            [
                Timestamp(f"{REF_D}08:00"),
                Timestamp(f"{REF_D}09:00"),
                Timestamp(f"{REF_D}12:00"),
                Timestamp(f"{REF_D}14:00"),
            ],
            [0, 2, 3],
            "2h",  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            {
                "has_df_head": True,
                "has_overlap": False,
                "has_df_tail": False,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  22 (5.b) /
            # Max row group size as freqstr.
            # df at the very start.
            # df is overlapping with existing row groups.
            # rg:            0            1        2
            # pf:           [8h10,9h10], [12h10], [13h10]
            # df:           [8h00]
            "drop_duplicates_insert_at_start_no_new_rg_timestamp_on_boundary",
            [Timestamp(f"{REF_D}8:00")],
            [
                Timestamp(f"{REF_D}08:10"),
                Timestamp(f"{REF_D}09:10"),
                Timestamp(f"{REF_D}12:10"),
                Timestamp(f"{REF_D}14:10"),
            ],
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 0,
                "rg_idx_merge_end_excl": 1,
                "df_idx_tmrg_starts": [0],
                "df_idx_tmrg_ends_excl": [1],
                "tmrg_n_rows": [2],
                "n_tmrgs": 1,
                "sort_rgs_after_rewrite": True,
            },
        ),
        # 3/ Adding data at complete tail, testing 'max_n_irgs'.
        (
            # Test  5 (1.a) /
            # Max row group size as int
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                [12]
            "max_n_irgs_not_reached_simple_append_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target
            False,  # drop_duplicates
            3,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": False,
                "has_df_tail": True,
                "df_idx_merge_start": None,
                "df_idx_merge_end_excl": None,
                "rg_idx_merge_start": None,
                "rg_idx_merge_end_excl": None,
                "df_idx_tmrg_starts": [],
                "df_idx_tmrg_ends_excl": [],
                "tmrg_n_rows": [],
                "n_tmrgs": 0,
                "sort_rgs_after_rewrite": False,
            },
        ),
        (
            # Test  5 (1.a) /
            # Max row group size as int
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                [12]
            "max_n_irgs_reached_rewrite_tail_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target
            False,  # drop_duplicates
            2,  # max_n_irgs
            {
                "has_df_head": False,
                "has_overlap": True,
                "has_df_tail": False,
                "df_idx_merge_start": 0,
                "df_idx_merge_end_excl": 1,
                "rg_idx_merge_start": 2,
                "rg_idx_merge_end_excl": 4,
                "df_idx_tmrg_starts": [0, 0],
                "df_idx_tmrg_ends_excl": [0, 1],
                "tmrg_n_rows": [1, 1],
                "n_tmrgs": 2,
                "sort_rgs_after_rewrite": True,
            },
        ),
        (
            # Test  6 (1.b) /
            # Max row group size as freqstr | df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # row grps:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                       [11h00]
            "append_tail_max_n_irgs_exceeded_timestamp",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=4,
                    ),
                },
            ),
            [0, 2, 3],
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            3,  # max_n_irgs
            (1, None, False),  # bool: need to sort rgs after write
        ),
        # 3/ Adding data at complete tail, testing 'max_n_irgs=None'.
        (
            # Test  7 (2.a) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # row grps:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                [12]
            "append_tail_max_n_irgs_none_int",
            DataFrame({"ordered_on": [12]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9, 10, 11, 12]}),
            [0, 4, 8, 9],
            4,  # row_group_size_target
            False,  # drop_duplicates
            None,  # max_n_irgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  8 (2.b) /
            # Max row group size as freqstr | df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # row grps:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                       [11h00]
            "append_tail_max_n_irgs_none_timestamp",
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=4,
                    ),
                },
            ),
            [0, 2, 3],
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            None,  # max_n_irgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        # 4/ Adding data just before last incomplete row groups.
        (
            # Test  9 (3.a) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Enough rows to rewrite all tail.
            # row grps:  0        1              2
            # pf: [0,1,2], [6,7,8],       [10]
            # df:                  [8, 9]
            "insert_before_incomplete_rg_rewrite_tail",
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # row_group_size_target | should rewrite tail
            False,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  10 (3.b) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Tail is rewritten because with df, 'row_group_size_target' is
            # reached.
            # row grps:  0        1              2
            # pf: [0,1,2], [6,7,8],       [10]
            # df:                  [8, 9]
            "insert_before_incomplete_rg_max_size_reached",
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # row_group_size_target | should rewrite tail
            True,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            (1, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  11 (3.c) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Tail is rewritten because with df, 'max_n_irgs' is reached.
            # row grps:  0       1              2
            # pf: [0,1,2],[6,7,8],       [10]
            # df:                 [8]
            "insert_before_incomplete_rg_max_n_irgs_reached",
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # row_group_size_target | should not rewrite tail
            True,  # drop_duplicates
            2,  # max_n_irgs | should rewrite tail
            (1, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  12 (3.d) /
            # Max row group size as int | df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Should not rewrite all tail.
            # row grps:  0         1                2
            # pf: [0,1,2,3],[6,7,8,8],       [10]
            # df:                     [8, 9]
            "insert_before_incomplete_rg_no_rewrite",
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 3, 6, 7, 8, 8, 10]}),
            [0, 4, 8],
            4,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
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
            "insert_timestamp_max_n_irgs_rewrite_tail",
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
    ],
)
def test_ordered_merge_info(
    test_id,
    df_data,
    pf_data,
    row_group_offsets,
    row_group_size_target,
    drop_duplicates,
    max_n_irgs,
    expected,
    create_parquet_file,
):
    df = DataFrame({"ordered": df_data})
    pf_data = DataFrame({"ordered": pf_data})
    pf = create_parquet_file(pf_data, row_group_offsets=row_group_offsets)
    merge_info = OrderedMergeInfo.analyze(
        df=df,
        pf=pf,
        ordered_on="ordered",
        row_group_size_target=row_group_size_target,
        drop_duplicates=drop_duplicates,
        max_n_irgs=max_n_irgs,
    )

    assert merge_info.has_df_head == expected["has_df_head"]
    assert merge_info.has_overlap == expected["has_overlap"]
    assert merge_info.has_df_tail == expected["has_df_tail"]
    assert merge_info.df_idx_merge_start == expected["df_idx_merge_start"]
    assert merge_info.df_idx_merge_end_excl == expected["df_idx_merge_end_excl"]
    assert merge_info.rg_idx_merge_start == expected["rg_idx_merge_start"]
    assert merge_info.rg_idx_merge_end_excl == expected["rg_idx_merge_end_excl"]
    assert array_equal(merge_info.df_idx_tmrg_starts, expected["df_idx_tmrg_starts"])
    assert array_equal(merge_info.df_idx_tmrg_ends_excl, expected["df_idx_tmrg_ends_excl"])
    assert merge_info.tmrg_n_rows == expected["tmrg_n_rows"]
    assert merge_info.n_tmrgs == expected["n_tmrgs"]
    assert merge_info.sort_rgs_after_rewrite == expected["sort_rgs_after_rewrite"]


"""
        (
            "pf_leading_df_with_df_tail",
            [4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6],
            {
                "has_pf_head": True,
                "has_df_head": False,
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": True,
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
                "has_overlap": False,
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
                "has_overlap": False,
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
