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

from oups.store.ordered_merge_info import analyze_chunks_to_merge
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
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing after pf data, no incomplete row groups.
            # rg:  0      1
            # pf: [0,1], [2,3]
            # df:               [3]
            "no_drop_duplicates_simple_append_int",
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
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at the end
            # of pf data.
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
            # df connected to incomplete rgs.
            # Values not on boundary to check 'floor()'.
            # Writing after pf data.
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
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
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
            "2h",  # row_group_size_target | should not merge irg
            True,  # drop_duplicates | should merge with irg
            3,  # max_n_irgs | should not rewrite irg
            {
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 2/ Adding data right at the start.
        (
            # Test  21 (5.a) /
            # Max row group size as int.
            # df at the start of pf data.
            # rg:         0       1       2    3
            # pf:        [2, 6], [7, 8], [9], [10]
            # df: [0,1]
            "drop_duplicates_insert_at_start_new_rg",
            [0, 1],
            [2, 6, 7, 8, 9, 10],
            [0, 2, 4, 5],  # row_group_offsets
            2,  # row_group_size_target
            True,  # drop_duplicates
            2,  # max_n_irgs | not triggered
            {
                "chunk_counter": [2],
                "sort_rgs_after_write": True,
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
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  22 (5.b) /
            # Max row group size as freqstr.
            # df at the very start.
            # df is not overlapping with existing row groups.
            # rg:           0            1        2
            # pf:          [8h00,9h00], [12h00], [13h00]
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
                "chunk_counter": [1],
                "sort_rgs_after_write": True,
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
                "chunk_counter": [0, 1],
                "sort_rgs_after_write": True,
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
            # df:                                   [12]
            "max_n_irgs_not_reached_simple_append_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target
            False,  # drop_duplicates
            3,  # max_n_irgs
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
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
            # df:                                   [12]
            "max_n_irgs_reached_rewrite_tail_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],  # row_group_offsets
            4,  # row_group_size_target
            False,  # drop_duplicates
            2,  # max_n_irgs
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  6 (1.b) /
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
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            3,  # max_n_irgs
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Test  6 (1.b) /
            # Max row group size as freqstr.
            # df connected to incomplete rgs.
            # Values on boundary.
            # Writing after pf data, incomplete row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_irgs_reached_rewrite_tail_timestamp",
            [Timestamp(f"{REF_D}11:00")],
            date_range(Timestamp(f"{REF_D}8:00"), freq="1h", periods=4),
            [0, 2, 3],  # row_group_offsets
            "2h",  # row_group_size_target
            False,  # drop_duplicates
            2,  # max_n_irgs
            {
                "chunk_counter": [0, 0, 0, 1],
                "sort_rgs_after_write": True,
            },
        ),
        # 3/ Adding data at complete tail, testing 'max_n_irgs=None'.
        (
            # Test  7 (2.a) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # rg:  0          1           2     3
            # pf: [0,1,2,6], [7,8,9,10], [11], [12]
            # df:                                   [12]
            "max_n_irgs_none_simple_append_int",
            [12],
            [0, 1, 2, 6, 7, 8, 9, 10, 11, 12],
            [0, 4, 8, 9],
            4,  # row_group_size_target
            False,  # drop_duplicates
            None,  # max_n_irgs
            {
                "chunk_counter": [1],
                "sort_rgs_after_write": False,
            },
        ),
        (
            # Test  8 (2.b) /
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
            # Test  9 (3.a) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Enough rows to rewrite all tail.
            # rg:  0        1                    2
            # pf: [0,1,2], [6,7,8],             [11]
            # df:                   [8, 9, 10]
            "insert_before_irgs_simple_append_int",
            [8, 9, 10],
            [0, 1, 2, 6, 7, 8, 11],
            [0, 3, 6],
            3,  # row_group_size_target | should not rewrite tail
            False,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [3],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  9 (3.a) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Writing at end of pf data, with incomplete row groups at
            # the end of pf data.
            # Enough rows to rewrite all tail.
            # rg:  0        1                2
            # pf: [0,1,2], [6,7,8],         [11]
            # df:                   [8, 9]
            "insert_before_irgs_rewrite_tail_int",
            [8, 9],
            [0, 1, 2, 6, 7, 8, 11],
            [0, 3, 6],
            3,  # row_group_size_target | should rewrite tail
            False,  # drop_duplicates
            3,  # max_n_irgs | should not rewrite tail
            {
                "chunk_counter": [0, 2],
                "sort_rgs_after_write": True,
            },
        ),
        (
            # Test  10 (3.b) /
            # Max row group size as int.
            # df connected to incomplete rgs.
            # Incomplete row groups at the end of pf data.
            # rg:  0        1           2
            # pf: [0,1,2], [6,7,8],    [10]
            # df:              [8, 9]
            "insert_before_irgs_drop_duplicates_rewrite_tail_int",
            [8, 9],
            [0, 1, 2, 6, 7, 8, 10],
            [0, 3, 6],
            3,  # row_group_size_target | should rewrite tail
            True,  # drop_duplicates | merge with preceding rg
            3,  # max_n_irgs | should not rewrite tail
            {
                # Other acceptable solution:
                # [0, 1, 1, 2]
                "chunk_counter": [0, 1, 2, 2],
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
            "insert_before_irgs_drop_duplicates_rewrite_tail_int",
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
            "insert_before_irgs_rewrite_tail_int",
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
        # Do "island" cases
        # pf:         [0, 1]                [7, 9]                [ 15, 16]
        # df1                       [4, 5            11, 12]  # should have df_head, df_tail, no merge?
        # df2                                [ 8, 11, 12]     # should have merge + df_tail?
        # df3                       [4, 5      8]             # should have df_head + merge?
        # df4                       [4, 5]   # here df not to be merged with following row group
        # df5                       [4, 5, 6]   # here, should be merged
        # df6                                         + same with row_group_size_target as str
    ],
)
def test_analyze_chunks_to_merge(
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
    chunk_counter, sort_rgs_after_write = analyze_chunks_to_merge(
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
