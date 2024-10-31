#!/usr/bin/env python3
"""
Created on Sat Dec 18 15:00:00 2021.

@author: yoh

Test utils.
- Initialize path:
tmp_path = os_path.expanduser('~/Documents/code/data/oups')

"""
from os import path as os_path

import pytest
from fastparquet import ParquetFile
from fastparquet import write as fp_write
from numpy import arange
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Timestamp
from pandas import concat
from pandas import date_range

from oups.store.writer import _indexes_of_overlapping_rrgs
from oups.store.writer import to_midx
from oups.store.writer import write_ordered as pswrite


REF_D = "2020/01/01 "


@pytest.mark.parametrize(
    (
        "new_data, df_to_record, row_group_offsets, max_row_group_size, drop_duplicates, max_nirgs, expected"
    ),
    [
        # 1/ Adding data at complete tail, testing 'drop_duplicates'.
        # 'max_nirgs' is never triggered.
        (
            # Test  0 (0.a.a) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing after recorded data, no incomplete row groups.
            # row grps:  0      1
            # recorded: [0,1], [2,3]
            # new data:             [3]
            DataFrame({"ordered_on": [3]}),
            DataFrame({"ordered_on": [0, 1, 2, 3]}),
            [0, 2],
            2,  # max_row_group_size
            False,  # drop_duplicates
            2,  # max_nirgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  1 (0.a.b) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # row grps:  0        1        2
            # recorded: [0,1,2], [6,7,8], [9]
            # new data:                   [9]
            DataFrame({"ordered_on": [9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9]}),
            [0, 3, 6],
            3,  # max_row_group_size
            True,  # drop_duplicates
            2,  # max_nirgs
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  2 (0.b.a) /
            # Max row group size as freqstr | New data connected to incomplete rgs.
            # Values on boundary to check 'floor()'.
            # Writing after recorded data.
            # row grps:  0            1
            # recorded: [8h10,9h10], [10h10]
            # new data:                     [10h10]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}10:10")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:10"),
                        freq="1h",
                        periods=3,
                    ),
                },
            ),
            [0, 2],
            "2h",  # max_row_group_size | not triggered
            False,  # drop_duplicates
            3,  # max_nirgs | not triggered
            (None, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  3 (0.b.b) /
            # Max row group size as freqstr | New data connected to incomplete rgs.
            # Values on boundary.
            # Writing after recorded data, incomplete row groups.
            # row grps:  0            1
            # recorded: [8h00,9h00], [10h00]
            # new data:                     [10h00]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}10:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=3,
                    ),
                },
            ),
            [0, 2],
            "2h",  # max_row_group_size
            False,  # drop_duplicates
            3,  # max_nirgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  4 (0.b.c) /
            # Max row group size as freqstr | New data connected to incomplete rgs.
            # Values on boundary.
            # Writing after recorded data, incomplete row groups.
            # row grps:  0            1
            # recorded: [8h00,9h00], [10h00]
            # new data:              [10h00]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}10:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=3,
                    ),
                },
            ),
            [0, 2],
            "2h",  # max_row_group_size
            True,  # drop_duplicates
            3,  # max_nirgs
            (1, None, False),  # bool: need to sort rgs after write
        ),
        # 2/ Adding data at complete tail, testing 'max_nirgs'.
        (
            # Test  5 (1.a) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # row grps:  0          1           2     3
            # recorded: [0,1,2,6], [7,8,9,10], [11], [12]
            # new data:                                [12]
            DataFrame({"ordered_on": [12]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9, 10, 11, 12]}),
            [0, 4, 8, 9],
            4,  # max_row_group_size
            False,  # drop_duplicates
            3,  # max_nirgs
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  6 (1.b) /
            # Max row group size as freqstr | New data connected to incomplete rgs.
            # Values on boundary.
            # Writing after recorded data, incomplete row groups.
            # row grps:  0            1        2
            # recorded: [8h00,9h00], [10h00], [11h00]
            # new data:                       [11h00]
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
            "2h",  # max_row_group_size
            False,  # drop_duplicates
            3,  # max_nirgs
            (1, None, False),  # bool: need to sort rgs after write
        ),
        # 3/ Adding data at complete tail, testing 'max_nirgs=None'.
        (
            # Test  7 (2.a) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # row grps:  0          1           2     3
            # recorded: [0,1,2,6], [7,8,9,10], [11], [12]
            # new data:                                [12]
            DataFrame({"ordered_on": [12]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9, 10, 11, 12]}),
            [0, 4, 8, 9],
            4,  # max_row_group_size
            False,  # drop_duplicates
            None,  # max_nirgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  8 (2.b) /
            # Max row group size as freqstr | New data connected to incomplete rgs.
            # Values on boundary.
            # Writing after recorded data, incomplete row groups.
            # row grps:  0            1        2
            # recorded: [8h00,9h00], [10h00], [11h00]
            # new data:                       [11h00]
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
            "2h",  # max_row_group_size
            False,  # drop_duplicates
            None,  # max_nirgs
            (None, None, False),  # bool: need to sort rgs after write
        ),
        # 4/ Adding data just before last incomplete row groups.
        (
            # Test  9 (3.a) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # Enough rows to rewrite all tail.
            # row grps:  0        1              2
            # recorded: [0,1,2], [6,7,8],       [10]
            # new data:                  [8, 9]
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # max_row_group_size | should rewrite tail
            False,  # drop_duplicates
            3,  # max_nirgs | should not rewrite tail
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  10 (3.b) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # Tail is rewritten because with new data, 'max_row_group_size' is
            # reached.
            # row grps:  0        1              2
            # recorded: [0,1,2], [6,7,8],       [10]
            # new data:                  [8, 9]
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # max_row_group_size | should rewrite tail
            True,  # drop_duplicates
            3,  # max_nirgs | should not rewrite tail
            (1, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  11 (3.c) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # Tail is rewritten because with new data, 'max_nirgs' is reached.
            # row grps:  0       1              2
            # recorded: [0,1,2],[6,7,8],       [10]
            # new data:                 [8]
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10]}),
            [0, 3, 6],
            3,  # max_row_group_size | should not rewrite tail
            True,  # drop_duplicates
            2,  # max_nirgs | should rewrite tail
            (1, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  12 (3.d) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # Should not rewrite all tail.
            # row grps:  0         1                2
            # recorded: [0,1,2,3],[6,7,8,8],       [10]
            # new data:                     [8, 9]
            DataFrame({"ordered_on": [8, 9]}),
            DataFrame({"ordered_on": [0, 1, 2, 3, 6, 7, 8, 8, 10]}),
            [0, 4, 8],
            4,  # max_row_group_size | should not rewrite tail
            False,  # drop_duplicates
            3,  # max_nirgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  13 (3.c) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # 'max_nirgs' reached to rewrite all tail.
            # row grps:  0            1                 2
            # recorded: [8h00,9h00], [10h00],          [13h00]
            # new data:                       [12h00]
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
            "2h",  # max_row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_nirgs | should rewrite tail
            (2, None, False),  # bool: need to sort rgs after write
        ),
        (
            # Test  14 (3.d) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # New data is not directly connected to existing values in row
            # groups, so tail is not rewritten.
            # row grps:  0            1                 2
            # recorded: [8h00,9h00], [10h00],          [13h00]
            # new data:                       [11h00]
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
            "2h",  # max_row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_nirgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  15 (3.e) /
            # Max row group size as int | New data connected to incomplete rgs.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # New data is not overlapping with existing row groups.
            # It should be added.
            # row grps:  0                   1        2
            # recorded: [8h00,9h00],        [12h00], [14h00]
            # new data:             [10h30]
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
            "2h",  # max_row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_nirgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  0 /
            # Max row group size as int.
            # drop_duplicates = True
            # Writing in-between recorded data, with incomplete row groups at
            # the end of recorded data.
            # row grps:  0      1            2       3    4
            # recorded: [0,1], [2,      6], [7, 8], [9], [10]
            # new data:        [2, 3, 4]
            DataFrame({"ordered_on": [2, 3, 4]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9, 10]}),
            [0, 2, 4, 6, 7],
            2,
            True,
            2,
            (1, 2, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  4 /
            # Max row group size as int.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # drop_duplicates = False
            # row grps:  0           1        2
            # recorded: [0,1,2],    [6,7,8],[9]
            # new data:         [3]
            DataFrame({"ordered_on": [3]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 9]}),
            [0, 3, 6],
            3,  # max_row_group_size
            False,
            2,
            (None, None, True),  # bool: need to sort rgs after write
        ),
        (
            # Test  7 /
            # Max row group size as int.
            # Writing at end of recorded data, with incomplete row groups at
            # the end of recorded data.
            # drop_duplicates = True
            # One-but last row group is complete, but because new data is
            # overlapping with it, it has to be rewritten.
            # By choice, the rewrite does not propagate till the end.
            # row grps:  0       1              2             3
            # recorded: [0,1,2],[6,7,8],       [10, 11, 12], [13]
            # new data:                 [9, 10]
            DataFrame({"ordered_on": [9, 10]}),
            DataFrame({"ordered_on": [0, 1, 2, 6, 7, 8, 10, 11, 12, 13]}),
            [0, 3, 6, 9],
            3,  # max_row_group_size
            True,
            2,
            (2, 3, True),
        ),
        (
            # Test  5 /
            # Max row group size as pandas freqstr.
            # Writing in-between recorded data, with incomplete row groups at
            # the end of recorded data.
            # drop_duplicates = True
            # row grps:  0        1           2           3      4
            # recorded: [8h,9h], [10h, 11h], [12h, 13h], [14h], [15h]
            # new data:               [11h]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}11:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=8,
                    ),
                },
            ),
            [0, 2, 4, 6, 7],
            2,
            True,
            2,
            (1, 2),
        ),
        (
            # Test  6 /
            # Max row group size as pandas freqstr.
            # Writing in-between recorded data, with incomplete row groups at
            # the end of recorded data.
            # drop_duplicates = False
            # row grps:  0        1           2           3      4
            # recorded: [8h,9h], [10h, 11h], [12h, 13h], [14h], [15h]
            # new data:       [9h]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}9:00")]}),
            DataFrame(
                {
                    "ordered_on": date_range(
                        Timestamp(f"{REF_D}8:00"),
                        freq="1h",
                        periods=8,
                    ),
                },
            ),
            [0, 2, 4, 6, 7],
            2,
            False,
            "2h",  # max_row_group_size
            (2, 2),
        ),
        (
            # Test  16 (3.f) /
            # Max row group size as int | New data at the very start.
            # New data is not overlapping with existing row groups.
            # It should be added.
            # row grps:            0            1        2
            # recorded:           [8h00,9h00], [12h00], [14h00]
            # new data:  [7h30]
            DataFrame({"ordered_on": [Timestamp(f"{REF_D}7:30")]}),
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
            "2h",  # max_row_group_size | should not specifically rewrite tail
            True,  # drop_duplicates
            2,  # max_nirgs | should not rewrite tail
            (None, None, True),  # bool: need to sort rgs after write
        ),
        # do a test with half month freqstr to check shift(freq=SMS) is accepted
    ],
)
def test_indexes_of_overlapping_rrgs(
    tmp_path,
    new_data,
    df_to_record,
    row_group_offsets,
    max_row_group_size,
    drop_duplicates,
    max_nirgs,
    expected,
):
    fp_write(
        f"{tmp_path}/test",
        df_to_record,
        row_group_offsets=row_group_offsets,
        file_scheme="hive",
        write_index=False,
    )
    recorded_pf = ParquetFile(f"{tmp_path}/test")
    res = _indexes_of_overlapping_rrgs(
        new_data=new_data,
        recorded_pf=recorded_pf,
        ordered_on="ordered_on",
        max_row_group_size=max_row_group_size,
        drop_duplicates=drop_duplicates,
        max_nirgs=max_nirgs,
    )
    assert res == expected


def test_init_and_append_std(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Existing folder.
    # (no row index, compression SNAPPY, row group size: 2)
    pdf1 = DataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    pswrite(str(tmp_path), pdf1, max_row_group_size=2, ordered_on="a")
    pf1 = ParquetFile(str(tmp_path))
    assert len(pf1.row_groups) == 3
    for rg in pf1.row_groups:
        assert rg.num_rows == 2
    res1 = pf1.to_pandas()
    assert pdf1.equals(res1)
    # Append
    pdf2 = DataFrame({"a": [6, 7], "b": ["at", "of"]})
    pswrite(str(tmp_path), pdf2, max_row_group_size=2, ordered_on="a")
    res2 = ParquetFile(str(tmp_path)).to_pandas()
    assert concat([pdf1, pdf2]).reset_index(drop=True).equals(res2)


def test_init_no_folder(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. No folder.
    # (no row index, compression SNAPPY, row group size: 2)
    tmp_path = os_path.join(tmp_path, "test")
    pdf = DataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    pswrite(str(tmp_path), pdf, max_row_group_size=2, ordered_on="a")
    res = ParquetFile(str(tmp_path)).to_pandas()
    assert pdf.equals(res)


def test_init_compression_brotli(tmp_path):
    # Initialize a parquet dataset from pandas dataframe. Compression 'BROTLI'.
    # (no row index)
    pdf = DataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]})
    tmp_path1 = os_path.join(tmp_path, "brotli")
    pswrite(str(tmp_path1), pdf, compression="BROTLI", ordered_on="a")
    brotli_s = os_path.getsize(os_path.join(tmp_path1, "part.0.parquet"))
    tmp_path2 = os_path.join(tmp_path, "snappy")
    pswrite(str(tmp_path2), pdf, compression="SNAPPY", ordered_on="a")
    snappy_s = os_path.getsize(os_path.join(tmp_path2, "part.0.parquet"))
    assert brotli_s < snappy_s


def test_init_idx_expansion(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # No level names.
    pdf = DataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        },
    )
    res_midx = to_midx(pdf.columns)
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")],
        names=["l0", "l1"],
    )
    assert res_midx.equals(ref_midx)
    pswrite(str(tmp_path), pdf, cmidx_expand=True, ordered_on=("lev1-col1", "lev2-col1"))
    res = ParquetFile(str(tmp_path)).to_pandas()
    pdf.columns = ref_midx
    assert res.equals(pdf)


def test_init_idx_expansion_sparse_levels(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # Sparse level names.
    pdf = DataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        },
    )
    res_midx = to_midx(pdf.columns, levels=["ah"])
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")],
        names=["ah", "l1"],
    )
    assert res_midx.equals(ref_midx)


def test_coalescing_first_rgs(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # Coalescing reaching first row group. Coalescing triggered because
    # max_row_group_size reached, & also 'max_nirgs'.
    # max_row_group_size = 2
    # max_nirgs = 2
    # rgs                          [ 0, 1]
    # idx                          [ 0, 1]
    # a                            [ 0, 1]
    # a (new data)                       [20]
    # rgs (new)                    [ 0,  , 1]
    pdf = DataFrame({"a": [0, 1]})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [2]
    max_row_group_size = 3
    max_nirgs = 2
    pdf2 = DataFrame({"a": [20]}, index=[0])
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, max_nirgs=max_nirgs, ordered_on="a")
    pf_rec = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs == [3]
    df_ref = concat([pdf, pdf2]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_coalescing_simple_irgs(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # max_row_group_size = 4
    # (incomplete row group size: 1 to be 'incomplete')
    # max_nirgs = 3

    # Case 1, 'max_nirgs" not reached yet.
    # (size of new data: 1)
    # One incomplete row group in the middle of otherwise complete row groups.
    # Because there is only 1 irgs, +1 with the new data to be added (while max
    # is 3), and 2 rows over all irgs (including data to be written), coalescing
    # is not activated.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # a (new data)                                               [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,4 ]
    pdf1 = DataFrame({"a": range(10)})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 4, 5, 9], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1]
    max_row_group_size = 4
    max_nirgs = 3
    pdf2 = DataFrame({"a": [20]}, index=[0])
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, max_nirgs=max_nirgs, ordered_on="a")
    pf_rec1 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec1.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1]
    df_ref1 = concat([pdf1, pdf2]).reset_index(drop=True)
    assert pf_rec1.to_pandas().equals(df_ref1)

    # Case 2, 'max_nirgs" now reached.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,20]
    # a (new data)                                                  [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ]
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, max_nirgs=max_nirgs, ordered_on="a")
    pf_rec2 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec2.row_groups]
    assert len_rgs == [4, 1, 4, 3]
    df_ref2 = concat([df_ref1, pdf2]).reset_index(drop=True)
    assert pf_rec2.to_pandas().equals(df_ref2)


def test_coalescing_simple_max_row_group_size(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # max_row_group_size = 4
    # (incomplete row group size: 1 to be 'incomplete')
    # max_nirgs = 5
    # Coalescing occurs because 'max_row_group_size' is reached.
    # In initial dataset, there are 3 row groups with a single row.
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4, 5]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
    # a (new data)                                                     [20]
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ,  ]
    pdf1 = DataFrame({"a": range(12)})
    dn = os_path.join(tmp_path, "test")
    fp_write(
        dn,
        pdf1,
        row_group_offsets=[0, 4, 5, 9, 10, 11],
        file_scheme="hive",
        write_index=False,
    )
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1, 1]
    max_row_group_size = 4
    max_nirgs = 5
    # With additional row of new data, 'max_row_group_size' is reached.
    pdf2 = DataFrame({"a": [20]}, index=[0])
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, max_nirgs=max_nirgs, ordered_on="a")
    pf_rec1 = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec1.row_groups]
    assert len_rgs == [4, 1, 4, 4]
    df_ref1 = concat([pdf1, pdf2]).reset_index(drop=True)
    assert pf_rec1.to_pandas().equals(df_ref1)


def test_appending_data_with_drop_duplicates(tmp_path):
    # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
    # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    # b                            [10,11,12,13,14,15,16,17,18,19,20]
    # a (new data, ordered_on, duplicates_on)                       [10,20]
    # b (new data, check last)                                      [11,31]
    # 1 duplicate                                                  x  x
    # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, x, 4,  ]
    pdf1 = DataFrame({"a": range(11), "b": range(10, 21)})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 4, 5, 9, 10], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [4, 1, 4, 1, 1]
    max_row_group_size = 4
    pdf2 = DataFrame({"a": [10, 20], "b": [11, 31]})
    # Dropping duplicates '10', 'ordered_on' with 'a'.
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="a")
    pf_rec = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec.row_groups]
    # 'coalesce' mode not requested, so no end row group merging.
    assert len_rgs == [4, 1, 4, 1, 2]
    df_ref = concat([pdf1, DataFrame({"a": 20, "b": 31}, index=[0])]).reset_index(drop=True)
    df_ref.iloc[10] = [10, 11]
    assert pf_rec.to_pandas().equals(df_ref)


def test_appending_data_with_sharp_starts(tmp_path):
    # Validate:
    # - 'sharp starts' (meaning that bins splitting the dataframe to be written
    #   are adjusted so as not to fall in the middle of duplicates.)
    # - is also tested the index 'a' being added to 'duplicates_on', as the 2
    #   last values in 'pdf2' are not dropped despite being duplicates on 'b'.
    # - drop duplicates, keep 'last.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a                    [0,1,2,3,3,3,4,5]
    # b                    [0, , ,3, , ,6, ]
    # c                    [0,0,0,0,0,0,0,0]
    # a (new data, ordered_on)             [5,5,5,5,6,6, 7, 8]
    # b (new data, duplicates_on)          [7,7,8,9,9,9,10,10]
    # c (new data, check last)             [1,2,3,4,5,6, 7, 8]
    # 3 duplicates (on b)                 x x x,  x x x, x  x
    # rgs (new)            [0, , ,1, , ,2,x,x,3, , ,x,4,  , 5]
    # idx                  [0, , ,3, , ,6,    7, , , 10,  ,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [5, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # 'ordered_on' with 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 1, 3, 2, 1]
    df_ref = concat([pdf1.iloc[:-1], pdf2.iloc[1:4], pdf2.iloc[5:]]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_appending_duplicates_on_a_list(tmp_path):
    # Validate same as previous test, but 'duplicates_on' is a list:
    # - 'sharp starts' (meaning that bins splitting the dataframe to be written
    #   are adjusted so as not to fall in the middle of duplicates.)
    # - is also tested the index 'a' being added to 'duplicates_on', as the 2
    #   last values in 'pdf2' are not dropped despite being duplicates on 'b'.
    # - drop duplicates, keep 'last.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a (ordered_on)       [0,1,2,3,3,3,4,5]
    # b (duplicates_on)    [0, , ,3, , ,6, ]
    # c (duplicate last)   [0,0,0,0,0,0,0,0]
    # a (new data)                         [5,5,5,5,6,6, 7, 8]
    # b (new data)                         [7,7,8,9,9,9,10,10]
    # c (new data)                         [1,2,3,4,5,6, 7, 8]
    # 3 duplicates (on c)                 x x x   x x x
    # rgs (new data)       [0, , ,1, , ,2,x,x,3, , ,x,4,  , 5]
    # idx                  [0, , ,3, , ,6,    7, , , 10,  ,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [5, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on=["b"])
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 1, 3, 2, 1]
    df_ref = concat([pdf1.iloc[:-1], pdf2.iloc[1:4], pdf2.iloc[5:]]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_appending_span_several_rgs(tmp_path):
    # Validate:
    # - sorting on 'ordered_on' of data which is the concatenation of existing
    #   data and new data.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a                    [0,1,2,3,3,3,4,5]
    # b                    [0, , ,3, , ,6, ]
    # c                    [0,0,0,0,0,0,0,0]
    # a (new data, ordered_on)        [3,  5,5, 5,6, 6, 7, 8]
    # b (new data, duplicates_on)     [7,  7,8, 9,9, 9,10,10]
    # c (new data, check last)        [1,  2,3, 4,5, 6, 7, 8]
    # 2 duplicates (on b)                 xx,   x x  x, x  x
    # rgs (new)            [0, , ,1, ,,, ,x2, , , x  4, x, 5]
    # idx                  [0, , ,3, ,,, , 8, , ,   11,  ,13]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 5, 5, 5, 6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 8, 9, 9, 9, 10, 10]
    c2 = arange(len_a2) + 1
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 5, 3, 2, 1]
    a_ref = [0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8]
    b_ref = [0, 1, 2, 3, 4, 5, 7, 6, 7, 8, 9, 9, 10, 10]
    c_ref = [0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 6, 7, 8]
    df_ref = DataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


def test_inserting_data(tmp_path):
    # Validate:
    # - inserting data in the middle of existing one, with row group sorting.
    #   Row group 3 becomes row group 4.
    # rgs                  [0, , ,1, , ,2, , , 3,  ]
    # idx                  [0, , ,3, , ,6, , , 9,  ]
    # a                    [0,1,3,3,5,7,7,8,9,11,12]
    # b                    [0, , ,3, , ,6, , , 9,  ]
    # c                    [0,0,0,0,0,0,0,0,0, 0, 0]
    # a (new data, ordered_on)    [3,4,5,  8]
    # b (new data, duplicates_on) [7,7,4,  9]
    # c (new data, check last)    [1,1,1,  1]
    # 1 duplicate (on b)            x  x
    # a (concat)           [11,12,0,1,3,3,3,4,5,5,7,7,8,8, 9] (before rg sorting)
    # b (concat)           [ 9,10,0,1,2,3,7,7,4,4,5,6,7,9, 8]
    # c (cocnat)           [ 0, 0,0,0,0,0,1,1,  1,0,0,0,1, 0]
    # rgs (not sharp)      [     ,x, , ,x, , ,x, , ,x, , , x]
    # rgs (sharp)          [ x,  ,0, ,1, , , ,x 2,3, , , , 4]
    # idx                  [     ,0, ,2, , , ,  6,7, , , ,11]
    a1 = [0, 1, 3, 3, 5, 7, 7, 8, 9, 11, 12]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf1 = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf1, row_group_offsets=[0, 3, 6, 9], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 4, 5, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 4, 9]
    c2 = [1] * len_a2
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [2, 4, 1, 4, 1, 2]
    a_ref = [0, 1, 3, 3, 3, 4, 5, 7, 7, 8, 8, 9, 11, 12]
    b_ref = [0, 1, 2, 3, 7, 7, 4, 5, 6, 7, 9, 8, 9, 10]
    c_ref = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
    df_ref = DataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


def test_inserting_data_with_drop_duplicates(tmp_path):
    # Validate:
    # - inserting data in the middle of existing one, with row group sorting.
    #   Row group 3 becomes row group 4 (same as previous test).
    # rgs                  [0, , ,1, , ,2, , , 3,  ]
    # idx                  [0, , ,3, , ,6, , , 9,  ]
    # a                    [0,1,3,3,5,7,7,8,9,11,12]
    # b                    [0, , ,3, , ,6, , , 9,  ]
    # c                    [0,0,0,0,0,0,0,0,0, 0, 0]
    # a (new data, ordered_on)    [3,4,5,  8]
    # b (new data, duplicates_on) [7,7,4,  9]
    # c (new data, check last)    [1,1,1,  1]
    # 1 duplicate (on b)            x  x
    # a (concat)           [11,12,0,1,3,3,3,4,5,5,7,7,8,8, 9] (before rg sorting)
    # b (concat)           [ 9,10,0,1,2,3,7,7,4,4,5,6,7,9, 8]
    # c (cocnat)           [ 0, 0,0,0,0,0,1,1,  1,0,0,0,1, 0]
    # rgs (not sharp)      [     ,x, , ,x, , ,x, , ,x, , , x]
    # rgs (sharp)          [ x,  ,0, ,1, , , ,x 2,3, , , , 4]
    # idx                  [     ,0, ,2, , , ,  6,7, , , ,11]
    a1 = [0, 1, 3, 3, 5, 7, 7, 8, 9, 11, 12]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0, 3, 6, 9], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 4, 5, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 4, 9]
    c2 = [1] * len_a2
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on="b")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [2, 4, 1, 4, 1, 2]
    a_ref = [0, 1, 3, 3, 3, 4, 5, 7, 7, 8, 8, 9, 11, 12]
    b_ref = [0, 1, 2, 3, 7, 7, 4, 5, 6, 7, 9, 8, 9, 10]
    c_ref = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0]
    df_ref = DataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


def test_inserting_data_no_drop_duplicate(tmp_path):
    # Validate:
    # - inserting data in the middle of existing one, with row group sorting.
    #   Row group 3 becomes row group 4 (same as previous test).
    # rgs                  [0, , ,1, , ,2, , , 3,  ]
    # idx                  [0, , ,3, , ,6, , , 9,  ]
    # a                    [0,1,3,3,5,7,7,8,9,11,12]
    # b                    [0, , ,3, , ,6, , , 9,  ]
    # c                    [0,0,0,0,0,0,0,0,0, 0, 0]
    # a (new data, ordered_on)    [3,4,5,  8]
    # b (new data)                [7,7,4,  9]
    # c (new data, check last)    [1,1,1,  1]
    # 1 duplicate (on b)            x  x
    # a (concat)           [11,12,0,1,3,3,3,4,5,5,7,7,8,8, 9] (before rg sorting)
    # b (concat)           [ 9,10,0,1,2,3,7,7,4,4,5,6,7,9, 8]
    # c (cocnat)           [ 0, 0,0,0,0,0,1,1,  1,0,0,0,1, 0]
    # rgs (not sharp)      [     ,x, , ,x, , ,x, , ,x, , , x]
    # rgs (sharp)          [ x,  ,0, ,1, , , ,2, ,3, , , , 4]
    # idx                  [     ,0, ,2, , , ,6, ,8, , , ,12]
    a1 = [0, 1, 3, 3, 5, 7, 7, 8, 9, 11, 12]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0, 3, 6, 9], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 4, 5, 8]
    len_a2 = len(a2)
    b2 = [7, 7, 4, 9]
    c2 = [1] * len_a2
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', no 'duplicates_on
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a")
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 2, 4, 1, 2]
    a_ref = [0, 1, 3, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9, 11, 12]
    b_ref = [0, 1, 2, 3, 7, 7, 4, 4, 5, 6, 7, 9, 8, 9, 10]
    c_ref = [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    df_ref = DataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


def test_appending_as_if_inserting_no_coalesce(tmp_path):
    # Append data as if it is insertion (no overlap with existing data).
    # Special case:
    # - while duplicate drop is requested, none is performed because data is
    #   not merged with existing one before being appended.
    # - the way number of rows are distributed per row group, the input data
    #   (4 rows) is split into 2 row groups of 2 rows.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a (ordered_on)       [0,1,2,3,3,3,4,5]
    # b (duplicates_on)    [0, , ,3, , ,6, ]
    # c (duplicate last)   [0,0,0,0,0,0,0,0]
    # a (new data)                         [6,6, 7, 8]
    # b (new data)                         [9,9,10,10]
    # c (new data)                         [5,6, 7, 8]
    # 3 duplicates (on c)                   x x
    # rgs (new data)       [0, , ,1, , ,2, ,3, , 4,  ]
    # idx                  [0, , ,3, , ,6, ,8, ,10,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [9, 9, 10, 10]
    c2 = arange(len_a2) + 5
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on=["b"])
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 2, 2, 2]
    df_ref = concat([pdf, pdf2]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_appending_as_if_inserting_with_coalesce(tmp_path):
    # Append data as if it is insertion (no overlap with existing data).
    # Special case:
    # - while duplicate drop is requested, none is performed because data is
    #   not merged with existing one before being appended.
    # - the way number of rows are distributed per row group, the input data
    #   (4 rows) is split into 2 row groups of 2 rows.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a (ordered_on)       [0,1,2,3,3,3,4,5]
    # b (duplicates_on)    [0, , ,3, , ,6, ]
    # c (duplicate last)   [0,0,0,0,0,0,0,0]
    # a (new data)                         [6,6, 7, 8]
    # b (new data)                         [9,9,10,10]
    # c (new data)                         [5,6, 7, 8]
    # 3 duplicates (on c)                   x x
    # rgs (new data)       [0, , ,1, , ,2, ,x 3,  ,  ]
    # idx                  [0, , ,3, , ,6, ,8, ,  ,  ]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * len_a1
    pdf = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    max_nirgs = 2
    a2 = [6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [9, 9, 10, 10]
    c2 = arange(len_a2) + 5
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(
        dn,
        pdf2,
        max_row_group_size=max_row_group_size,
        ordered_on="a",
        duplicates_on=["b"],
        max_nirgs=max_nirgs,
    )
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 3, 2, 3]
    df_ref = concat([pdf, pdf2.iloc[1:]]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_duplicates_all_cols(tmp_path):
    # Validate:
    # - use of empty list '[]' for 'duplicates_on'.
    # rgs                  [0, , ,1, , ,2, ]
    # idx                  [0, , ,3, , ,6, ]
    # a                    [0,1,2,3,3,3,4,5]
    # b                    [0, , ,3, , ,6, ]
    # c                    [0,0,0,0,0,0,0,2]
    # a (new data, ordered_on)        [3,  5,5,5, 6]
    # b (new data, duplicates_on)     [7,  7,7,7, 9]
    # c (new data, check last)        [1,  2,3,2, 5]
    # 1 duplicates (b & c)                xx,  x
    # rgs (new)            [0, , ,1, ,,, ,   2, , 3]
    # idx                  [0, , ,3, ,,, ,   8, ,10]
    a1 = [0, 1, 2, 3, 3, 3, 4, 5]
    len_a1 = len(a1)
    b1 = range(len_a1)
    c1 = [0] * (len_a1 - 1) + [2]
    pdf = DataFrame({"a": a1, "b": b1, "c": c1})
    dn = os_path.join(tmp_path, "test")
    fp_write(dn, pdf, row_group_offsets=[0, 3, 6], file_scheme="hive", write_index=False)
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [3, 3, 2]
    max_row_group_size = 3
    a2 = [3, 5, 5, 5, 6]
    b2 = [7, 7, 7, 7, 9]
    c2 = [1, 2, 3, 2, 5]
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    pswrite(dn, pdf2, max_row_group_size=max_row_group_size, ordered_on="a", duplicates_on=[])
    pf_rec = ParquetFile(dn)
    len_rgs_rec = [rg.num_rows for rg in pf_rec.row_groups]
    assert len_rgs_rec == [3, 5, 2, 1]
    a_ref = [0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 6]
    b_ref = [0, 1, 2, 3, 4, 5, 7, 6, 7, 7, 9]
    c_ref = [0, 0, 0, 0, 0, 0, 1, 0, 3, 2, 5]
    df_ref = DataFrame({"a": a_ref, "b": b_ref, "c": c_ref})
    assert pf_rec.to_pandas().equals(df_ref)


def test_drop_duplicates_wo_coalescing_irgs(tmp_path):
    # Initialize a parquet dataset directly with fastparquet.
    # 2 last row groups are incomplete row groups.
    # But coalescing is not triggered with new data.
    # However, drop duplicate is requested and new data overlap with last
    # existing row group.
    # Targeted result is that one-but-last row group (incomplete one) is not
    # coalesced.
    # max_row_group_size = 5 (one row per row group for incomplete rgs)
    # max_nirgs = 4 (4 incomplete rgs, but drop duplicate with new data, so
    # after merging new data, only 2 incomplete row groups will remain)
    # rgs                          [0, , , , ,1,2]
    # idx                          [0,1,2,3,4,5,6]
    # a (ordered_on)               [0,1,2,3,4,5,6]
    # b (keep last)                [0,0,0,0,0,0,0]
    # a (new data)                               [6]
    # b (new data)                               [1]
    # rgs (new)                    [0, , , , ,1,x,2]
    n_val = 7
    pdf = DataFrame({"a": range(n_val), "b": [0] * n_val})
    dn = os_path.join(tmp_path, "test")
    fp_write(
        dn,
        pdf,
        row_group_offsets=[0, n_val - 2, n_val - 1],
        file_scheme="hive",
        write_index=False,
    )
    pf = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf.row_groups]
    assert len_rgs == [n_val - 2, 1, 1]
    max_row_group_size = 5
    max_nirgs = 4
    pdf2 = DataFrame({"a": [n_val - 1], "b": [1]})
    pswrite(
        dn,
        pdf2,
        max_row_group_size=max_row_group_size,
        max_nirgs=max_nirgs,
        ordered_on="a",
        duplicates_on="a",
    )
    pf_rec = ParquetFile(dn)
    len_rgs = [rg.num_rows for rg in pf_rec.row_groups]
    # Only the last row group has been rewritten, without coalescing
    # one-but-last incomplete row group.
    assert len_rgs == [n_val - 2, 1, 1]
    df_ref = concat([pdf[:-1], pdf2]).reset_index(drop=True)
    assert pf_rec.to_pandas().equals(df_ref)


def test_exception_ordered_on_not_existing(tmp_path):
    # While 'ordered_on' is defined, it is not in seed data.
    n_val = 5
    pdf = DataFrame({"a": range(n_val), "b": [0] * n_val})
    dn = os_path.join(tmp_path, "test")
    # Write a 1st set of data.
    fp_write(
        dn,
        pdf,
        row_group_offsets=[0, n_val - 2, n_val - 1],
        file_scheme="hive",
        write_index=False,
    )
    # Append with oups same set of data, pandas dataframe.
    with pytest.raises(ValueError, match="^column 'ts'"):
        pswrite(dn, pdf, ordered_on="ts")


def test_exception_check_cmidx(tmp_path):
    # Check 1st no column level names.
    df = DataFrame({("a", 1): [1]})
    with pytest.raises(ValueError, match="^not possible to have level name"):
        pswrite(tmp_path, df, ordered_on="a")
    # Check with one column name not being a string.
    # Correcting column names.
    df.columns.set_names(["1", "2"], level=[0, 1], inplace=True)
    with pytest.raises(TypeError, match="^name 1 has to be"):
        pswrite(tmp_path, df, ordered_on="a")
