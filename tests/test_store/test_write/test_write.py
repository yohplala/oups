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
from pandas import concat

from oups.store.write import to_pandas_midx
from oups.store.write import write_ordered
from tests.test_store.conftest import create_parquet_file


def test_to_cmidx(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # No level names.
    pdf = DataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        },
    )
    res_midx = to_pandas_midx(pdf.columns)
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")],
        names=["l0", "l1"],
    )
    assert res_midx.equals(ref_midx)
    write_ordered(str(tmp_path), ordered_on=("lev1-col1", "lev2-col1"), df=pdf, to_cmidx=[])
    res = ParquetFile(str(tmp_path)).to_pandas()
    pdf.columns = ref_midx
    assert res.equals(pdf)


def test_to_cmidx_sparse_levels(tmp_path):
    # Expanding column index into a 2-level column multi-index.
    # Sparse level names.
    pdf = DataFrame(
        {
            "('lev1-col1','lev2-col1')": range(6, 12),
            "('lev1-col2','lev2-col2')": ["ah", "oh", "uh", "ih", "ai", "oi"],
        },
    )
    res_midx = to_pandas_midx(pdf.columns, levels=["ah"])
    ref_midx = MultiIndex.from_tuples(
        [("lev1-col1", "lev2-col1"), ("lev1-col2", "lev2-col2")],
        names=["ah", "l1"],
    )
    assert res_midx.equals(ref_midx)


@pytest.mark.parametrize(
    "test_id, initial_data,write_ordered_data,row_group_target_size,max_n_off_target_rgs,duplicates_on,expected",
    [
        (
            "init_and_append_std",
            {},  # init with 'create_parquet_file'
            [
                DataFrame({"a": range(6), "b": ["ah", "oh", "uh", "ih", "ai", "oi"]}),
                DataFrame({"a": [6, 7], "b": ["at", "of"]}),
            ],  # 'write_ordered'
            2,  # row_group_target_size
            None,  # max_n_off_target_rgs
            None,  # duplicates_on
            {
                "rgs_length": [[2, 2, 2], [2, 2, 2, 2]],
                "dfs": [
                    DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": ["ah", "oh", "uh", "ih", "ai", "oi"]}),
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 4, 5, 6, 7],
                            "b": ["ah", "oh", "uh", "ih", "ai", "oi", "at", "of"],
                        },
                    ),
                ],
            },
        ),
        (  # Coalescing reaching first row group. Coalescing triggered because
            # 'row_group_target_size' reached, & also 'max_n_off_target_rgs'.
            # row_group_target_size = 2
            # max_n_off_target_rgs = 2
            # rgs                          [ 0, 1]
            # idx                          [ 0, 1]
            # a                            [ 0, 1]
            # a (new data)                       [20]
            # rgs (new)                    [ 0,  , 1]
            "coalescing_first_rg",
            {"df": DataFrame({"a": [0, 1]}), "row_group_offsets": [0]},
            [DataFrame({"a": [20]}, index=[0])],  # write_ordered_data
            4,  # row_group_target_size
            1,  # max_n_off_target_rgs
            None,  # duplicates_on
            {
                "rgs_length": [[3]],
                "dfs": [
                    DataFrame({"a": [0, 1, 20]}),
                ],
            },
        ),
        (
            "coalescing_multiple_rgs",
            # Initialize with rg 3 incomplete row groups (size 1)
            {"df": DataFrame({"a": range(10)}), "row_group_offsets": [0, 4, 5, 9]},
            [
                # Case 1, 'max_n_off_target_rgs' not reached yet.
                # (size of new data: 1)
                # One incomplete row group in the middle of otherwise complete
                # row groups. Because there is only 1 irgs, +1 with the new data
                # to be added (while max is 3), and 2 rows over all irgs
                # (including data to be written), coalescing is not activated.
                # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3]
                # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                # a (new data)                                               [20]
                # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,4 ]
                DataFrame({"a": [20]}, index=[0]),  # First append
                # Case 2, 'max_n_off_target_rgs' now reached.
                # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
                # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
                # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,20]
                # a (new data)                                                  [20]
                # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ]
                DataFrame({"a": [20]}, index=[0]),  # Second append
            ],
            4,  # row_group_target_size
            2,  # max_n_off_target_rgs
            None,  # duplicates_on
            {
                "rgs_length": [
                    [4, 1, 4, 1, 1],  # After first write
                    [4, 1, 4, 3],  # After second write
                ],
                "dfs": [
                    DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20]}),
                    DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 20]}),
                ],
            },
        ),
        (
            "coalescing_row_group_target_size",
            # Initialize with rg 3 incomplete row groups (size 1)
            {"df": DataFrame({"a": range(12)}), "row_group_offsets": [0, 4, 5, 9, 10, 11]},
            # Coalescing occurs because 'row_group_target_size' is reached.
            # In initial dataset, there are 3 row groups with a single row.
            # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4, 5]
            # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
            # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11]
            # a (new data)                                                     [20]
            # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3,  ,  ,  ]
            [DataFrame({"a": [20]}, index=[0])],
            4,  # row_group_target_size
            5,  # max_n_off_target_rgs
            None,  # duplicates_on
            {
                "rgs_length": [[4, 1, 4, 4]],
                "dfs": [
                    DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20]}),
                ],
            },
        ),
        (
            "appending_with_drop_duplicates",
            {
                "df": DataFrame({"a": range(11), "b": range(10, 21)}),
                "row_group_offsets": [0, 4, 5, 9, 10],
            },
            # rgs                          [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, 4]
            # idx                          [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
            # a                            [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
            # b                            [10,11,12,13,14,15,16,17,18,19,20]
            # a (new data, ordered_on, duplicates_on)                       [10,20]
            # b (new data, check last)                                      [11,31]
            # 1 duplicate                                                  x  x
            # rgs (new)                    [ 0,  ,  ,  , 1, 2,  ,  ,  , 3, x, 4,  ]
            [DataFrame({"a": [10, 20], "b": [11, 31]})],
            4,  # row_group_target_size
            None,  # max_n_off_target_rgs
            "a",  # duplicates_on
            {
                "rgs_length": [[4, 1, 4, 1, 2]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
                            "b": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 11, 31],
                        },
                    ),
                ],
            },
        ),
        (
            "appending_with_duplicates_on_as_str",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 2, 3, 3, 3, 4, 5],
                        "b": range(8),
                        "c": [0] * 8,
                    },
                ),
                "row_group_offsets": [0, 3, 6],
            },
            # Validate:
            # - index 'a' being added to 'duplicates_on', as the 2 last values
            #   in 'pdf2' are not dropped despite being duplicates on 'b'.
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
            [
                DataFrame(
                    {
                        "a": [5, 5, 5, 5, 6, 6, 7, 8],
                        "b": [7, 7, 8, 9, 9, 9, 10, 10],
                        "c": arange(8) + 1,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            "b",  # duplicates_on (a is added implicitly as ordered_on)
            {
                "rgs_length": [[3, 3, 3, 3, 1]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                            "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10],
                            "c": [0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 6, 7, 8],
                        },
                    ),
                ],
            },
        ),
        (
            "appending_with_duplicates_on_as_list",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 2, 3, 3, 3, 4, 5],
                        "b": range(8),
                        "c": [0] * 8,
                    },
                ),
                "row_group_offsets": [0, 3, 6],
            },
            # Validate same as previous test, but 'duplicates_on' is a list:
            # - is also tested the index 'a' being added to 'duplicates_on', as
            #   the 2 last values in 'pdf2' are not dropped despite being
            #   duplicates on 'b'.
            # - drop duplicates, keep 'last.
            # rgs                  [0, , ,1, , ,2, ]
            # idx                  [0, , ,3, , ,6, ]
            # a (ordered_on)       [0,1,2,3,3,3,4,5]
            # b (duplicates_on)    [0, , ,3, , ,6, ]
            # c (duplicate last)   [0,0,0,0,0,0,0,0]
            # a (new data)                         [5,5,5,5,6,6, 7, 8]
            # b (new data)                         [7,7,8,9,9,9,10,10]
            # c (new data)                         [1,2,3,4,5,6, 7, 8]
            # 3 duplicates (on b)                 x x x,  x x x, x  x
            # rgs (new)            [0, , ,1, , ,2,x,x,3, , ,x,4,  , 5]
            # idx                  [0, , ,3, , ,6,    7, , , 10,  ,  ]
            [
                DataFrame(
                    {
                        "a": [5, 5, 5, 5, 6, 6, 7, 8],
                        "b": [7, 7, 8, 9, 9, 9, 10, 10],
                        "c": arange(8) + 1,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            ["b"],  # duplicates_on (a is added implicitly as ordered_on)
            {
                "rgs_length": [[3, 3, 3, 3, 1]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                            "b": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10],
                            "c": [0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 6, 7, 8],
                        },
                    ),
                ],
            },
        ),
        (
            "appending_span_several_rgs",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 2, 3, 3, 3, 4, 5],
                        "b": range(8),
                        "c": [0] * 8,
                    },
                ),
                "row_group_offsets": [0, 3, 6],
            },
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
            [
                DataFrame(
                    {
                        "a": [3, 5, 5, 5, 6, 6, 7, 8],
                        "b": [7, 7, 8, 9, 9, 9, 10, 10],
                        "c": arange(8) + 1,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            "b",  # duplicates_on
            {
                "rgs_length": [[3, 3, 3, 3, 2]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8],
                            "b": [0, 1, 2, 3, 4, 5, 7, 6, 7, 8, 9, 9, 10, 10],
                            "c": [0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 4, 6, 7, 8],
                        },
                    ),
                ],
            },
        ),
        (
            "inserting_data_with_drop_duplicates",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 3, 3, 5, 7, 7, 8, 9, 11, 12],
                        "b": range(11),
                        "c": [0] * 11,
                    },
                ),
                "row_group_offsets": [0, 3, 6, 9],
            },
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
            # c (concat)           [ 0, 0,0,0,0,0,1,1,  1,0,0,0,1, 0]
            # rgs (not sharp)      [     ,x, , ,x, , ,x, , ,x, , , x]
            # rgs (sharp)          [ x,  ,0, ,1, , , ,x 2,3, , , , 4]
            # idx                  [     ,0, ,2, , , ,  6,7, , , ,11]
            [
                DataFrame(
                    {
                        "a": [3, 4, 5, 8],
                        "b": [7, 7, 4, 9],
                        "c": [1] * 4,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            "b",  # duplicates_on
            {
                "rgs_length": [[3, 3, 3, 3, 2]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 3, 3, 3, 4, 5, 7, 7, 8, 8, 9, 11, 12],
                            "b": [0, 1, 2, 3, 7, 7, 4, 5, 6, 7, 9, 8, 9, 10],
                            "c": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                        },
                    ),
                ],
            },
        ),
        (
            "inserting_data_no_drop_duplicate",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 3, 3, 5, 7, 7, 8, 9, 11, 12],
                        "b": range(11),
                        "c": [0] * 11,
                    },
                ),
                "row_group_offsets": [0, 3, 6, 9],
            },
            [
                DataFrame(
                    {
                        "a": [3, 4, 5, 8],
                        "b": [7, 7, 4, 9],
                        "c": [1] * 4,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            None,  # duplicates_on
            {
                "rgs_length": [[3, 3, 3, 3, 1, 2]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 3, 3, 3, 4, 5, 5, 7, 7, 8, 8, 9, 11, 12],
                            "b": [0, 1, 2, 3, 7, 7, 4, 4, 5, 6, 7, 9, 8, 9, 10],
                            "c": [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        },
                    ),
                ],
            },
        ),
        (
            "appending_as_if_inserting_no_coalesce",
            {
                "df": DataFrame(
                    {
                        "a": [0, 1, 2, 3, 3, 3, 4, 5],
                        "b": range(8),
                        "c": [0] * 8,
                    },
                ),
                "row_group_offsets": [0, 3, 6],
            },
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
            [
                DataFrame(
                    {
                        "a": [6, 6, 7, 8],
                        "b": [9, 9, 10, 10],
                        "c": arange(4) + 5,
                    },
                ),
            ],
            3,  # row_group_target_size
            None,  # max_n_off_target_rgs
            ["b"],  # duplicates_on
            {
                "rgs_length": [[3, 3, 2, 3]],
                "dfs": [
                    DataFrame(
                        {
                            "a": [0, 1, 2, 3, 3, 3, 4, 5, 6, 7, 8],
                            "b": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 10],
                            "c": [0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8],
                        },
                    ),
                ],
            },
        ),
    ],
)
def test_write_ordered(
    test_id,
    tmp_path,
    initial_data,
    write_ordered_data,
    row_group_target_size,
    max_n_off_target_rgs,
    duplicates_on,
    expected,
):
    """
    Test writing and appending data to a parquet file.

    Parameters
    ----------
    test_id : str
        Test identifier.
    initial_data : Union[DataFrame, Dict[str, Any]]
        Initial data to write. If DataFrame, uses write_ordered. If dict with 'df' and 'row_group_offsets',
        uses fp_write.
    write_ordered_data : DataFrame
        Data to write using write_ordered.
    row_group_target_size : int
        Target size for row groups.
    max_n_off_target_rgs : Optional[int]
        Maximum number of off-target row groups allowed.
    duplicates_on : Optional[str]
        Column to use for duplicate detection.
    expected : List[int]
        Expected number of rows in each row group after appending.

    """
    ordered_on = "a"
    tmp_path = f"{str(tmp_path)}/test_data"
    # Init phase.
    if initial_data:
        # Use 'write' from fastparquet with 'row_group_offsets'.
        create_parquet_file(
            tmp_path,
            **initial_data,
        )

    # Phase 'write_ordered()'.
    for write_ordered_df, expected_rgs, expected_df in zip(
        write_ordered_data,
        expected["rgs_length"],
        expected["dfs"],
    ):
        write_ordered(
            tmp_path,
            ordered_on=ordered_on,
            df=write_ordered_df,
            row_group_target_size=row_group_target_size,
            max_n_off_target_rgs=max_n_off_target_rgs,
            duplicates_on=duplicates_on,
        )
        # Verify state after this append
        pf_rec = ParquetFile(tmp_path)
        assert [rg.num_rows for rg in pf_rec.row_groups] == expected_rgs
        print("pf_rec.to_pandas()")
        print(pf_rec.to_pandas())
        assert pf_rec.to_pandas().equals(expected_df)


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
    row_group_target_size = 3
    max_n_off_target_rgs = 2
    a2 = [6, 6, 7, 8]
    len_a2 = len(a2)
    b2 = [9, 9, 10, 10]
    c2 = arange(len_a2) + 5
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    write_ordered(
        dn,
        pdf2,
        row_group_target_size=row_group_target_size,
        ordered_on="a",
        duplicates_on=["b"],
        max_n_off_target_rgs=max_n_off_target_rgs,
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
    row_group_target_size = 3
    a2 = [3, 5, 5, 5, 6]
    b2 = [7, 7, 7, 7, 9]
    c2 = [1, 2, 3, 2, 5]
    pdf2 = DataFrame({"a": a2, "b": b2, "c": c2})
    # ordered on 'a', duplicates on 'b' ('a' added implicitly)
    write_ordered(
        dn,
        pdf2,
        row_group_target_size=row_group_target_size,
        ordered_on="a",
        duplicates_on=[],
    )
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
    # row_group_target_size = 5 (one row per row group for incomplete rgs)
    # max_n_off_target_rgs = 4 (4 incomplete rgs, but drop duplicate with new data, so
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
    row_group_target_size = 5
    max_n_off_target_rgs = 4
    pdf2 = DataFrame({"a": [n_val - 1], "b": [1]})
    write_ordered(
        dn,
        pdf2,
        row_group_target_size=row_group_target_size,
        max_n_off_target_rgs=max_n_off_target_rgs,
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
        write_ordered(dn, pdf, ordered_on="ts")
