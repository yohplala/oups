#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
import pytest
from numpy import array_equal
from pandas import DataFrame

from oups.store.data_overlap import DataOverlapInfo
from tests.test_store.conftest import create_parquet_file


@pytest.mark.parametrize(
    "test_id,df_data,pf_data,expected,max_row_group_size",
    [
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
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_data_overlap(
    test_id,
    df_data,
    pf_data,
    expected,
    max_row_group_size,
    create_parquet_file,
):
    """
    Test DataOverlapInfo analysis with various scenarios.

    Parameters
    ----------
    test_id : str
        Identifier for the test case.
    df_data : list
        Data for input DataFrame.
    pf_data : list
        Data for ParquetFile.
    expected : dict
        Expected values for overlap analysis.
    max_row_group_size : int
        Maximum size for row groups.
    create_parquet_file : callable
        Fixture to create temporary parquet files.

    """
    df = DataFrame({"ordered": df_data})
    pf_data = DataFrame({"ordered": pf_data})
    pf = create_parquet_file(pf_data, row_group_offsets=max_row_group_size)

    overlap_info = DataOverlapInfo.analyze(
        df=df,
        pf=pf,
        ordered_on="ordered",
        max_row_group_size=max_row_group_size,
    )

    assert overlap_info.has_pf_head == expected["has_pf_head"]
    assert overlap_info.has_df_head == expected["has_df_head"]
    assert overlap_info.has_overlap == expected["has_overlap"]
    assert overlap_info.has_pf_tail == expected["has_pf_tail"]
    assert overlap_info.has_df_tail == expected["has_df_tail"]
    assert overlap_info.df_idx_overlap_start == expected["df_idx_overlap_start"]
    assert overlap_info.df_idx_overlap_end_excl == expected["df_idx_overlap_end_excl"]
    assert overlap_info.rg_idx_overlap_start == expected["rg_idx_overlap_start"]
    assert overlap_info.rg_idx_overlap_end_excl == expected["rg_idx_overlap_end_excl"]
    assert array_equal(overlap_info.df_idx_rg_starts, expected["df_idx_rg_starts"])
    assert array_equal(overlap_info.df_idx_rg_ends_excl, expected["df_idx_rg_ends_excl"])
