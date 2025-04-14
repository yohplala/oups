#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import Dict

import pytest
from numpy import array
from numpy import bool_
from numpy import int_
from numpy import ones
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.ordered_atomic_regions import DF_IDX_END_EXCL
from oups.store.ordered_atomic_regions import HAS_DF_CHUNK
from oups.store.ordered_atomic_regions import HAS_ROW_GROUP
from oups.store.ordered_atomic_regions import RG_IDX_END_EXCL
from oups.store.ordered_atomic_regions import RG_IDX_START
from oups.store.ordered_atomic_regions import NRowsSplitStrategy
from oups.store.ordered_atomic_regions import TimePeriodSplitStrategy
from oups.store.ordered_atomic_regions import compute_ordered_atomic_regions


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
                HAS_DF_CHUNK: array([True, True, True]),
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
                HAS_DF_CHUNK: array([True, True, True]),
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
                HAS_DF_CHUNK: array([True, True, True]),
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
                HAS_DF_CHUNK: array([True, True, True]),
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
                HAS_DF_CHUNK: array([False, True, True]),
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
                HAS_DF_CHUNK: array([True, False, True]),
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
                HAS_DF_CHUNK: array([True, True, False]),
            },
        ),
        (
            "multiple_gaps_df_not_overlapping_rg",
            array([20, 40, 43]),  # rg_mins
            array([25, 42, 45]),  # rg_maxs
            Series([5, 22, 32, 41, 42, 46, 52]),  # df_ordered_on
            True,
            {
                RG_IDX_START: array([0, 0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([0, 1, 1, 2, 3, 3]),
                DF_IDX_END_EXCL: array([1, 2, 3, 5, 5, 7]),
                HAS_ROW_GROUP: array([False, True, False, True, True, False]),
                HAS_DF_CHUNK: array([True, True, True, True, False, True]),
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
                HAS_DF_CHUNK: array([True, True, True, False, True, True]),
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
                HAS_DF_CHUNK: array([False, True, True, True]),
            },
        ),
    ],
)
def test_compute_ordered_atomic_regions(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: Series,
    drop_duplicates: bool,
    expected: NDArray,
) -> None:
    """
    Test compute_ordered_atomic_regions with various scenarios.
    """
    oars_prop = compute_ordered_atomic_regions(
        rg_mins,
        rg_maxs,
        df_ordered_on,
        drop_duplicates,
    )
    # Check structured array fields
    assert_array_equal(oars_prop[RG_IDX_START], expected[RG_IDX_START])
    assert_array_equal(oars_prop[RG_IDX_END_EXCL], expected[RG_IDX_END_EXCL])
    assert_array_equal(oars_prop[DF_IDX_END_EXCL], expected[DF_IDX_END_EXCL])
    assert_array_equal(oars_prop[HAS_ROW_GROUP], expected[HAS_ROW_GROUP])
    assert_array_equal(oars_prop[HAS_DF_CHUNK], expected[HAS_DF_CHUNK])


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
def test_compute_ordered_atomic_regions_validation(
    test_id: str,
    rg_ordered_on_mins: array,
    rg_ordered_on_maxs: array,
    df_ordered_on: Series,
    expected_error: Exception,
) -> None:
    """
    Test input validation in compute_ordered_atomic_regions.

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
        compute_ordered_atomic_regions(
            rg_ordered_on_mins=rg_ordered_on_mins,
            rg_ordered_on_maxs=rg_ordered_on_maxs,
            df_ordered_on=df_ordered_on,
            drop_duplicates=True,
        )


def test_nrows_split_strategy_likely_on_target_size():
    """
    Test NRowsSplitStrategy strategy and likely_on_target_size.
    """
    target_size = 100  # min size: 80
    # Create mock oars_desc with 5 regions:
    # 1. RG only (90 rows) - on target
    # 2. DF only (85 rows) - on target
    # 3. Both RG (50) and DF (40) - on target due to potential duplicates
    # 4. Both RG (30) and DF (30) - too small
    # 5. Both RG (120) and DF (0) - too large and only row group.
    # 6. Both RG (30) and DF (80) - too large but accepted since it has a dfc.
    oars_desc = ones(
        6,
        dtype=[
            (RG_IDX_START, int_),
            (RG_IDX_END_EXCL, int_),
            (DF_IDX_END_EXCL, int_),
            (HAS_ROW_GROUP, bool_),
            (HAS_DF_CHUNK, bool_),
        ],
    )
    # Set which regions have row groups and DataFrame chunks
    oars_desc[HAS_ROW_GROUP] = array([True, False, True, True, True, True])
    oars_desc[HAS_DF_CHUNK] = array([False, True, True, True, False, True])
    # Set up DataFrame chunk sizes through df_idx_end_excl
    oars_desc[DF_IDX_END_EXCL] = array([0, 85, 125, 155, 155, 235])
    # Create row group sizes array
    rgs_n_rows = array([90, 50, 30, 120, 30])
    # Initialize strategy
    strategy = NRowsSplitStrategy(
        drop_duplicates=True,
        oars_desc=oars_desc,
        rgs_n_rows=rgs_n_rows,
        row_group_target_size=target_size,
        max_n_off_target_rgs=1,
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
    result = strategy.likely_on_target_size
    expected = array([True, True, True, False, False, True])
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, oars, expected",
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
                DF_IDX_END_EXCL: array([0, 0, 1]),
                HAS_ROW_GROUP: array([True, True, False]),
                HAS_DF_CHUNK: array([False, False, True]),
            },
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
                DF_IDX_END_EXCL: array([0, 1]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
                DF_IDX_END_EXCL: array([1]),
                HAS_ROW_GROUP: array([True]),
                HAS_DF_CHUNK: array([True]),
            },
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
                DF_IDX_END_EXCL: array([2, 2]),
                HAS_ROW_GROUP: array([False, True]),
                HAS_DF_CHUNK: array([True, False]),
            },
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
            #   1,         Jun,   10/06,   15/06,                , False (both RG & DFc in period)
            #   2,         Jun,                    16/06,   18/06, False (both RG & DFc in period)
            array([Timestamp("2024-06-10")]),  # rg_mins
            array([Timestamp("2024-06-15")]),  # rg_maxs
            Series([Timestamp("2024-06-16"), Timestamp("2024-06-18")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
            #   1,         Jul,                    15/07,        , False (DFc spans several periods)
            #   1,         Aug,                             10/08,
            #   2,         Aug,   15/08,   17/08,                , False (both RG & DFc in period)
            array([Timestamp("2024-08-15")]),  # rg_mins
            array([Timestamp("2024-08-17")]),  # rg_maxs
            Series([Timestamp("2024-07-15"), Timestamp("2024-08-10")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([2, 2]),
                HAS_ROW_GROUP: array([False, True]),
                HAS_DF_CHUNK: array([True, False]),
            },
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
            #   1,         Sep,   11/09,                         , False (RG spans several periods)
            #   1,         Oct,            15/10,
            #   2,         Oct,                    16/10,   18/10, False (both RG & DFc in period)
            array([Timestamp("2024-09-11")]),  # rg_mins
            array([Timestamp("2024-10-15")]),  # rg_maxs
            Series([Timestamp("2024-10-16"), Timestamp("2024-10-18")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
            #   1,         Nov,                    15/11,   17/11, False (both RG & DFc in period)
            #   2,         Nov,   18/11,                         , False (RG spans several periods)
            #   2,         Dec,            05/12,
            array([Timestamp("2024-11-18")]),  # rg_mins
            array([Timestamp("2024-12-05")]),  # rg_maxs
            Series([Timestamp("2024-11-15"), Timestamp("2024-11-17")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([2, 2]),
                HAS_ROW_GROUP: array([False, True]),
                HAS_DF_CHUNK: array([True, False]),
            },
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
            #  14,         Jan,   01/01,   04/01,                , False (both RG & DFc in period)
            #  15,         Jan,                    16/01,        , False (DFc spans several periods)
            #  15,         Feb,                             01/02,
            array([Timestamp("2024-01-01")]),  # rg_mins
            array([Timestamp("2024-01-04")]),  # rg_maxs
            Series([Timestamp("2024-01-16"), Timestamp("2024-02-01")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
            #   1,         Mar,   01/03,   02/03,                , True (single RG in period)
            #   2,         Apr,                    01/04,   30/04, True (single DFc in period)
            array([Timestamp("2024-03-01")]),  # rg_mins
            array([Timestamp("2024-03-02")]),  # rg_maxs
            Series([Timestamp("2024-04-01"), Timestamp("2024-04-30")]),  # df_ordered_on
            {
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
                DF_IDX_END_EXCL: array([2, 4]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([True, True]),
            },
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
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, True]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
                DF_IDX_END_EXCL: array([0, 2]),
                HAS_ROW_GROUP: array([True, False]),
                HAS_DF_CHUNK: array([False, True]),
            },
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
def test_time_period_split_strategy(test_id, rg_mins, rg_maxs, df_ordered_on, oars, expected):
    """
    Test TimePeriodSplitStrategy initialization and likely_on_target_size.
    """
    time_period = "MS"
    # Create oars_desc with 6 regions
    oars_desc = ones(
        len(expected["likely_on_target"]),
        dtype=[
            (RG_IDX_START, int_),
            (RG_IDX_END_EXCL, int_),
            (DF_IDX_END_EXCL, int_),
            (HAS_ROW_GROUP, bool_),
            (HAS_DF_CHUNK, bool_),
        ],
    )
    # Set required OAR description
    oars_desc[DF_IDX_END_EXCL] = oars[DF_IDX_END_EXCL]
    oars_desc[HAS_ROW_GROUP] = oars[HAS_ROW_GROUP]
    oars_desc[HAS_DF_CHUNK] = oars[HAS_DF_CHUNK]
    # Initialize strategy
    strategy = TimePeriodSplitStrategy(
        rg_ordered_on_mins=rg_mins,
        rg_ordered_on_maxs=rg_maxs,
        df_ordered_on=df_ordered_on,
        oars_desc=oars_desc,
        row_group_period=time_period,
    )
    # Test period_bounds
    assert_array_equal(strategy.period_bounds, expected["period_bounds"])
    # Test oars_mins_maxs
    assert_array_equal(strategy.oars_mins_maxs, expected["oars_mins_maxs"])
    # Test likely_on_target_size
    assert_array_equal(strategy.likely_on_target_size, expected["likely_on_target"])


@pytest.mark.parametrize(
    "test_id, drop_duplicates, oars_desc_dict, rgs_n_rows, row_group_target_size, oar_idx_mrs_starts_ends_excl, expected",
    [
        (
            "single_sequence_encompassing_all_oars_drop_duplicates",
            # row_group_target_size : 100
            # rgs_n_rows    :  [  50,   50,      ,   50,    50]
            # has_row_group :  [True, True, False, True,  True]
            # dfc_ends_excl:   [  10,   25,    60,   75,      ]
            # has_df_chunk  :  [True, True,  True, True, False]
            True,  # drop_duplicates
            {
                RG_IDX_START: array([0, 1, 1, 2, 3]),
                RG_IDX_END_EXCL: array([1, 2, 2, 3, 4]),
                DF_IDX_END_EXCL: array([10, 25, 60, 75, 75]),
                HAS_ROW_GROUP: array([True, True, False, True, True]),
                HAS_DF_CHUNK: array([True, True, True, True, False]),
            },
            array([50, 50, 50, 50]),
            100,  # row_group_target_size
            array([[0, 5]]),  # merge region contains 3 OARs
            {
                "oars_min_n_rows": array([50, 50, 35, 50, 50]),
                "oars_merge_sequences": [
                    (0, array([[2, 25], [4, 75]])),  # single sequence with 3 OARs
                ],
            },
        ),
        # /!\ Todo: test when there are fewer rows in selection than target size.
    ],
)
def test_nrows_split_strategy_partition_merge_regions(
    test_id: str,
    drop_duplicates: bool,
    oars_desc_dict: Dict[str, NDArray],
    rgs_n_rows: NDArray,
    row_group_target_size: int,
    oar_idx_mrs_starts_ends_excl: NDArray,
    expected: Dict,
) -> None:
    """
    Test NRowsSplitStrategy.partition_merge_regions method.

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
    # Create mock oars_desc with 6 regions
    oars_desc = ones(
        len(oars_desc_dict[RG_IDX_START]),
        dtype=[
            (RG_IDX_START, int_),
            (RG_IDX_END_EXCL, int_),
            (DF_IDX_END_EXCL, int_),
            (HAS_ROW_GROUP, bool_),
            (HAS_DF_CHUNK, bool_),
        ],
    )
    oars_desc[RG_IDX_START] = oars_desc_dict[RG_IDX_START]
    oars_desc[RG_IDX_END_EXCL] = oars_desc_dict[RG_IDX_END_EXCL]
    oars_desc[DF_IDX_END_EXCL] = oars_desc_dict[DF_IDX_END_EXCL]
    oars_desc[HAS_ROW_GROUP] = oars_desc_dict[HAS_ROW_GROUP]
    oars_desc[HAS_DF_CHUNK] = oars_desc_dict[HAS_DF_CHUNK]

    # Initialize strategy with target size of 100 rows
    strategy = NRowsSplitStrategy(
        drop_duplicates=drop_duplicates,
        rgs_n_rows=rgs_n_rows,
        oars_desc=oars_desc,
        row_group_target_size=row_group_target_size,
        max_n_off_target_rgs=1,
    )
    assert_array_equal(strategy.oars_min_n_rows, expected["oars_min_n_rows"])
    # Test partition_merge_regions
    result = strategy.partition_merge_regions(oar_idx_mrs_starts_ends_excl)
    # Check
    assert len(result) == len(expected["oars_merge_sequences"])
    for (result_rg_start, result_cmpt_ends_excl), (
        expected_rg_start,
        expected_cmpt_ends_excl,
    ) in zip(result, expected["oars_merge_sequences"]):
        assert result_rg_start == expected_rg_start
        print()
        print("result_cmpt_ends_excl")
        print(result_cmpt_ends_excl)
        print()
        print("expected_cmpt_ends_excl")
        print(expected_cmpt_ends_excl)
        assert_array_equal(result_cmpt_ends_excl, expected_cmpt_ends_excl)
