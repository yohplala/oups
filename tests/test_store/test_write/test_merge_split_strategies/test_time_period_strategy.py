#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from typing import Dict

import pytest
from numpy import array
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.write.merge_split_strategies import TimePeriodMergeSplitStrategy


RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS = "rg_idx_ends_excl_not_to_use_as_split_points"
REF_D = "2020/01/01 "


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
                "rg_idx_mrs_starts_ends_excl": [slice(0, 1)],
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
                "rg_idx_mrs_starts_ends_excl": [slice(0, 1)],
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
                "rg_idx_mrs_starts_ends_excl": [slice(0, 1), slice(2, 3)],
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
                "rg_idx_mrs_starts_ends_excl": [slice(0, 2)],
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
                "rg_idx_mrs_starts_ends_excl": [slice(0, 1)],
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
    result = strategy._specialized_compute_merge_sequences()
    # Check
    assert strategy.rg_idx_mrs_starts_ends_excl == expected["rg_idx_mrs_starts_ends_excl"]
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
def test_time_period_compute_split_sequence(test_id, df_dates, target_period, expected_offsets):
    # Initialize strategy
    strategy = TimePeriodMergeSplitStrategy(
        rg_ordered_on_mins=Series([Timestamp("2024/01/01 04:00:00")]),  # dummy value
        rg_ordered_on_maxs=Series([Timestamp("2024/01/05 14:00:00")]),  # dummy value
        df_ordered_on=Series(Timestamp("2024/01/06 04:00:00")),  # dummy value
        row_group_time_period=target_period,
    )
    # Get offsets
    offsets = strategy.compute_split_sequence(df_ordered_on=Series(df_dates))
    # Verify results
    assert offsets == expected_offsets


@pytest.mark.parametrize(
    "test_id, rg_mins, rg_maxs, df_ordered_on, row_group_time_period, drop_duplicates, rgs_n_rows, max_n_off_target, expected",
    [
        (  # Values not on boundary to check 'floor()'.
            # Writing after pf data, no off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [12h10]
            "new_rg_simple_append_timestamp_not_on_boundary",
            array([Timestamp(f"{REF_D}08:10"), Timestamp(f"{REF_D}10:10")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:10"), Timestamp(f"{REF_D}10:10")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}12:10")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge incomplete rg
            False,  # drop_duplicates | should not merge with preceding rg
            array([2, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Values not on boundary to check 'floor()'.
            # Writing after pf data, not merging with off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:                      [10h10]
            "no_drop_duplicates_simple_append_timestamp_not_on_boundary",
            array([Timestamp(f"{REF_D}08:10"), Timestamp(f"{REF_D}10:10")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:10"), Timestamp(f"{REF_D}10:10")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}10:10")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge incomplete rg
            False,  # drop_duplicates | should not merge with preceding rg
            array([2, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Values not on boundary to check 'floor()'.
            # Writing after pf data, merging with off target size row group.
            # rg:  0            1
            # pf: [8h10,9h10], [10h10]
            # df:              [10h10]
            "drop_duplicates_merge_tail_timestamp_not_on_boundary",
            array([Timestamp(f"{REF_D}08:10"), Timestamp(f"{REF_D}10:10")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:10"), Timestamp(f"{REF_D}10:10")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}10:10")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge incomplete rg
            True,  # drop_duplicates | should merge with incomplete rg
            array([2, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Values on boundary.
            # Writing after pf data, not merging with off target size row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:                      [10h00]
            "no_drop_duplicates_simple_append_timestamp_on_boundary",
            array([Timestamp(f"{REF_D}08:00"), Timestamp(f"{REF_D}10:00")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:00"), Timestamp(f"{REF_D}10:00")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}10:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge incomplete rg
            False,  # drop_duplicates | should not merge with preceding rg
            array([2, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Values on boundary.
            # Writing after pf data, merging with off target size row group.
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:              [10h00]
            "drop_duplicates_merge_tail_timestamp_on_boundary",
            array([Timestamp(f"{REF_D}08:00"), Timestamp(f"{REF_D}10:00")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:00"), Timestamp(f"{REF_D}10:00")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}10:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge incomplete rg
            True,  # drop_duplicates | should merge with incomplete rg
            array([2, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # Writing after pf data, off target size row group should be merged.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [13h00]
            "last_row_group_exceeded_merge_tail",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}13:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | new period, should merge incomplete rgs
            True,  # drop_duplicates | no duplicates to drop
            array([2, 1, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite incomplete rg
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[3, 0], [3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # df at the start of pf data.
            # df is not overlapping with existing row groups.
            # rg:           0            1        2
            # pf:          [8h00,9h00], [12h00], [13h00]
            # df:  [7h30]
            "no_duplicates_insert_at_start_new_rg_timestamp_not_on_boundary",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}13:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}13:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}07:30")]),  # df_ordered_on
            "2h",  # row_group_time_period | no rg in same period to merge with
            True,  # drop_duplicates | no duplicates to drop
            array([2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[0, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # df at the start of pf data.
            # df is overlapping with time period of existing row groups.
            # rg:            0            1        2
            # pf:           [8h10,9h10], [12h10], [13h10]
            # df:           [8h00]
            "no_duplicates_insert_at_start_no_new_rg_timestamp_on_boundary",
            array(
                [
                    Timestamp(f"{REF_D}08:10"),
                    Timestamp(f"{REF_D}12:10"),
                    Timestamp(f"{REF_D}13:10"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:10"),
                    Timestamp(f"{REF_D}12:10"),
                    Timestamp(f"{REF_D}13:10"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}08:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should merge with rg in same period
            True,  # drop_duplicates | no duplicates to drop
            array([2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(0, array([[1, 1], [3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # df connected to off target size rgs. Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_not_reached_simple_append",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}11:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not rewrite tail
            False,  # drop_duplicates | should not merge with preceding rg
            array([2, 1, 1]),  # rgs_n_rows
            3,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(3, array([[3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # df connected to off target size rgs. Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_reached_tail_rewrite",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}11:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge with incomplete rg.
            False,  # drop_duplicates | should not merge with preceding rg
            array([2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # df connected to off target size rgs. Values on boundary.
            # Writing after pf data, off target size row groups.
            # rg:  0            1        2
            # pf: [8h00,9h00], [10h00], [11h00]
            # df:                               [11h00]
            "max_n_off_target_rgs_none_simple_append",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}11:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}11:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not merge with incomplete rg.
            False,  # drop_duplicates
            array([2, 1, 1]),  # rgs_n_rows
            None,  # max_n_off_target_rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(3, array([[3, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
        (  # 'max_n_off_target_rgs' not reached.
            # rg:  0            1                 2
            # pf: [8h00,9h00], [10h00],          [13h00]
            # df:                       [12h00]
            "insert_timestamp_max_n_off_target_rgs_tail_rewrite",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}13:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}13:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}12:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not specifically rewrite tail
            True,  # drop_duplicates
            array([2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs | should not rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # df is not overlapping with existing row groups. It should be added.
            # rg:  0                   1        2
            # pf: [8h00,9h00],        [12h00], [14h00]
            # df:             [10h30]
            "insert_timestamp_non_overlapping",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}14:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}14:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}10:30")]),  # df_ordered_on
            "2h",  # row_group_time_period
            True,  # drop_duplicates
            array([2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[1, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # rg:  0        1           2           3      4
            # pf: [8h,9h], [10h, 11h], [12h, 13h], [14h], [15h]
            # df:               [11h]
            "insert_timestamp_middle_with_off_target_rgs",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}14:00"),
                    Timestamp(f"{REF_D}15:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}11:00"),
                    Timestamp(f"{REF_D}13:00"),
                    Timestamp(f"{REF_D}14:00"),
                    Timestamp(f"{REF_D}15:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}11:00")]),  # df_ordered_on
            "2h",  # row_group_size
            True,  # drop_duplicates
            array([2, 2, 2, 1, 1]),  # rgs_n_rows
            1,  # max_n_off_target_rgs | should rewrite tail
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[2, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # df within pf data.
            # Writing in-between pf data, with off target size row groups at
            # the end of pf data.
            # rg:  0          1           2           3      4
            # pf: [8h,9h],   [10h, 11h], [12h, 13h], [14h], [15h]
            # df:        [9h]
            "insert_timestamp_middle_no_rewrite",
            array(
                [
                    Timestamp(f"{REF_D}08:00"),
                    Timestamp(f"{REF_D}10:00"),
                    Timestamp(f"{REF_D}12:00"),
                    Timestamp(f"{REF_D}14:00"),
                    Timestamp(f"{REF_D}15:00"),
                ],
            ),  # rg_mins
            array(
                [
                    Timestamp(f"{REF_D}09:00"),
                    Timestamp(f"{REF_D}11:00"),
                    Timestamp(f"{REF_D}13:00"),
                    Timestamp(f"{REF_D}14:00"),
                    Timestamp(f"{REF_D}15:00"),
                ],
            ),  # rg_maxs
            Series([Timestamp(f"{REF_D}09:00")]),  # df_ordered_on
            "2h",  # row_group_time_period
            False,  # drop_duplicates
            array([2, 2, 2, 1, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(1, array([[1, 1]]))],
                "sort_rgs_after_write": True,
            },
        ),
        (  # Several empty periods, to make sure the empty periods are not in
            # the output
            # rg:  0            1
            # pf: [8h00,9h00], [10h00]
            # df:                      [23h00]
            "insert_timestamp_several_empty_periods",
            array([Timestamp(f"{REF_D}08:00"), Timestamp(f"{REF_D}10:00")]),  # rg_mins
            array([Timestamp(f"{REF_D}09:00"), Timestamp(f"{REF_D}10:00")]),  # rg_maxs
            Series([Timestamp(f"{REF_D}23:00")]),  # df_ordered_on
            "2h",  # row_group_time_period | should not specifically rewrite tail
            True,  # drop_duplicates
            array([2, 1]),  # rgs_n_rows
            2,  # max_n_off_target_rgs
            {
                RG_IDX_ENDS_EXCL_NOT_TO_USE_AS_SPLIT_POINTS: None,
                "oars_merge_sequences": [(2, array([[2, 1]]))],
                "sort_rgs_after_write": False,
            },
        ),
    ],
)
def test_time_period_integration_compute_merge_sequences(
    test_id: str,
    rg_mins: NDArray,
    rg_maxs: NDArray,
    df_ordered_on: NDArray,
    row_group_time_period: str,
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
    row_group_time_period : str
        Time period for row groups.
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
    strategy = TimePeriodMergeSplitStrategy(
        rg_ordered_on_mins=rg_mins,
        rg_ordered_on_maxs=rg_maxs,
        df_ordered_on=df_ordered_on,
        row_group_time_period=row_group_time_period,
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
