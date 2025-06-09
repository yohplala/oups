"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""

import pytest
from numpy import dtype
from pandas import DataFrame
from pandas import Timestamp

from oups import Store
from oups import toplevel
from oups.store.store.iter_intersections import _get_intersections
from oups.store.store.iter_intersections import iter_intersections


DTYPE_DATETIME64 = dtype("datetime64[ns]")
ORDERED_ON = "ts"


@toplevel
class Indexer:
    id: str


KEY1 = Indexer(id="key1")
KEY2 = Indexer(id="key2")
KEY3 = Indexer(id="key3")


# Test data
# 2 rows per row group, 3 keys
#   key1   key2   key3   rg_idx    oo_end_excl
#                       k1 k2 k3
#   8:00                 0
#         8:35              0
#         8:35
#         8:35              1      is collapsed with previous row group
#   9:00
#  10:00  10:00          1         10:00
#  10:00
#  10:00                 2         is collapsed with previous row group
#  10:00
#         12:10             2      12:10
#         12:10
#         12:10             3      is collapsed with previous row group
#         12:10
#  14:00  14:00          3  4      14:00
#  14:15
#         15:15  15:15         0
#                16:00
#         18:00             5      18:00
#  18:15         18:15   4     1   18:15
#  18:15
#                19:00
#         19:15
#                22:00         2   22:00
#                22:05
#                22:05         3   is collapsed with previous row group
#                22:05
@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path, Indexer)
    store[KEY1].write(
        ordered_on=ORDERED_ON,
        df=DataFrame(
            {
                ORDERED_ON: [
                    Timestamp(f"2025-01-01 {h}")
                    for h in (
                        ["08:00", "09:00", "10:00", "10:00", "10:00"]
                        + ["10:00", "14:00", "14:15", "18:15", "18:15"]
                    )
                ],
            },
        ),
        row_group_target_size=2,
    )
    store[KEY2].write(
        ordered_on=ORDERED_ON,
        df=DataFrame(
            {
                ORDERED_ON: [
                    Timestamp(f"2025-01-01 {h}")
                    for h in (
                        ["08:35", "08:35", "08:35", "10:00", "12:10", "12:10"]
                        + ["12:10", "12:10", "14:00", "15:15", "18:00", "19:15"]
                    )
                ],
            },
        ),
        row_group_target_size=2,
    )
    KEY3 = Indexer(id="key3")
    store[KEY3].write(
        ordered_on=ORDERED_ON,
        df=DataFrame(
            {
                ORDERED_ON: [
                    Timestamp(f"2025-01-01 {h}")
                    for h in (
                        ["15:15", "16:00", "18:15", "19:00", "22:00", "22:05", "22:05", "22:05"]
                    )
                ],
            },
        ),
        row_group_target_size=2,
    )
    return store


# Expected results.
ORDERED_ON_END_EXCL_REF = [
    Timestamp(f"2025-01-01 {h}") for h in ["10:00", "12:10", "14:00", "18:00", "18:15", "22:00"]
] + [None]
# Expected results when 'end_excl' is set and 'start' is None.
RG_IDX_END_EXCL_REF = [
    {KEY1: 1, KEY2: 2, KEY3: 1},  # 'end_excl' till 10:00
    {KEY1: 3, KEY2: 2, KEY3: 1},  # 'end_excl' till 12:10
    {KEY1: 3, KEY2: 4, KEY3: 1},  # 'end_excl' till 14:00
    {KEY1: 4, KEY2: 5, KEY3: 1},  # 'end_excl' till 18:00
    {KEY1: 4, KEY2: 6, KEY3: 1},  # 'end_excl' till 18:15
    {KEY1: 5, KEY2: 6, KEY3: 2},  # 'end_excl' till 22:00
    {KEY1: 5, KEY2: 6, KEY3: 4},  # 'end_excl' is None
]
# First 'rg_idx_end_excl' expected when 'start' is set.
RG_IDX_FIRST_END_EXCL_REF = [
    {KEY1: 1, KEY2: 2, KEY3: 1},  # 'start' till  9:00 included
    {KEY1: 3, KEY2: 2, KEY3: 1},  # 'start' till 10:00 included
    {KEY1: 4, KEY2: 4, KEY3: 1},  # 'start' till 12:10 included
    {KEY1: 4, KEY2: 5, KEY3: 1},  # 'start' till 14:00 included
    {KEY1: 5, KEY2: 5, KEY3: 1},  # 'start' till 15:15 included
    {KEY1: 5, KEY2: 6, KEY3: 1},  # 2nd row for 'start' till 15:15 included
]

INTERSECTIONS_AS_DF_REF = [
    {  # 0 - 10:00
        KEY1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 08:00"), Timestamp("2025-01-01 09:00")]},
        ),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 08:35")] * 3}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {  # 1 - 12:10
        KEY1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")] * 4}),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {  # 2 - 14:00
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 12:10")] * 4}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {  # 3 - 18:00
        KEY1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 14:00"), Timestamp("2025-01-01 14:15")]},
        ),
        KEY2: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 14:00"), Timestamp("2025-01-01 15:15")]},
        ),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 15:15"), Timestamp("2025-01-01 16:00")]},
        ),
    },
    {  # 4 - 18:15
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 18:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {  # 5 - 22:00
        KEY1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 18:15")] * 2}),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 19:15")]}),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 18:15"), Timestamp("2025-01-01 19:00")]},
        ),
    },
    {  # 6 - None
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 22:00")] + [Timestamp("2025-01-01 22:05")] * 3},
        ),
    },
]


@pytest.mark.parametrize(
    "test_id, full_test, start, end_excl, expected",
    [
        (
            "end_excl_10h00_on_edge",
            True,
            None,
            Timestamp("2025-01-01 10:00"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 10:00")],
                        RG_IDX_END_EXCL_REF[:1],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:1],
            },
        ),
        (
            "end_excl_10h05",
            True,
            None,
            Timestamp("2025-01-01 10:05"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 10:00"), Timestamp("2025-01-01 10:05")],
                        RG_IDX_END_EXCL_REF[:2],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:2],
            },
        ),
        (
            "end_excl_13h00",
            True,
            None,
            Timestamp("2025-01-01 13:00"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp(f"2025-01-01 {h}") for h in ["10:00", "12:10", "13:00"]],
                        RG_IDX_END_EXCL_REF[:3],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:3],
            },
        ),
        (
            "end_excl_16h00",
            True,
            None,
            Timestamp("2025-01-01 16:00"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[:3] + [Timestamp("2025-01-01 16:00")],
                        RG_IDX_END_EXCL_REF[:4],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:4],
            },
        ),
        (
            "end_excl_22h05",
            True,
            None,
            Timestamp("2025-01-01 22:05"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[:6] + [Timestamp("2025-01-01 22:05")],
                        RG_IDX_END_EXCL_REF[:7],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:7],
            },
        ),
        (
            "end_excl_none",
            True,
            None,
            None,
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF,
                        RG_IDX_END_EXCL_REF,
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF,
            },
        ),
        (
            "start_end_excl_09h00_14h00",
            True,
            Timestamp("2025-01-01 09:00"),
            Timestamp("2025-01-01 14:00"),
            {
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[0],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[:2] + [Timestamp("2025-01-01 14:00")],
                        RG_IDX_END_EXCL_REF[:3],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:3],
            },
        ),
        (  # Maybe the most representative test of intended use.
            "start_end_excl_14h00_14h05",
            True,
            Timestamp("2025-01-01 14:00"),
            Timestamp("2025-01-01 14:05"),
            {
                "rg_idx_starts": {KEY1: 3, KEY2: 4},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[3],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 14:05")],
                        [RG_IDX_FIRST_END_EXCL_REF[3]],
                    ),
                ),
                "intersections_as_df": [INTERSECTIONS_AS_DF_REF[3]],
            },
        ),
        (
            "start_10h00_on_edge",
            True,
            Timestamp("2025-01-01 10:00"),
            None,
            {
                "rg_idx_starts": {KEY1: 1, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[1],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[1:],
                        [RG_IDX_FIRST_END_EXCL_REF[1]] + RG_IDX_END_EXCL_REF[2:],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[1:],
            },
        ),
        (
            "start_10h05",
            True,
            Timestamp("2025-01-01 10:05"),
            None,
            {
                "rg_idx_starts": {KEY1: 3, KEY2: 2, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[2],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[2:],
                        [RG_IDX_FIRST_END_EXCL_REF[2]] + RG_IDX_END_EXCL_REF[3:],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[2:],
            },
        ),
        (
            "start_15h15",
            True,
            Timestamp("2025-01-01 15:15"),
            None,
            {
                "rg_idx_starts": {KEY1: 4, KEY2: 4, KEY3: 0},
                "rg_idx_first_ends_excl": RG_IDX_FIRST_END_EXCL_REF[4],
                "intersections_as_rg_idx": list(
                    zip(
                        ORDERED_ON_END_EXCL_REF[3:],
                        RG_IDX_FIRST_END_EXCL_REF[4:6] + RG_IDX_END_EXCL_REF[5:],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[3:],
            },
        ),
        (
            "edge_case_empty_same_row_groups",
            True,
            Timestamp("2025-01-01 10:10"),
            Timestamp("2025-01-01 10:20"),
            {
                "rg_idx_starts": {},
                "rg_idx_first_ends_excl": {},
                "intersections_as_rg_idx": [],
                "intersections_as_df": [],
            },
        ),
    ],
)
def test_iter_intersections(store, test_id, full_test, start, end_excl, expected):
    rg_idx_starts, rg_idx_first_ends_excl, rg_idx_intersections = _get_intersections(
        store=store,
        keys=list(store.keys),
        start=start,
        end_excl=end_excl,
    )
    rg_idx_intersections = list(rg_idx_intersections)
    assert rg_idx_starts == expected["rg_idx_starts"]
    for key in expected["rg_idx_starts"]:
        assert rg_idx_first_ends_excl[key] == expected["rg_idx_first_ends_excl"][key]
    for i, (ordered_on_end_excl_ref, rg_idx_end_excl_ref) in enumerate(
        expected["intersections_as_rg_idx"],
    ):
        assert ordered_on_end_excl_ref == rg_idx_intersections[i][0]
        for key in expected["rg_idx_starts"]:
            rg_idx_end_excl_res = rg_idx_intersections[i][1].pop(key)
            assert rg_idx_end_excl_ref[key] == rg_idx_end_excl_res
        assert rg_idx_intersections[i][1] == {}

    if full_test:
        dataset_intersections = list(
            iter_intersections(
                store=store,
                keys=list(store.keys),
                start=start,
                end_excl=end_excl,
            ),
        )
        n_intersections = len(expected["intersections_as_df"])
        if n_intersections == 0:
            assert dataset_intersections == []
            return
        for i, df_dict in enumerate(expected["intersections_as_df"]):
            for key, df_ref in df_dict.items():
                if key in expected["rg_idx_starts"]:
                    if i == 0 and start is not None:
                        df_ref = df_ref.set_index(ORDERED_ON).loc[start:].reset_index()
                    if i == n_intersections - 1 and end_excl is not None:
                        trim_end_idx = df_ref.loc[:, ORDERED_ON].searchsorted(end_excl, side="left")
                        df_ref = df_ref.iloc[:trim_end_idx]
                    df_res = dataset_intersections[i].pop(key)
                    assert df_ref.equals(df_res)
            assert dataset_intersections[i] == {}
