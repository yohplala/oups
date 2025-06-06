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
from oups.store.store.iter_row_groups import _get_intersections
from oups.store.store.iter_row_groups import iter_row_groups


DTYPE_DATETIME64 = dtype("datetime64[ns]")
FULL_TEST = False
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
INTERSECTIONS_AS_RG_IDX_REF = {
    "ordered_on_end_excl": [
        Timestamp(f"2025-01-01 {h}") for h in ["10:00", "12:10", "14:00", "18:00", "18:15", "22:00"]
    ]
    + [None],
    "rg_idx_end_excl": [
        {KEY1: 1, KEY2: 2, KEY3: 1},  # 10:00
        {KEY1: 3, KEY2: 2, KEY3: 1},  # 12:10
        {KEY1: 3, KEY2: 4, KEY3: 1},  # 14:00
        {KEY1: 4, KEY2: 5, KEY3: 1},  # 18:00
        {KEY1: 4, KEY2: 6, KEY3: 1},  # 18:15
        {KEY1: 5, KEY2: 6, KEY3: 2},  # 22:00
        {KEY1: 5, KEY2: 6, KEY3: 4},  # None
    ],
}

INTERSECTIONS_AS_DF_REF = [
    {
        KEY1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 08:00"), Timestamp("2025-01-01 09:00")]},
        ),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 08:35")] * 3}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")] * 4}),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")] * 4}),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 12:10")] * 4}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 14:00"), Timestamp("2025-01-01 14:15")]},
        ),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 14:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 15:15")]}),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 15:15"), Timestamp("2025-01-01 16:00")]},
        ),
    },
    {
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 18:00")]}),
        KEY3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        KEY1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 18:15"), Timestamp("2025-01-01 18:15")]},
        ),
        KEY2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 19:15")]}),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 18:15"), Timestamp("2025-01-01 19:00")]},
        ),
    },
    {
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 22:00"), Timestamp("2025-01-01 22:05")]},
        ),
    },
    {
        KEY1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY2: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        KEY3: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 22:05"), Timestamp("2025-01-01 22:05")]},
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
                "start": Timestamp("2025-01-01 08:00"),
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 10:00")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:1],
                    ),
                ),
                "intersections_as_df": INTERSECTIONS_AS_DF_REF[:1],
            },
        ),
        (
            "end_excl_10h05",
            False,
            None,
            Timestamp("2025-01-01 10:05"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 10:00"), Timestamp("2025-01-01 10:05")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:2],
                    ),
                ),
            },
        ),
        (
            "end_excl_13h00",
            False,
            None,
            Timestamp("2025-01-01 13:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "rg_idx_starts": {KEY1: 0, KEY2: 0},
                "rg_idx_first_ends_excl": INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][0],
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp(f"2025-01-01 {h}") for h in ["10:00", "12:10", "13:00"]],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:3],
                    ),
                ),
            },
        ),
        (
            "end_excl_16h00",
            False,
            None,
            Timestamp("2025-01-01 16:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "rg_idx_starts": {KEY1: 0, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][0],
                "intersections_as_rg_idx": list(
                    zip(
                        INTERSECTIONS_AS_RG_IDX_REF["ordered_on_end_excl"][:3]
                        + [Timestamp("2025-01-01 16:00")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:4],
                    ),
                ),
            },
        ),
        (
            "end_excl_22h05",
            False,
            None,
            Timestamp("2025-01-01 22:05"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "rg_idx_starts": {KEY1: 0, KEY2: 0, KEY3: 0},
                "rg_idx_first_ends_excl": INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][0],
                "intersections_as_rg_idx": list(
                    zip(
                        INTERSECTIONS_AS_RG_IDX_REF["ordered_on_end_excl"][:6]
                        + [Timestamp("2025-01-01 22:05")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:7],
                    ),
                ),
            },
        ),
    ],
)
def test_get_intersections(store, test_id, full_test, start, end_excl, expected):
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
        print("i ", i)
        assert ordered_on_end_excl_ref == rg_idx_intersections[i][0]
        for key in expected["rg_idx_starts"]:
            assert rg_idx_end_excl_ref[key] == rg_idx_intersections[i][1][key]

    if full_test:
        dataset_intersections = list(
            iter_row_groups(
                store=store,
                keys=list(store.keys),
                start=start,
                end_excl=end_excl,
            ),
        )
        n_intersections = len(expected["intersections_as_df"])
        for i, df_dict in enumerate(expected["intersections_as_df"]):
            for key, df_ref in df_dict.items():
                if key in expected["rg_idx_starts"]:
                    if i == 0:
                        df_ref = df_ref.set_index(ORDERED_ON).iloc[start:].reset_index()
                        print("df_ref after start trimming")
                        print(df_ref)
                    if i == n_intersections - 1 and end_excl is not None:
                        print("df_ref before end trimming")
                        print(df_ref)
                        trim_end_idx = df_ref.loc[:, ORDERED_ON].searchsorted(end_excl, side="left")
                        print("trim_end_idx")
                        print(trim_end_idx)
                        df_ref = df_ref.iloc[:trim_end_idx]
                        print("df_ref after end trimming")
                        print(df_ref)
                    assert df_ref.equals(dataset_intersections[i][key])
