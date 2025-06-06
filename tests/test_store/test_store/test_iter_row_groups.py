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


key1 = Indexer(id="key1")
key2 = Indexer(id="key2")
key3 = Indexer(id="key3")


# Test data
# 2 rows per row group, 3 keys
#   key1   key2   key3   rg_idx    oo_end_excl
#                       k1 k2 k3
#   8:00                 0
#         8:35              0
#   9:00
#  10:00  10:00          1         10:00
#  10:00
#  10:00                 2
#  10:00
#         12:10             1      12:10
#         12:10
#         12:10             2
#         12:10
#  14:00  14:00          3  3      14:00
#  14:15
#         15:15  15:15         0   15:15
#                16:00
#         18:00             4      18:00
#  18:15         18:15   4     1   18:15
#  18:15
#                19:00
#         19:15
#                22:00         2   22:00
#                22:05
#                22:05         3   22:05
#                22:05
@pytest.fixture()
def store(tmp_path):
    store = Store(tmp_path, Indexer)
    store[key1].write(
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
    store[key2].write(
        ordered_on=ORDERED_ON,
        df=DataFrame(
            {
                ORDERED_ON: [
                    Timestamp(f"2025-01-01 {h}")
                    for h in (
                        ["08:35", "10:00", "12:10", "12:10", "12:10"]
                        + ["12:10", "14:00", "15:15", "18:00", "19:15"]
                    )
                ],
            },
        ),
        row_group_target_size=2,
    )
    key3 = Indexer(id="key3")
    store[key3].write(
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
        Timestamp(f"2025-01-01 {h}")
        for h in ["10:00", "12:10", "14:00", "15:15", "18:00", "18:15", "22:00", "22:05"]
    ]
    + [None],
    "rg_idx_end_excl": [
        {key1: 1, key2: 1, key3: 1},  # 10:00
        {key1: 3, key2: 1, key3: 1},  # 12:10
        {key1: 3, key2: 3, key3: 1},  # 14:00
        {key1: 4, key2: 4, key3: 1},  # 15:15
        {key1: 4, key2: 4, key3: 1},  # 18:00
        {key1: 4, key2: 5, key3: 1},  # 18:15
        {key1: 5, key2: 5, key3: 2},  # 22:00
        {key1: 5, key2: 5, key3: 3},  # 22:05
        {key1: 5, key2: 5, key3: 4},  # None
    ],
}

INTERSECTIONS_AS_DF_REF = [
    {
        key1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 08:00"), Timestamp("2025-01-01 09:00")]},
        ),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 08:35")]}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        key1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")] * 4}),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")]}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        key1: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")] * 4}),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 10:00")]}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        key1: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 12:10")] * 4}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        key1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 14:00"), Timestamp("2025-01-01 14:15")]},
        ),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 14:00")]}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
    {
        key1: DataFrame(
            {ORDERED_ON: [Timestamp("2025-01-01 14:00"), Timestamp("2025-01-01 14:15")]},
        ),
        key2: DataFrame({ORDERED_ON: [Timestamp("2025-01-01 14:00")]}),
        key3: DataFrame({ORDERED_ON: []}, dtype=DTYPE_DATETIME64),
    },
]


@pytest.mark.parametrize(
    "test_id, full_test, start, end_excl, expected",
    [
        (
            "end_excl_10h00_on_edge",
            FULL_TEST,
            None,
            Timestamp("2025-01-01 10:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
                "intersections_as_rg_idx": list(
                    zip(
                        [Timestamp("2025-01-01 10:00")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:1],
                    ),
                ),
            },
        ),
        (
            "end_excl_10h05",
            FULL_TEST,
            None,
            Timestamp("2025-01-01 10:05"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
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
            FULL_TEST,
            None,
            Timestamp("2025-01-01 13:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
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
            FULL_TEST,
            None,
            Timestamp("2025-01-01 16:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0, key3: 0},
                "intersections_as_rg_idx": list(
                    zip(
                        INTERSECTIONS_AS_RG_IDX_REF["ordered_on_end_excl"][:3]
                        + [Timestamp("2025-01-01 16:00")],
                        INTERSECTIONS_AS_RG_IDX_REF["rg_idx_end_excl"][:4],
                    ),
                ),
            },
        ),
    ],
)
def test_get_intersections(store, test_id, full_test, start, end_excl, expected):
    first_rg_indices, intersections = _get_intersections(
        store=store,
        keys=list(store.keys),
        start=start,
        end_excl=end_excl,
    )
    intersections = list(intersections)
    assert first_rg_indices == expected["first_rg_indices"]
    for i, (ordered_on_end_excl_ref, rg_idx_end_excl_ref) in enumerate(
        expected["intersections_as_rg_idx"],
    ):
        assert ordered_on_end_excl_ref == intersections[i][0]
        for key in expected["first_rg_indices"]:
            assert rg_idx_end_excl_ref[key] == intersections[i][1][key]

    if full_test:
        dataset_intersections = list(
            iter_row_groups(
                store=store,
                keys=list(store.keys),
                start=start,
                end_excl=end_excl,
            ),
        )
        for i, df_dict in enumerate(expected["intersections_as_df"]):
            for key, df_ref in df_dict.items():
                if key in expected["first_rg_indices"]:
                    assert df_ref.equals(dataset_intersections[i][key])
