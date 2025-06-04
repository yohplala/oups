"""
Created on Mon Jun 02 18:35:00 2025.

@author: yoh

"""

import pytest
from pandas import DataFrame
from pandas import Timestamp

from oups import Store
from oups import toplevel
from oups.store.store.iter_row_groups import _get_intersections


ORDERED_ON = "ts"


@toplevel
class Indexer:
    id: str


key1 = Indexer(id="key1")
key2 = Indexer(id="key2")
key3 = Indexer(id="key3")


# Test data
# 2 rows per row group, 3 keys
#   key1   key2   key3   consolidated
#   8:00   8:35
#   9:00
#  10:00  10:00
#  10:00
#  10:00
#  10:00
#         12:10
#         12:10
#         12:10
#         12:10
#  14:00  14:00
#  14:15
#         15:15  15:15
#                16:00
#  18:15  18:00  18:15
#  18:15  19:15  19:00
#                22:00
#                22:05
#                22:05
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
                    for h in [
                        "08:00",
                        "09:00",
                        "10:00",
                        "10:00",
                        "10:00",
                        "10:00",
                        "14:00",
                        "14:15",
                        "18:15",
                        "18:15",
                    ]
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
                    for h in [
                        "08:35",
                        "10:00",
                        "12:10",
                        "12:10",
                        "12:10",
                        "12:10",
                        "14:00",
                        "15:15",
                        "18:00",
                        "19:15",
                    ]
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
                    for h in [
                        "15:15",
                        "16:00",
                        "18:15",
                        "19:00",
                        "22:00",
                        "22:05",
                        "22:05",
                        "22:05",
                    ]
                ],
            },
        ),
        row_group_target_size=2,
    )
    return store


@pytest.mark.parametrize(
    "test_id, start, end_excl, expected",
    [
        (
            "end_excl_10h05",
            None,
            Timestamp("2025-01-01 10:05"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
                "intersection_df": list(
                    zip(
                        [Timestamp(f"2025-01-01 {h}") for h in ["10:00", "10:05"]],
                        [{key1: 1, key2: 1}, {key1: 3, key2: 1}],
                    ),
                ),
            },
        ),
        (
            "end_excl_10h00_on_edge",
            None,
            Timestamp("2025-01-01 10:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
                "intersection_df": list(
                    zip(
                        [Timestamp("2025-01-01 10:00")],
                        [{key1: 1, key2: 1}],
                    ),
                ),
            },
        ),
        (
            "end_excl_13h00",
            None,
            Timestamp("2025-01-01 13:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0},
                "intersection_df": list(
                    zip(
                        [Timestamp(f"2025-01-01 {h}") for h in ["10:00", "12:10", "13:00"]],
                        [{key1: 1, key2: 1}, {key1: 3, key2: 1}, {key1: 3, key2: 3}],
                    ),
                ),
            },
        ),
        (
            "end_excl_16h00",
            None,
            Timestamp("2025-01-01 16:00"),
            {
                "start": Timestamp("2025-01-01 08:00"),
                "first_rg_indices": {key1: 0, key2: 0, key3: 0},
                "intersection_df": list(
                    zip(
                        [
                            Timestamp(f"2025-01-01 {h}")
                            for h in ["10:00", "12:10", "14:00", "16:00"]
                        ],
                        [
                            {key1: 1, key2: 1, key3: 1},
                            {key1: 3, key2: 1, key3: 1},
                            {key1: 3, key2: 3, key3: 1},
                            {key1: 4, key2: 4, key3: 1},
                        ],
                    ),
                ),
            },
        ),
    ],
)
def test_get_intersections(store, test_id, start, end_excl, expected):
    first_rg_indices, intersections = _get_intersections(
        store=store,
        keys=list(store.keys),
        start=start,
        end_excl=end_excl,
    )
    assert first_rg_indices == expected["first_rg_indices"]
    assert list(intersections) == expected["intersection_df"]
