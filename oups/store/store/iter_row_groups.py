#!/usr/bin/env python3
"""
Created on Mon May 26 18:00:00 2025.

@author: yoh

"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

from numpy import arange
from numpy import full
from numpy import insert
from numpy import nan
from numpy import put
from numpy import searchsorted
from pandas import DataFrame
from pandas import Int64Dtype
from pandas import Series
from pandas import Timestamp

from oups.defines import KEY_ORDERED_ON_MAXS
from oups.defines import KEY_ORDERED_ON_MINS
from oups.numpy_utils import isnotin_ordered


KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ENDS_EXCL = "ends_excl"


def _get_and_validate_ordered_on_column(store, keys: List[dataclass]) -> str:
    """
    Get and validate the 'ordered_on' column name across all datasets.

    Parameters
    ----------
    store : Store
        Store instance containing the datasets.
    keys : List[dataclass]
        List of dataset keys to validate.

    Returns
    -------
    str
        The common 'ordered_on' column name.

    Raises
    ------
    ValueError
        If ordered_on column names differ between datasets.

    """
    # Get 'ordered_on' from first key.
    ordered_on_col = store[keys[0]].ordered_on
    # Validate all keys have the same 'ordered_on' column.
    for key in keys[1:]:
        if store[key].ordered_on != ordered_on_col:
            raise ValueError(
                f"inconsistent 'ordered_on' columns. '{keys[0]}' has "
                f"'{ordered_on_col}', but '{key}' has '{store[key].ordered_on}'.",
            )
    return ordered_on_col


def _get_intersection_iterator(
    store,  # Store instance
    keys: List[dataclass],
    start: Optional[Union[int, float, Timestamp]] = None,
    end_excl: Optional[Union[int, float, Timestamp]] = None,
) -> Tuple[Union[int, float, Timestamp], Dict[dataclass, int], Iterator[tuple]]:
    """
    Create an iterator over intersection boundaries with row group indices.

    This function analyzes row group statistics across all keys to determine
    intersection boundaries and the corresponding row group indices for each key.
    Returns the global minimum ordered_on value and yields intersection boundaries.

    Parameters
    ----------
    store : Store
        Store instance containing the datasets.
    keys : List[dataclass]
        List of dataset keys to synchronize.
    start : Optional[Union[int, float, Timestamp]], default None
        Start value for the 'ordered_on' column range.
    end_excl : Optional[Union[int, float, Timestamp]], default None
        End value (exclusive) for the 'ordered_on' column range.

    Returns
    -------
    tuple[Union[int, float, Timestamp], Dict[dataclass, int], Iterator[tuple]]
        Tuple containing:
        - Global minimum 'ordered_on' value across all keys,
        - Dictionary mapping each key to its starting row group index,
        - Iterator yielding (current_end_excl, rg_indices) tuples where:
          * current_end_excl: End boundary (exclusive) for current intersection
          * rg_indices: Dict mapping each key to its row group index for this
            intersection

    Notes
    -----
    Algorithm:
    1. For each key, load ordered_on_mins Series (index = row_group_idx)
    2. Find global minimum across all keys
    3. Collect unique boundary values and sort them
    4. Filter boundaries by start/end_excl and ensure end_excl is final boundary
    5. For each intersection, determine active row group index per key

    """
    print()
    keys_ordered_on_mins = {
        key: store[key].row_group_stats.loc[:, KEY_ORDERED_ON_MINS].to_numpy() for key in keys
    }
    keys_ordered_on_maxs = {
        key: store[key].row_group_stats.loc[:, KEY_ORDERED_ON_MAXS].to_numpy() for key in keys
    }
    defaultdict_0 = defaultdict(lambda: 0)
    rg_idx_starts = (
        {key: searchsorted(keys_ordered_on_mins[key], start, side=KEY_LEFT) for key in keys}
        if start
        else defaultdict_0
    )
    defaultdict_none = defaultdict(lambda: None)
    rg_idx_ends_excl = (
        {key: searchsorted(keys_ordered_on_maxs[key], end_excl, side=KEY_RIGHT) for key in keys}
        if end_excl
        else defaultdict_none
    )
    # Collect 'ordered_on_mins' for each key.
    ordered_on_starts = keys_ordered_on_mins[keys[0]][
        rg_idx_starts[keys[0]] : rg_idx_ends_excl[keys[0]]
    ]
    for key, ordered_on_mins in keys_ordered_on_mins.items():
        # Identify duplicates in ordered_on_mins.
        unique_mask, insert_idx = isnotin_ordered(
            ordered_on_starts,
            ordered_on_mins[rg_idx_starts[key] : rg_idx_ends_excl[key]],
        )
        print("key ", key)
        print("insert_idx")
        print(insert_idx)
        print("unique_mask")
        print(unique_mask)
        ordered_on_starts = insert(
            ordered_on_starts,
            insert_idx[unique_mask],
            ordered_on_mins[unique_mask],
        )
    print("ordered_on_starts")
    print(ordered_on_starts)
    # DataFrame to hold 'ends_excl' values & row group indices.
    len_intersection_df = len(ordered_on_starts)
    keys_rg_indices = {}
    for key, ordered_on_mins in keys_ordered_on_mins.items():
        rg_indices = full(len_intersection_df, nan)
        put(
            rg_indices,
            searchsorted(
                ordered_on_starts,
                ordered_on_mins[rg_idx_starts[key] : rg_idx_ends_excl[key]],
                side=KEY_LEFT,
            ),
            arange(rg_idx_starts[key], rg_idx_ends_excl[key]),
        )
        print("key ", key)
        print("rg_indices")
        print(rg_indices)
        keys_rg_indices[str(key)] = Series(rg_indices, dtype=Int64Dtype()).ffill()

    intersections = DataFrame(
        {
            KEY_ENDS_EXCL: Series(ordered_on_starts).shift(-1, fill_value=end_excl),
            **keys_rg_indices,
        },
    )

    #    def intersection_iter():
    #        for row_dict in intersection_df.to_dict(orient="records", index=False):
    #            yield row_dict.pop(KEY_ENDS_EXCL), {store.indexer.from_str(key_str): val for key_str, val in row_dict.items()}

    return ordered_on_starts[0], rg_idx_starts, intersections


def _initialize_start_indices(
    ordered_on_col_name: str,
    in_memory_data: Dict[dataclass, DataFrame],
    current_start: Union[int, float, Timestamp],
) -> Dict[dataclass, int]:
    """
    Initialize start indices for all keys based on current_start value.

    Parameters
    ----------
    ordered_on_col_name : str
        Name of the ordered_on column.
    in_memory_data : Dict[dataclass, DataFrame]
        Cached DataFrames by key.
    current_start : Union[int, float, Timestamp]
        Start value to search for.

    Returns
    -------
    Dict[dataclass, int]
        Dictionary mapping each key to its start index.

    """
    return {
        key: value.loc[:, ordered_on_col_name].searchsorted(current_start, side=KEY_LEFT)
        for key, value in in_memory_data.items()  # value is a DataFrame
    }


def iter_row_groups(
    store,  # Store instance
    keys: List[dataclass],
    start: Optional[Union[int, float, Timestamp]] = None,
    end_excl: Optional[Union[int, float, Timestamp]] = None,
) -> Iterator[Dict[dataclass, DataFrame]]:
    """
    Iterate over synchronized row groups across multiple datasets in the store.

    This function yields data from multiple datasets (keys) in synchronized
    chunks, ensuring that all returned DataFrames share overlapping spans
    in their 'ordered_on' column. This allows processing 'ordered_on'-aligned
    data from multiple sources.

    Parameters
    ----------
    store : Store
        Store instance containing the datasets to iterate over.
    keys : List[dataclass]
        List of dataset keys to iterate over. All keys must have datasets
        with the same 'ordered_on' column name.
    start : Optional[Union[int, float, Timestamp]], default None
        Start value (inclusive) for the 'ordered_on' column range. If None,
        starts from the earliest value across all specified keys.
    end_excl : Optional[Union[int, float, Timestamp]], default None
        End value (exclusive) for the 'ordered_on' column range. If None,
        continues until the latest value across all specified keys.

    Yields
    ------
    Dict[dataclass, DataFrame]
        Dictionary mapping each key to its corresponding DataFrame chunk.
        All DataFrames in each yielded dictionary share a common span in their
        'ordered_on' column using [start, end_excl) semantics (start inclusive,
        end exclusive).

    Notes
    -----
    - All datasets must have an 'ordered_on' column with the same name.
    - The iteration is synchronized: each yield contains data from the same
      span across all datasets.
    - Uses [start, end_excl) interval semantics throughout.

    Examples
    --------
    >>> store = Store(...)
    >>> store[key1].write(data1, ordered_on='timestamp')
    >>> store[key2].write(data2, ordered_on='timestamp')
    >>> for data_dict in store.iter_row_groups([key1, key2], start="2022-01-01"):
    ...     df1 = data_dict[key1]  # DataFrame for key1
    ...     df2 = data_dict[key2]  # DataFrame for key2
    ...     # Process synchronized data

    """
    # Get and validate ordered_on column name.
    ordered_on_col_name = _get_and_validate_ordered_on_column(store, keys)
    # Get global minimum and intersection boundary iterator and initialize
    # the "previous" (the first) row group indices.
    global_min, prev_rg_indices, intersection_iter = _get_intersection_iterator(
        store,
        keys,
        start,
        end_excl,
    )
    # Initialize state tracking.
    current_start = start if start is not None else global_min
    # Load initial row groups and initialize start indices.
    in_memory_data = {key: store[key][prev_rg_indices[key]].to_pandas() for key in keys}
    current_start_indices = _initialize_start_indices(
        ordered_on_col_name,
        in_memory_data,
        current_start,
    )
    current_end_indices = {}
    for current_end_excl, rg_indices in intersection_iter:
        for key in keys:
            if rg_indices[key] != prev_rg_indices[key]:
                in_memory_data[key] = store[key][rg_indices[key]].to_pandas()
                prev_rg_indices[key] = rg_indices[key]
                # Reset start index to 0 for new row group.
                current_start_indices[key] = 0
            # Calculate end indices for current_end_excl.
            current_end_indices[key] = (
                None
                if current_end_excl is None
                else in_memory_data[key]
                .loc[:, ordered_on_col_name]
                .searchsorted(
                    current_end_excl,
                    side=KEY_LEFT,
                )
            )
        # Extract synchronized views for current span.
        # [current_start, current_end_excl)
        yield {
            key: in_memory_data[key].iloc[
                current_start_indices[key] : current_end_indices[key],
                :,
            ]
            for key in keys
        }
        # Buffer end indices for next iteration as start indices.
        current_start_indices = current_end_indices.copy()


# TODO: implement 'cache_opds' in Store class?
# could the cache be removed, in the way of a context manager:
# with store.workspace() as ws:
# for each getitem in store if odp is not in keys, it is added and kept
# the should be a lock system to avoid race conditions
# lock when reading operation: then only write allowed
# lock when modifying row group stats vs file_id (remove_rgs & align_fids)
# if the operation makes that fis in metadata file lead to unexisting or wrong file
# any other operations is forbidden
# if the operation does not corrupt correspondence between fids in metadata file
# and row group file on disk, then the lock does not prevent other read operations
# there are 2 locks: soft lock and hard lock
