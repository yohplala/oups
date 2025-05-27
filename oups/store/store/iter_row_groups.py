#!/usr/bin/env python3
"""
Created on Mon May 26 18:00:00 2025.

@author: yoh

"""
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

from pandas import DataFrame
from pandas import Timestamp


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
) -> Iterator[tuple]:
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
    tuple[Union[int, float, Timestamp], Iterator[tuple]]
        Tuple containing:
        - Global minimum ordered_on value across all keys
        - Iterator yielding (current_end_excl, rg_indices) tuples

    Notes
    -----
    Algorithm:
    1. For each key, load ordered_on_mins Series (index = row_group_idx)
    2. Find global minimum across all keys
    3. Collect unique boundary values and sort them
    4. Filter boundaries by start/end_excl and ensure end_excl is final boundary
    5. For each intersection, determine active row group index per key

    """
    # 1. Collect ordered_on_mins for each key and find global minimum
    # key_mins = {}
    # all_mins = []
    # for key in keys:
    #     ordered_on_mins = store[key].row_group_stats.loc[:, "ordered_on_mins"]
    #     key_mins[key] = ordered_on_mins  # index is row_group_idx
    #     all_mins.extend(ordered_on_mins.tolist())

    # global_min = min(all_mins)

    # 2. Merge all unique ordered_on_mins values
    # all_boundaries = []
    # for key in keys:
    #     unique_values = key_mins[key][~key_mins[key].isin(all_boundaries)]
    #     all_boundaries.extend(unique_values.tolist())

    # 3. Sort boundaries and filter by start/end_excl
    # sorted_boundaries = sorted(set(all_boundaries))
    # if start is not None:
    #     sorted_boundaries = [b for b in sorted_boundaries if b >= start]
    # if end_excl is not None:
    #     sorted_boundaries = [b for b in sorted_boundaries if b <= end_excl]
    #     # Ensure end_excl is the final boundary
    #     if not sorted_boundaries or sorted_boundaries[-1] != end_excl:
    #         sorted_boundaries.append(end_excl)

    # def boundary_iterator():
    #     for boundary in sorted_boundaries:
    #         rg_indices = {}
    #         for key in keys:
    #             rg_idx = (key_mins[key] <= boundary).sum() - 1
    #             rg_indices[key] = rg_idx
    #         yield (boundary, rg_indices)

    # return global_min, boundary_iterator()

    pass


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
    # Get global minimum and intersection boundary iterator.
    global_min, intersection_iter = _get_intersection_iterator(store, keys, start, end_excl)
    # Initialize state tracking.
    in_memory_data = {}  # Dict[key, DataFrame] - cached row groups
    prev_rg_indices = {}  # Dict[key, int] - previous row group indices
    current_start = start if start is not None else global_min
    # Iterate over intersection end excluded, and row group indices to load for
    # each key.
    for current_end_excl, rg_indices in intersection_iter:
        # Load new row groups if indices changed.
        for key in keys:
            if rg_indices[key] != prev_rg_indices.get(key, -1):
                in_memory_data[key] = store[key][rg_indices[key]].to_pandas()
                prev_rg_indices[key] = rg_indices[key]
        # Extract synchronized views for current span,
        # [current_start, current_end_excl) and yield.
        yield {
            key: in_memory_data[key].iloc[
                slice(
                    *in_memory_data[key][ordered_on_col_name].searchsorted(
                        [current_start, current_end_excl],
                        side="left",
                    ),
                )
            ]
            for key in keys
        }
        # Update state for next iteration.
        current_start = current_end_excl  # Next start is current end


# TODO: implement 'cache_opds' in Store class?
