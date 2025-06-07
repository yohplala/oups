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
from numpy import ones
from numpy import r_
from numpy import searchsorted
from pandas import DataFrame
from pandas import Int64Dtype
from pandas import Timestamp

from oups.defines import KEY_ORDERED_ON_MAXS
from oups.defines import KEY_ORDERED_ON_MINS
from oups.numpy_utils import isnotin_ordered


KEY_LEFT = "left"


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


def _get_intersections(
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
        - Iterator yielding (current_end_excl, rg_idx_ends_excl) tuples where:
          * current_end_excl: End boundary (exclusive) for current intersection
          * rg_idx_ends_excl: Dict mapping each key to its row group index for
            this intersection

    Notes
    -----
    - A key without value in the span of interest in 'ordered_on' will not
      appear in returned dict and yielded items.
    - The intersection is not exactly optimized in the sense that it will not
      try to load row groups only when needed, but when potentially needed.
      Current implementation, simplified, may load row group in advance and do
      not try to release them when not needed any longer, but only when the next
      row group needs to be loaded.
      This enable to only have to rely on 'ordered_on_mins' values.
      Another simplification is to load the first row group to appear for each
      key right at the first iteration, even though it would not be needed
      immediately.

    """
    #    print()
    #    print("-- 1st loop --")
    if isinstance(start, Timestamp):
        start = start.to_numpy()
    if isinstance(end_excl, Timestamp):
        end_excl = end_excl.to_numpy()
    # Store "ordered_on_mins" in a dict, only keeping unique value and
    # corresponding row group indices.
    unique_ordered_on_mins = None
    keys_ordered_on_ends_excl = {}
    keys_rg_idx_starts = {}
    keys_rg_idx_first_ends_excl = {}
    keys_rg_idx_ends_excl = {}
    for key in keys:
        #        print("key ", key)
        #        print("unique_ordered_on_mins")
        #        print(unique_ordered_on_mins)
        row_group_stats = store[key].row_group_stats
        ordered_on_mins = row_group_stats.loc[:, KEY_ORDERED_ON_MINS].to_numpy()
        ordered_on_maxs = row_group_stats.loc[:, KEY_ORDERED_ON_MAXS].to_numpy()
        n_rgs = len(row_group_stats)
        rg_idx = arange(n_rgs)
        # Main row groups are those not overlapping with next ones.
        mask_main_rgs = ones(n_rgs).astype(bool)
        mask_main_rgs[1:] = ordered_on_mins[1:] != ordered_on_maxs[:-1]
        _unique_ordered_on_mins = ordered_on_mins[mask_main_rgs]
        _unique_rg_idx_ends_excl = rg_idx[mask_main_rgs]
        # _unique_ordered_on_mins, _unique_rg_idx_ends_excl = unique(
        #    store[key].row_group_stats.loc[:, KEY_ORDERED_ON_MINS].to_numpy(),
        #    return_index=True,
        # )
        #        print("_unique_ordered_on_mins")
        #        print(_unique_ordered_on_mins)
        #        print("_unique_rg_idx_ends_excl")
        #        print(_unique_rg_idx_ends_excl)
        # Skip first row group in trimming.
        trim_idx_first_end_excl = (
            searchsorted(_unique_ordered_on_mins, start, side=KEY_LEFT) + 1 if start else 1
        )

        trim_idx_last_end_excl = (
            searchsorted(_unique_ordered_on_mins, end_excl, side=KEY_LEFT)
            if end_excl
            else len(_unique_ordered_on_mins)
        )
        #        print("trim_idx_first_end_excl")
        #        print(trim_idx_first_end_excl)
        #        print("trim_idx_last_end_excl")
        #        print(trim_idx_last_end_excl)
        keys_ordered_on_ends_excl[key] = _unique_ordered_on_mins[
            trim_idx_first_end_excl:trim_idx_last_end_excl
        ]
        #        print("keys_ordered_on_ends_excl[key]")
        #        print(keys_ordered_on_ends_excl[key])
        # 'unique_rg_idx_ends_excl' is completed with its length as last value.
        if trim_idx_first_end_excl < trim_idx_last_end_excl + 1:
            _unique_rg_idx_ends_excl = r_[_unique_rg_idx_ends_excl, n_rgs]
            keys_rg_idx_ends_excl[key] = _unique_rg_idx_ends_excl[
                trim_idx_first_end_excl : trim_idx_last_end_excl + 1
            ]
            keys_rg_idx_starts[key] = _unique_rg_idx_ends_excl[trim_idx_first_end_excl - 1]
            keys_rg_idx_first_ends_excl[key] = keys_rg_idx_ends_excl[key][0]
            # Collect 'ordered_on_mins' for each key, keeping unique values only.
            if unique_ordered_on_mins is None:
                unique_ordered_on_mins = keys_ordered_on_ends_excl[key]
            else:
                is_not_found, unfound_insert_idx = isnotin_ordered(
                    sorted_array=unique_ordered_on_mins,
                    query_elements=keys_ordered_on_ends_excl[key],
                    return_insert_positions=True,
                )
                unique_ordered_on_mins = insert(
                    unique_ordered_on_mins,
                    unfound_insert_idx,
                    keys_ordered_on_ends_excl[key][is_not_found],
                )
    #    print("unique_ordered_on_mins")
    #    print(unique_ordered_on_mins)
    #    print("keys_ordered_on_ends_excl")
    #    print(keys_ordered_on_ends_excl)
    #    print("keys_rg_idx_starts")
    #    print(keys_rg_idx_starts)
    #    print("keys_rg_idx_first_ends_excl")
    #    print(keys_rg_idx_first_ends_excl)
    #    print("keys_rg_idx_ends_excl")
    #    print(keys_rg_idx_ends_excl)
    #    print()
    #    print("-- 2nd loop --")
    # Adding one for last value, will be either 'end_excl' or None.
    len_unique_ordered_on_mins = len(unique_ordered_on_mins) + 1
    #    print("len_unique_ordered_on_mins")
    #    print(len_unique_ordered_on_mins)
    for key in keys:
        #        print("key ", key)
        if key in keys_rg_idx_ends_excl:
            #        if len(keys_rg_idx_ends_excl[key]) > 0:
            rg_idx_ends_excl = full(len_unique_ordered_on_mins, nan)
            # To accommodate with last row group index, which cannot be
            # be always positioned, depending content of
            # 'keys_ordered_on_ends_excl[key]'.
            rg_idx_ends_excl[-1] = keys_rg_idx_ends_excl[key][-1]
            #            if len_unique_ordered_on_mins > 1:
            #            print("keys_ordered_on_ends_excl[key]")
            #            print(keys_ordered_on_ends_excl[key])
            #            print("searchsorted")
            #            print(
            #                searchsorted(
            #                    unique_ordered_on_mins,
            #                    keys_ordered_on_ends_excl[key],
            #                    side=KEY_LEFT,
            #                ),
            #            )
            confirmed_ordered_on_ends_excl_idx = searchsorted(
                unique_ordered_on_mins,
                keys_ordered_on_ends_excl[key],
                side=KEY_LEFT,
            )
            rg_idx_ends_excl[confirmed_ordered_on_ends_excl_idx] = keys_rg_idx_ends_excl[key][
                : len(confirmed_ordered_on_ends_excl_idx)
            ]
            #            print("rg_idx_ends_excl with ends_excl values")
            #            print(rg_idx_ends_excl)
            #            keys_rg_idx_ends_excl[key] = Series(rg_idx_ends_excl, dtype=Int64Dtype()).bfill()#
            keys_rg_idx_ends_excl[key] = rg_idx_ends_excl
    #        else:
    # If a key has no row group indices, the key is not added in the
    # results. This prevents useless row group in-memory loading.
    #            del keys_rg_idx_starts[key]
    #            del keys_rg_idx_first_ends_excl[key]
    #            del keys_rg_idx_ends_excl[key]

    #    print("keys_rg_idx_ends_excl")
    #    print(keys_rg_idx_ends_excl)
    #    intersections = (
    #        dict(zip(keys_rg_idx_ends_excl, t)) for t in zip(*keys_rg_idx_ends_excl.values())
    #    )
    intersections = DataFrame(keys_rg_idx_ends_excl, dtype=Int64Dtype())
    intersections.bfill(axis=0, inplace=True)
    #    print("unique_ordered_on_mins + end_excl")
    #    print(list(unique_ordered_on_mins) + [end_excl])
    #    print("intersections")
    #    intersections = list(intersections)
    #    print(intersections)
    #    print()
    return (
        keys_rg_idx_starts,
        keys_rg_idx_first_ends_excl,
        #        zip(list(unique_ordered_on_mins) + [end_excl], intersections),
        zip(list(unique_ordered_on_mins) + [end_excl], intersections.to_dict(orient="records")),
    )


def iter_intersections(
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
    print()
    # Get and validate ordered_on column name.
    ordered_on_col_name = _get_and_validate_ordered_on_column(store, keys)
    # Get global minimum and intersection boundary iterator and initialize
    # the "previous" (the first) row group indices.
    rg_idx_starts, prev_rg_idx_ends_excl, intersections = _get_intersections(
        store,
        keys,
        start,
        end_excl,
    )
    # Load initial row groups and initialize start indices.
    # Iterate over 'rg_idx_starts' because only keys with data are kept.
    #   print()
    #   print("rg_idx_starts")
    #   print(rg_idx_starts)
    #   print("prev_rg_idx_ends_excl")
    #   print(prev_rg_idx_ends_excl)
    in_memory_data = {
        key: store[key][rg_idx_start : prev_rg_idx_ends_excl[key]].to_pandas()
        for key, rg_idx_start in rg_idx_starts.items()
    }
    print("in_memory_data")
    print(in_memory_data)
    current_start_indices = (
        {
            key: value.loc[:, ordered_on_col_name].searchsorted(start, side=KEY_LEFT)
            for key, value in in_memory_data.items()  # value is a DataFrame
        }
        if start is not None
        else defaultdict(lambda: None)
    )
    current_end_indices = {}
    for current_end_excl, rg_idx_ends_excl in intersections:
        # TODO: rg_idx_ends_excl has to be a end_excl index
        # use slice notation to get severral row groups when having duplicates
        # in orered_on_mins.
        print("current_end_excl")
        print(current_end_excl)
        print("prev_rg_idx_ends_excl")
        print(prev_rg_idx_ends_excl)
        print("rg_idx_ends_excl")
        print(rg_idx_ends_excl)
        for key, rg_idx_end_excl in rg_idx_ends_excl.items():
            if rg_idx_end_excl != prev_rg_idx_ends_excl[key]:
                print("updating in memory data for key ", key)
                print("prev in memory data")
                print(in_memory_data[key])
                in_memory_data[key] = store[key][
                    prev_rg_idx_ends_excl[key] : rg_idx_end_excl
                ].to_pandas()
                prev_rg_idx_ends_excl[key] = rg_idx_end_excl
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
        print("current in memory data")
        print(in_memory_data)
        print("current_start_indices")
        print(current_start_indices)
        print("current_end_indices")
        print(current_end_indices)
        # Yield synchronized views for current span.
        # [current_start, current_end_excl)
        yield {
            key: df.iloc[current_start_indices[key] : current_end_indices[key], :].reset_index(
                drop=True,
            )
            for key, df in in_memory_data.items()
        }
        # Buffer end indices for next iteration as start indices.
        current_start_indices = current_end_indices.copy()
