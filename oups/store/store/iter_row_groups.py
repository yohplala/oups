#!/usr/bin/env python3
"""
Created on Mon May 26 18:00:00 2025.

@author: yoh

"""
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union

from pandas import DataFrame
from pandas import Timestamp


def iter_row_groups(
    store,  # Store instance
    keys: List[dataclass],
    start: Optional[Union[int, float, Timestamp]] = None,
    end: Optional[Union[int, float, Timestamp]] = None,
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
        Start value for the 'ordered_on' column range. If None, starts from
        the earliest value across all specified keys.
    end : Optional[Union[int, float, Timestamp]], default None
        End value for the 'ordered_on' column range. If None, continues until
        the latest value across all specified keys.

    Yields
    ------
    Dict[dataclass, DataFrame]
        Dictionary mapping each key to its corresponding DataFrame chunk.
        All DataFrames in each yielded dictionary share a common span in their
        'ordered_on' column. The span size is determined by the intersection of
        available data ranges from all keys currently loaded in memory.

    Notes
    -----
    - All datasets must have an 'ordered_on' column with the same name.
    - The iteration is synchronized: each yield contains data from the same
      span across all datasets.
    - Chunks that extend beyond the current common span are cached for the next
      iteration.

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
