#!/usr/bin/env python3
"""
Created on Mon Mar 17 18:00:00 2025.

Atomic merge regions for Parquet files and DataFrames.

This module defines the base component for analyzing how DataFrame can be merged
with existing Parquet files when both are ordered on the same column.
An atomic merge region ('amr') represents the smallest unit for merging, which
is either:
  - A single row group in a ParquetFile and its corresponding overlapping
    DataFrame chunk (if any)
  - A DataFrame chunk that doesn't overlap with any row group in the ParquetFile

@author: yoh

"""
from typing import List, Tuple, Union

from numpy import arange
from numpy import bool_ as np_bool
from numpy import diff
from numpy import flatnonzero
from numpy import insert
from numpy import int_ as np_int
from numpy import ndarray as NDArray
from numpy import ones
from numpy import r_
from numpy import searchsorted
from numpy import vstack
from pandas import Series
from pandas import Timestamp


LEFT = "left"
RIGHT = "right"
HAS_ROW_GROUP = "has_row_group"
HAS_DF_CHUNK = "has_df_chunk"


def compute_atomic_merge_regions(
    rg_mins: List[Union[int, float, Timestamp]],
    rg_maxs: List[Union[int, float, Timestamp]],
    df_ordered_on: Series,
    drop_duplicates: bool,
) -> Tuple[NDArray, NDArray[np_int]]:
    """
    Compute atomic merge regions.

    An atomic merge region ('amr') is
      - either defined by an existing row group in
        ParquetFile and if existing, its corresponding overlapping DataFrame
        chunk,
      - or an orphan DataFrame chunk, i.e. not overlapping with any row group in
        ParquetFile.

    Returned arrays provide the start and end (excluded) indices in row groups
    and end (excluded) indices in DataFrame for each of these atomic merge
    regions. All these arrays are of same size and describe how are composed the
    atomic merge regions.

    Parameters
    ----------
    rg_mins : List
        Minimum values of 'ordered_on' in each row group.
    rg_maxs : List
        Maximum values of 'ordered_on' in each row group.
    df_ordered_on : Series[Timestamp]
        Values of 'ordered_on' column in DataFrame.
    drop_duplicates : bool
        Flag impacting how overlapping boundaries have to be managed.
        More exactly, 'pf' is considered as first data, and 'df' as second
        data, coming after. In case of 'pf' leading 'df', if last value in
        'pf' is a duplicate of a value in 'df', then
        - If True, at this index, overlap starts
        - If False, no overlap at this index

    Returns
    -------
    NDArray
        Atomic merge regions properties contained in a structured array with
        fields:
        - 'rg_idx_start': indices in ParquetFile containing the starts of each
          row group to merge with corresponding DataFrame chunk.
        - 'rg_idx_end_excl': indices in ParquetFile containing the ends
          (excluded) of each row group to merge with corresponding DataFrame
          chunk.
        - 'df_idx_end_excl': indices in DataFrame containing the ends
          (excluded) of each DataFrame chunk to merge with corresponding
          row group.
        - 'has_row_group': boolean indicating if this region contains a
          row group.
        - 'has_df_chunkp': boolean indicating if this region contains a
          DataFrame chunk.

    Notes
    -----
    Start indices in DataFrame are not provided, as they can be inferred from
    the end (excluded) indices in DataFrame of the previous atomic merge region
    (no part of the DataFrame is omitted for the write).

    In case 'drop_duplicates' is False, and there are duplicate values between
    row group max values and DataFrame 'ordered_on' values, then DataFrame
    'ordered_on' values are considered to be the last occurrences of the
    duplicates in 'ordered_on'. Leading row groups (with duplicate max values)
    will not be in the same atomic merge region as the DataFrame chunk starting
    at the duplicate 'ordered_on' value. This is an optimization to prevent
    rewriting these leading row groups.

    """
    # Find regions in DataFrame overlapping with row groups.
    if drop_duplicates:
        # Determine overlap start/end indices in row groups
        df_idx_amr_starts = searchsorted(df_ordered_on, rg_mins, side=LEFT)
        df_idx_amr_ends_excl = searchsorted(df_ordered_on, rg_maxs, side=RIGHT)
    else:
        df_idx_amr_starts, df_idx_amr_ends_excl = searchsorted(
            df_ordered_on,
            vstack((rg_mins, rg_maxs)),
            side=LEFT,
        )
    # Find regions in DataFrame not overlapping with any row group.
    df_interlaces_wo_overlap = r_[
        df_idx_amr_starts[0],  # gap at start (0 to first start)
        df_idx_amr_ends_excl[:-1] - df_idx_amr_starts[1:],
        len(df_ordered_on) - df_idx_amr_ends_excl[-1],  # gap at end
    ]
    # Indices in row groups where a DataFrame chunk is not overlapping with any
    # row group.
    rg_idx_df_orphans = flatnonzero(df_interlaces_wo_overlap)
    n_rgs = len(rg_mins)
    rg_idxs_template = arange(n_rgs + 1)
    # Create a structured array to hold all related indices
    # DataFrame orphans are regions in DataFrame that do not overlap with any
    # row group.
    n_df_orphans = len(rg_idx_df_orphans)
    amrs_info = ones(
        n_rgs + n_df_orphans,
        dtype=[
            ("rg_idx_start", np_int),
            ("rg_idx_end_excl", np_int),
            ("df_idx_end_excl", np_int),
            (HAS_ROW_GROUP, np_bool),
            (HAS_DF_CHUNK, np_bool),
        ],
    )
    if n_df_orphans != 0:
        # Case of non-overlapping regions in DataFrame.
        # Resize 'rg_idxs', and duplicate values where there are non-overlapping
        # regions in DataFrame.
        rg_idx_to_insert = rg_idxs_template[rg_idx_df_orphans]
        rg_idxs_template = insert(
            rg_idxs_template,
            rg_idx_df_orphans,
            rg_idx_to_insert,
        )
        # 'Resize 'df_idx_amr_ends_excl', and duplicate values where there are
        # non-overlapping regions in DataFrame.
        if rg_idx_df_orphans[-1] == len(df_ordered_on):
            df_idx_to_insert = df_idx_amr_starts[rg_idx_df_orphans]
        else:
            df_idx_to_insert = r_[df_idx_amr_starts, len(df_ordered_on)][rg_idx_df_orphans]
        df_idx_amr_ends_excl = insert(
            df_idx_amr_ends_excl,
            rg_idx_df_orphans,
            df_idx_to_insert,
        )

    amrs_info["rg_idx_start"] = rg_idxs_template[:-1]
    amrs_info["rg_idx_end_excl"] = rg_idxs_template[1:]
    amrs_info["df_idx_end_excl"] = df_idx_amr_ends_excl
    amrs_info[HAS_ROW_GROUP] = amrs_info["rg_idx_start"] != amrs_info["rg_idx_end_excl"]
    amrs_info[HAS_DF_CHUNK] = diff(amrs_info["df_idx_end_excl"], prepend=0) != 0
    return amrs_info
