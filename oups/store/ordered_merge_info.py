#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

Ordered merge information for Parquet files and DataFrames.

This module provides functionality to analyze and plan how to merge DataFrames
with existing Parquet files when both are ordered on the same column. It
identifies optimal merge regions and strategies for splitting data into row
groups.

@author: yoh

"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

from fastparquet import ParquetFile
from numpy import arange
from numpy import bool_ as np_bool
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import flatnonzero
from numpy import insert
from numpy import int8
from numpy import int_ as np_int
from numpy import ones
from numpy import r_
from numpy import searchsorted
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp

from oups.store.split_strategies import MergeRegionSplitStrategy
from oups.store.split_strategies import NRowsSplitStrategy
from oups.store.split_strategies import TimePeriodSplitStrategy
from oups.store.utils import get_region_start_end_delta


MIN = "min"
MAX = "max"
LEFT = "left"
RIGHT = "right"
IS_OVERLAP = "is_overlap"
IS_ROW_GROUP = "is_row_group"


def _compute_atomic_merge_regions(
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
    Tuple[NDArray, NDArray[np_int]]
        - First NDArray: Atomic merge regions properties contained in a
          structured array with fields:
          - 'rg_idx_start': indices in ParquetFile containing the starts of each
            row group to merge with corresponding DataFrame chunk.
          - 'rg_idx_end_excl': indices in ParquetFile containing the ends
            (excluded) of each row group to merge with corresponding DataFrame
            chunk.
          - 'df_idx_end_excl': indices in DataFrame containing the ends
            (excluded) of each DataFrame chunk to merge with corresponding
            row group.
          - 'is_row_group': boolean indicating if this region contains a
            row group.
          - 'is_overlap': boolean indicating if this region contains an
            overlap between a row group and DataFrame chunk.

        - Second NDArray: indices in DataFrame containing the ends (excluded)
          of each DataFrame chunk to merge with corresponding row group.

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
    amrs_prop = ones(
        n_rgs + n_df_orphans,
        dtype=[
            ("rg_idx_start", np_int),
            ("rg_idx_end_excl", np_int),
            ("df_idx_end_excl", np_int),
            (IS_ROW_GROUP, np_bool),
            (IS_OVERLAP, np_bool),
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
        # No overlap where no row group.
        amrs_prop[IS_OVERLAP][rg_idx_df_orphans + arange(n_df_orphans)] = False

    amrs_prop["rg_idx_start"] = rg_idxs_template[:-1]
    amrs_prop["rg_idx_end_excl"] = rg_idxs_template[1:]
    amrs_prop["df_idx_end_excl"] = df_idx_amr_ends_excl
    amrs_prop[IS_ROW_GROUP] = amrs_prop["rg_idx_start"] != amrs_prop["rg_idx_end_excl"]
    # No overlap where no DataFrame chunk.
    amrs_prop[IS_OVERLAP][diff(amrs_prop["df_idx_end_excl"], prepend=0) == 0] = False
    return amrs_prop, rg_idx_df_orphans


def get_region_indices_of_true_values(mask: NDArray[np_bool]) -> NDArray[np_int]:
    """
    Compute the start and end indices of each connected components in `mask`.

    Taken from https://stackoverflow.com/questions/68514880/finding-contiguous-regions-in-a-1d-boolean-array.

    Parameters
    ----------
    mask : NDArray[np_bool]
        A 1d numpy array of dtype `bool`.

    Returns
    -------
    NDArray[np_int]
        A numpy array containing the list of the start and end indices of each
        connected components in `mask`.

    """
    return flatnonzero(diff(r_[int8(0), mask.astype(int8), int8(0)])).reshape(-1, 2)


def get_region_indices_of_same_values(ints: NDArray[np_int]) -> NDArray[np_int]:
    """
    Compute the start and end indices of regions made of same values in `ints`.

    Parameters
    ----------
    ints : NDArray[np_int]
        A 1d numpy array of dtype `int`.

    Returns
    -------
    NDArray[np_int]
        A numpy array containing the list of the start and end indices of
        regions made of same values in `ints`.

    """
    boundaries = r_[0, flatnonzero(diff(ints)) + 1, len(ints)]
    return column_stack((boundaries[:-1], boundaries[1:]))


def set_true_in_regions(length: int, regions: NDArray[np_int]) -> NDArray[np_bool]:
    """
    Set regions in a boolean array to True based on start-end index pairs.

    Regions have to be non overlapping.

    Parameters
    ----------
    length : int
        Length of the output array.
    regions : NDArray[np_int]
        2D array of shape (n, 2) where each row contains [start, end) indices.
        Start indices are inclusive, end indices are exclusive.
        Regions are assumed to be non-overlapping.

    Returns
    -------
    NDArray[np_bool]
        Boolean array of length 'length' with True values in specified regions.

    """
    # Array of changes with +1 at starts, and -1 at ends of regions.
    changes = zeros(length + 1, dtype=int8)
    changes[regions[:, 0]] = 1
    changes[regions[:, 1]] = -1
    # Positive cumulative sum provides which positions are inside regions
    return cumsum(changes[:-1]).astype(np_bool)


def _compute_enlarged_merge_regions(
    max_n_irgs: int,
    amr_split_strategy: MergeRegionSplitStrategy,
) -> NDArray[np_int]:
    """
    Aggregate atomic merge regions into enlarged merge regions.

    Sets of contiguous atomic merge regions with DataFrame chunks are extended
    with neighbor incomplete regions depending on two conditions,
      - if the atomic merge region with DataFrame chunks is found to result in a
       potentially complete row group.
      - if the total number of incomplete atomic merge regions in a given
        enlarged merge region is greater than `max_n_irgs`.

    Parameters
    ----------
    max_n_irgs : int
        Maximum number of resulting incomplete contiguous row groups allowed in
        a merge region.
    amr_split_strategy : MergeRegionSplitStrategy
        MergeRegionSplitStrategy providing methods to analyze completeness and
        likelihood of creating complete row groups.

    Returns
    -------
    NDArray[np_int]
        A numpy array of shape (e, 2) containing the list of the start and end
        indices of each enlarged merge region.

    """
    # Step 1: assess start indices (included) and end indices (excluded) of
    # enlarged merge regions.
    likely_incomplete = ~amr_split_strategy.likely_complete
    enlarged_mrs = amr_split_strategy.has_df_chunk | likely_incomplete
    indices_emrs = get_region_indices_of_true_values(enlarged_mrs)

    # Step 2: Filter out enlarged candidates based on multiple criteria.
    # 2.a - Get number of incomplete merge regions per enlarged merged region.
    # Those where 'max_n_irgs' is not reached will be filtered out
    n_irgs_in_emrs = get_region_start_end_delta(
        m_values=cumsum(amr_split_strategy.is_incomplete),
        indices=indices_emrs,
    )
    # 2.b Get which enlarged regions into which the merge will likely create
    # complete row groups.
    creates_likely_complete_in_emrs = get_region_start_end_delta(
        m_values=cumsum(amr_split_strategy.has_df_chunk | amr_split_strategy.likely_complete),
        indices=indices_emrs,
    ).astype(np_bool)
    # Keep enlarged merge regions with too many incomplete atomic merge regions
    # or with likely creation of complete row groups.
    indices_emrs = indices_emrs[(n_irgs_in_emrs >= max_n_irgs) | creates_likely_complete_in_emrs]

    # Step 3: Retrieve indices of merge regions which have DataFrame chunks but
    # are not in retained enlarged merge regions.
    # Create an array of length the number of atomic merge regions, with value
    # 1 if the atomic merge region is within an enlarged merge regions.
    # Create an array of changes: +1 at starts, -1 at ends
    indices_amrs = get_region_indices_of_true_values(amr_split_strategy.has_df_chunk)
    enlarged_mrs = set_true_in_regions(
        length=len(amr_split_strategy),
        regions=indices_emrs,
    )
    overlaps_with_emrs = get_region_start_end_delta(
        m_values=cumsum(enlarged_mrs),
        indices=indices_amrs,
    ).astype(np_bool)
    indices_amrs = indices_amrs[~overlaps_with_emrs]

    return vstack((indices_amrs, indices_emrs))


@dataclass
class OrderedMergePlan:
    """
    Information about how to merge a DataFrame with a ParquetFile.

    DataFrame and ParquetFile are both ordered on a same column.

    Attributes
    ----------
    rg_idx_starts : NDArray
        Indices in ParquetFile containing the starts of each row group to merge.
    rg_idx_ends_excl : NDArray
        Indices in ParquetFile containing the ends (excluded) of each row group to merge.
    df_idx_ends_excl : NDArray
        Indices in DataFrame containing the ends (excluded) of each DataFrame chunk to merge.
    rg_sizer : Callable
        Callable determining the sizing logic for row groups.
    sort_rgs_after_write : bool
        Flag indicating if row groups need to be resorted after merge.

    """

    rg_idx_starts: NDArray[np_int]
    rg_idx_ends_excl: NDArray[np_int]
    df_idx_ends_excl: NDArray[np_int]
    rg_sizer: Callable
    sort_rgs_after_write: bool


def compute_ordered_merge_plan(
    pf: ParquetFile,
    df: DataFrame,
    ordered_on: str,
    row_group_size_target: Union[int, str],
    drop_duplicates: bool,
    max_n_irgs: Optional[int] = None,
) -> OrderedMergePlan:
    """
    Describe how DataFrame and ParquetFile chunks can be merged.

    Parameters
    ----------
    pf : ParquetFile
        Input ParquetFile. Must contain statistics for the 'ordered_on'
        column.
    df : DataFrame
        Input DataFrame. Must contain the 'ordered_on' column.
    ordered_on : str
        Column name by which data is ordered. Must exist in both DataFrame
        and ParquetFile.
    row_group_size_target : Union[int, str]
        Target row group size.
    drop_duplicates : bool
        Flag impacting how overlap boundaries have to be managed.
        More exactly, ParquetFile is considered as leading data, and DataFrame
        as trailing  data. If last values in ParquetFile row groups are
        duplicates of values in DataFrame, then
        - If True, at this index, merge of data between ParquetFile and
          DataFrame is scheduled.
        - If False, merge of data between ParquetFile and DataFrame is not
          scheduled at this index.

    max_n_irgs : Optional[int]
        Max allowed number of 'incomplete' row groups.
        - ``None`` value induces no coalescing of row groups. If there is no
          drop of duplicates, new data is systematically appended.
        - A value of ``0`` means that new data will be merged to neighbors
          existing row groups to attempt yielding only complete row groups after
          the merge.
        - A value of ``1`` means that new data will be merged to the last
          existing row group only if it is not 'complete'.

    Returns
    -------
    OrderedMergePlan

    Notes
    -----
    Important: this function returns indices to slice the input DataFrame. This
    slicing should then be achieved on the same input DataFrame. If targeting
    to drop duplicates in the DataFrame, then duplicate removal should be
    carried out before running this function.

    When a row in ParquetFile shares the same value in 'ordered_on' column as a
    row in DataFrame, the row in ParquetFile is considered as leading the row in
    DataFrame i.e. anterior to it. This has an impact in overlap identification
    in case duplicates are not dropped.

    When 'max_n_irgs' is specified, the method will analyze neighbor incomplete
    row groups and may adjust the overlap regions to include them in the merge
    and rewrite step.

    """
    # Compute atomic merge regions.
    # An atomic merge region is defined
    # - either as a single existing row group (overlapping or not with a
    #   DataFrame chunk),
    # - or as a DataFrame chunk not overlapping with any existing row groups.
    pf_statistics = pf.statistics
    amrs_prop, rg_idx_df_orphans = _compute_atomic_merge_regions(
        rg_mins=pf_statistics[MIN][ordered_on],
        rg_maxs=pf_statistics[MAX][ordered_on],
        df_ordered_on=df.loc[:, ordered_on],
        drop_duplicates=drop_duplicates,
    )
    # Initialize row group split strategy.
    rg_split_strategy = (
        NRowsSplitStrategy(
            pf=pf,
            amrs_properties=amrs_prop,
            rg_idx_df_orphans=rg_idx_df_orphans,
            df_n_rows=len(df),
            row_group_size_target=row_group_size_target,
            max_n_irgs=max_n_irgs,
        )
        if isinstance(row_group_size_target, int)
        else TimePeriodSplitStrategy(
            row_group_size_target,
        )
    )
    # Compute enlarged merge regions start and end indices.
    # Enlarged merge regions are set of contiguous atomic merge regions,
    # possibly extended with neighbor incomplete row groups depending on
    # criteria.
    # It also restricts to the set of atomic merge regions to be yielded.
    amr_idx_emrs_starts_ends_excl = _compute_enlarged_merge_regions(
        max_n_irgs=max_n_irgs,
        rg_idx_df_orphans=rg_idx_df_orphans,
        is_overlap=amrs_prop[IS_OVERLAP],
        rg_split_strategy=rg_split_strategy,
    )
    # Filter and reshape arrays describing atomic merge regions into enlarged
    # merge regions.
    # For each enlarged merge regions, aggregate atomic merge regions depending
    # on the split strategy.
    emrs_idxs = [
        rg_split_strategy.consolidate_enlarged_merge_regions(
            amrs_properties=amrs_prop[start:end_excl],
        )
        for start, end_excl in amr_idx_emrs_starts_ends_excl
    ]

    # Assess if row groups have to be sorted after write step
    #  - either if there is a merge region (new row groups are written first,
    #    then old row groups are removed).
    #  - or there is no merge region but df is not appended at the tail of
    #    existing data.
    sort_rgs_after_write = True
    return OrderedMergePlan(
        emrs_idxs=emrs_idxs,
        rg_sizer=rg_split_strategy.rg_sizer,
        sort_rgs_after_write=sort_rgs_after_write,
    )
