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
from numpy import any as np_any
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
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp

from oups.store.split_strategies import NRowsSplitStrategy
from oups.store.split_strategies import RowGroupSplitStrategy
from oups.store.split_strategies import TimePeriodSplitStrategy


MIN = "min"
MAX = "max"
LEFT = "left"
RIGHT = "right"


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


def _compute_enlarged_merge_region(
    rg_split_strategy: RowGroupSplitStrategy,
    max_n_irgs: int,
    is_overlap: NDArray[np_bool],
) -> NDArray[np_int]:
    """
    Aggregate atomic merge regions into enlarged merge regions.

    Atomic merge regions are extended with neighbor incomplete row groups
    depending on two conditions,
      - if the atomic merge region (resulting from the merge of the Dataframe
        chunks and corresponding overlapping row groups) is found to be a
        potentially complete row group.
      - if the total number of existing incomplete row groups in the enlarged
        merge region is greater than `max_n_irgs`.

    A set of contiguous atome merge regions result finally into an enlarged
    merge region.

    Parameters
    ----------
    rg_split_strategy : RowGroupSplitStrategy
        Strategy object that determines how row groups should be split and
        provides methods to analyze completeness and fragmentation risk.
    max_n_irgs : int
        Maximum number of incomplete row groups allowed in a merge region.
    is_overlap : NDArray[np_bool]
        Boolean array indicating overlap regions between DataFrame and row
        groups.

    Returns
    -------
    NDArray[np_int]
        A numpy array of shape (e, 2) containing the list of the start and end
        indices of each enlarged merge region.

    """
    # Step 1: assess start indices (included) and end indices (excluded) of
    # regions.
    indices_overlap = get_region_indices_of_true_values(is_overlap)
    is_incomplete = rg_split_strategy.is_incomplete()
    is_enlarged = is_overlap | is_incomplete
    indices_enlarged = get_region_indices_of_true_values(is_enlarged)
    # Split regions are regions of complete row group, not overlapping with any
    # DataFrame chunks.
    # indices_split = get_region_indices_of_true_values(~is_enlarged)

    # Step 2: keep only enlarged regions that overlap with at least one overlap

    # Step 2: keep only enlarged regions that overlap with at least one overlap
    # region.
    # Should not do that, to keep incomplete row groups that are neighbors to a
    # a large df chunk to be written together, even if they are not overlapping.

    # is_overlapping_with_overlap = np_any(
    #    (indices_enlarged[:, None, 0] <= indices_overlap[None, :, 1])
    #    & (indices_enlarged[:, None, 1] >= indices_overlap[None, :, 0]),
    #    axis=1,  # Note: axis=1 to reduce along columns (overlap regions)
    # )
    # indices_enlarged = indices_enlarged[is_overlapping_with_overlap]

    # Step 3: Filter out enlarged candidates based on multiple criteria.
    # Get number of incomplete row groups at region boundaries.
    # Offset starts by first region's start value.
    cumsum_incomplete = cumsum(is_incomplete)
    n_irgs_region_end = cumsum_incomplete[indices_enlarged[:, 1] - 1]
    n_irgs_region_start = (
        cumsum_incomplete[indices_enlarged[:, 0]] - cumsum_incomplete[indices_enlarged[0, 0]]
    )
    # Get which enlarged regions contain overlap regions that risk fragmentation.
    fragmentation_risk = rg_split_strategy.get_risk_of_exceeding_rgst(
        indices_overlap=indices_overlap,
        indices_enlarged=indices_enlarged,
    )
    # Keep regions with too many incomplete row groups or with fragmentation risk.
    indices_enlarged = indices_enlarged[
        ((n_irgs_region_end - n_irgs_region_start) >= max_n_irgs) | fragmentation_risk
    ]

    # Step 4: Some enlarged sections have been filtered out during previous
    # step. At this step, overlap regions that do not overlap with retained
    # enlarged regions are added back in the regions to merge (without their
    # possibly incomplete neighbor row groups).
    is_overlapping_with_enlarged = np_any(
        (indices_enlarged[:, None, 0] <= indices_overlap[None, :, 0])
        & (indices_enlarged[:, None, 1] >= indices_overlap[None, :, 1]),
        axis=0,
    )
    indices_overlap = indices_overlap[~is_overlapping_with_enlarged]

    return vstack((indices_overlap, indices_enlarged))


def _compute_atomic_merge_regions(
    rg_mins: List[Union[int, float, Timestamp]],
    rg_maxs: List[Union[int, float, Timestamp]],
    df_ordered_on: Series,
    drop_duplicates: bool,
) -> Tuple[NDArray, NDArray[np_int]]:
    """
    Get atomic merge regions.

    An atomic merge region ('amr') is
      - either defined by an existing row group in
    ParquetFile and if existing, its corresponding overlapping DataFrame chunk,
      - or a DataFrame chunk that is not overlapping with any row group in
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
        Flag impacting how overlap boundaries have to be managed.
        More exactly, 'pf' is considered as first data, and 'df' as second
        data, coming after. In case of 'pf' leading 'df', if last value in
        'pf' is a duplicate of the first in 'df', then
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
            ("is_row_group", np_bool),
            ("is_overlap", np_bool),
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
        amrs_prop["is_overlap"][rg_idx_df_orphans + arange(n_df_orphans)] = False

    amrs_prop["rg_idx_start"] = rg_idxs_template[:-1]
    amrs_prop["rg_idx_end_excl"] = rg_idxs_template[1:]
    amrs_prop["df_idx_end_excl"] = df_idx_amr_ends_excl
    amrs_prop["is_row_group"] = amrs_prop["rg_idx_start"] != amrs_prop["rg_idx_end_excl"]
    # No overlap where no DataFrame chunk.
    amrs_prop["is_overlap"][diff(amrs_prop["df_idx_end_excl"], prepend=0) == 0] = False
    return (
        amrs_prop,
        rg_idx_df_orphans,
    )


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
    amrs_prop, rg_idx_df_interlaces_wo_overlap = _compute_atomic_merge_regions(
        rg_mins=pf_statistics[MIN][ordered_on],
        rg_maxs=pf_statistics[MAX][ordered_on],
        df_ordered_on=df.loc[:, ordered_on],
        drop_duplicates=drop_duplicates,
    )
    # Initialize row group split strategy.
    rg_split_strategy = (
        NRowsSplitStrategy(
            pf=pf,
            rg_idx_df_interlaces_wo_overlap=rg_idx_df_interlaces_wo_overlap,
            df_n_rows=len(df),
            amrs_properties=amrs_prop,
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
    amr_idx_emrs_starts_ends_excl = _compute_enlarged_merge_region(
        rg_split_strategy=rg_split_strategy,
        max_n_irgs=max_n_irgs,
        amrs_properties=amrs_prop,
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
