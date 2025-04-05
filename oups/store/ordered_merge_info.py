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
from typing import Callable, Optional, Union

from fastparquet import ParquetFile
from numpy import array
from numpy import bool_ as np_bool
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import flatnonzero
from numpy import int8
from numpy import int_ as np_int
from numpy import r_
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import DataFrame

from oups.store.ordered_atomic_regions import HAS_DF_CHUNK
from oups.store.ordered_atomic_regions import NRowsSplitStrategy
from oups.store.ordered_atomic_regions import OARSplitStrategy
from oups.store.ordered_atomic_regions import TimePeriodSplitStrategy
from oups.store.ordered_atomic_regions import compute_ordered_atomic_regions
from oups.store.utils import get_region_start_end_delta


MIN = "min"
MAX = "max"


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
    oars_info: NDArray,
    oar_split_strategy: OARSplitStrategy,
    max_n_irgs: int,
) -> NDArray[np_int]:
    """
    Aggregate atomic merge regions into enlarged merge regions.

    Sets of contiguous atomic merge regions with DataFrame chunks are extended
    with neighbor regions outside target size depending on two conditions,
      - if the atomic merge region with DataFrame chunks is found to result in a
        row group potentially meeting target size.
      - if the total number of atomic merge regions outside target size in a
        given enlarged merge region is greater than `max_n_irgs`.

    Parameters
    ----------
    oars_info : NDArray
        Array of shape (n, 5) containing the information about each atomic
        merge region.
    oar_split_strategy : OARSplitStrategy
        OARSplitStrategy providing methods to analyze completeness and
        likelihood of creating complete row groups.
    max_n_irgs : int
        Maximum number of resulting contiguous row groups outside target size
        (incomplete) allowed in a merge region.

    Returns
    -------
    NDArray[np_int]
        A numpy array of shape (e, 2) containing the list of the start and end
        indices for each enlarged merge regions.

    """
    # Step 1: assess start indices (included) and end indices (excluded) of
    # enlarged merge regions.
    likely_outside_target_size = ~oar_split_strategy.likely_meets_target_size
    enlarged_mrs = oars_info[HAS_DF_CHUNK] | likely_outside_target_size
    indices_emrs = get_region_indices_of_true_values(enlarged_mrs)

    # Step 2: Filter out enlarged candidates based on multiple criteria.
    # 2.a - Get number of incomplete merge regions per enlarged merged region.
    # Those where 'max_n_irgs' is not reached will be filtered out
    n_oars_in_emrs = get_region_start_end_delta(
        m_values=cumsum(likely_outside_target_size),
        indices=indices_emrs,
    )
    # 2.b Get which enlarged regions into which the merge will likely create
    # right sized row groups.
    creates_likely_right_sized_in_emrs = get_region_start_end_delta(
        m_values=cumsum(
            oar_split_strategy.has_df_chunk | oar_split_strategy.likely_meets_target_size,
        ),
        indices=indices_emrs,
    ).astype(np_bool)
    # Keep enlarged merge regions with too many incomplete atomic merge regions
    # or with likely creation of complete row groups.
    indices_emrs = indices_emrs[(n_oars_in_emrs >= max_n_irgs) | creates_likely_right_sized_in_emrs]

    # Step 3: Retrieve indices of merge regions which have DataFrame chunks but
    # are not in retained enlarged merge regions.
    # Create an array of length the number of atomic merge regions, with value
    # 1 if the atomic merge region is within an enlarged merge regions.
    # Create an array of changes: +1 at starts, -1 at ends
    indices_oars = get_region_indices_of_true_values(oars_info[HAS_DF_CHUNK])
    enlarged_mrs = set_true_in_regions(
        length=len(oar_split_strategy),
        regions=indices_emrs,
    )
    overlaps_with_emrs = get_region_start_end_delta(
        m_values=cumsum(enlarged_mrs),
        indices=indices_oars,
    ).astype(np_bool)
    indices_oars = indices_oars[~overlaps_with_emrs]

    return vstack((indices_oars, indices_emrs))


@dataclass
class OrderedMergePlan:
    """
    Information about how to merge a DataFrame with a ParquetFile.

    DataFrame and ParquetFile are both ordered on a same column.

    Attributes
    ----------
    emrs_info : NDArray
        Array of shape (e, 3) containing the information about each enlarged
        merge regions.
        Each row contains the following information:
        - indices in ParquetFile containing the starts of each row group to
          merge.
        - indices in ParquetFile containing the ends (excluded) of each row
          group to merge.
        - indices in DataFrame containing the ends (excluded) of each DataFrame
          chunk to merge.
    rg_sizer : Callable
        Callable determining the sizing logic for row groups.
    sort_rgs_after_write : bool
        Flag indicating if row groups need to be resorted after merge.

    """

    emrs_info: NDArray
    rg_sizer: Callable
    sort_rgs_after_write: bool


def compute_ordered_merge_plan(
    pf: ParquetFile,
    df: DataFrame,
    ordered_on: str,
    row_group_target_size: Union[int, str],
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
    row_group_target_size : Union[int, str]
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
    df_ordered_on = df.loc[:, ordered_on]
    oars_info = compute_ordered_atomic_regions(
        rg_mins=pf_statistics[MIN][ordered_on],
        rg_maxs=pf_statistics[MAX][ordered_on],
        df_ordered_on=df_ordered_on,
        drop_duplicates=drop_duplicates,
    )
    # Initialize row group split strategy.
    oar_split_strategy = (
        NRowsSplitStrategy(
            rgs_n_rows=array([rg.num_rows for rg in pf], dtype=int),
            df_n_rows=len(df),
            oars_info=oars_info,
            row_group_target_size=row_group_target_size,
            max_n_irgs=max_n_irgs,
        )
        if isinstance(row_group_target_size, int)
        else TimePeriodSplitStrategy(
            rg_ordered_on_mins=pf_statistics[MIN][ordered_on],
            rg_ordered_on_maxs=pf_statistics[MAX][ordered_on],
            df_ordered_on=df_ordered_on,
            oars_info=oars_info,
            row_group_period=row_group_target_size,
        )
    )
    # Compute enlarged merge regions start and end indices.
    # Enlarged merge regions are set of contiguous atomic merge regions,
    # possibly extended with neighbor incomplete row groups depending on
    # criteria.
    # It also restricts to the set of atomic merge regions to be yielded.
    oar_idx_emrs_starts_ends_excl = _compute_enlarged_merge_regions(
        oars_info=oars_info,
        oar_split_strategy=oar_split_strategy,
        max_n_irgs=max_n_irgs,
    )
    # Filter and reshape arrays describing atomic merge regions into enlarged
    # merge regions.
    # For each enlarged merge regions, aggregate atomic merge regions depending
    # on the split strategy.
    emrs_info = [
        oar_split_strategy.consolidate_enlarged_merge_regions(
            oars_info=oars_info[start:end_excl],
        )
        for start, end_excl in oar_idx_emrs_starts_ends_excl
    ]

    # Assess if row groups have to be sorted after write step
    #  - either if there is a merge region (new row groups are written first,
    #    then old row groups are removed).
    #  - or there is no merge region but df is not appended at the tail of
    #    existing data.
    sort_rgs_after_write = True
    return OrderedMergePlan(
        emrs_info=emrs_info,
        rg_sizer=oar_split_strategy.rg_sizer,
        sort_rgs_after_write=sort_rgs_after_write,
    )
