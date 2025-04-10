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
from oups.store.ordered_atomic_regions import TimePeriodSplitStrategy
from oups.store.ordered_atomic_regions import compute_ordered_atomic_regions


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


def get_region_start_end_delta(m_values: NDArray, indices: NDArray) -> NDArray:
    """
    Get difference between values at end and start of each region.

    Parameters
    ----------
    m_values : NDArray
        Array of monotonic values, such as coming from a cumulative sum.
    indices : NDArray
        Array of shape (n, 2) where 'n' is the number of regions, and each row
        contains start included and end excluded indices of a region.

    Returns
    -------
    NDArray
        Array of length 'n' containing the difference between values at end and
        start of each region.

    """
    if indices[0, 0] == 0:
        start_values = m_values[indices[:, 0] - 1]
        start_values[0] = 0
        return m_values[indices[:, 1] - 1] - start_values
    else:
        return m_values[indices[:, 1] - 1] - m_values[indices[:, 0] - 1]


def compute_emrs_start_ends_excl(
    oars_has_df_chunk: NDArray[np_bool],
    oars_likely_meets_target_size: NDArray[np_bool],
    max_n_off_target_rgs: int,
) -> NDArray[np_int]:
    """
    Aggregate atomic merge regions into enlarged merge regions.

    Sets of contiguous atomic merge regions with DataFrame chunks are extended
    with neighbor regions that are off target size depending on two
    conditions,
      - if the atomic merge region with DataFrame chunks is found to result in a
        row group potentially meeting target size.
      - if the total number of atomic merge regions off target size in a
        given enlarged merge region is greater than `max_n_off_target_rgs`.

    Parameters
    ----------
    oars_has_df_chunk : NDArray[np_bool]
        Boolean array of shape (n) indicating if each atomic merge region
        contains a DataFrame chunk.
    oars_likely_meets_target_size : NDArray[np_bool]
        Boolean array of shape (n) indicating if each atomic merge region
        is likely to result in a row group meeting target size.
    max_n_off_target_rgs : int
        Maximum number of off-target size row groups allowed in a contiguous set
        of row groups. This parameter helps limiting fragmentation by limiting
        number of contiguous row groups off target size.

    Returns
    -------
    NDArray[np_int]
        A numpy array of shape (e, 2) containing the list of the start and end
        indices for each enlarged merge regions. This list is unsorted. It
        starts with start and end indices excluded of the simple (not enlarged)
        merge regions, and then continues with start and end indices excluded
        of the enlarged merge regions.

    Notes
    -----
    Reason for including off target size OARs contiguous to a newly added OAR
    likely to meet target size is to prevent that the addition of new data
    creates isolated sets of off-target size row groups followed by complete
    row groups. This most notably applies when new data is appended at the tail
    of the DataFrame.

    """
    # Step 1: assess start indices (included) and end indices (excluded) of
    # enlarged merge regions.
    oars_off_target = ~oars_likely_meets_target_size
    potential_enlarged_mrs = oars_has_df_chunk | oars_off_target
    potential_emrs_starts_ends_excl = get_region_indices_of_true_values(potential_enlarged_mrs)
    print()
    print("potential_emrs_starts_ends_excl")
    print(potential_emrs_starts_ends_excl)

    # Step 2: Filter out enlarged candidates based on multiple criteria.
    # 2.a - Get number of off target size OARs per enlarged merged region.
    # Those where 'max_n_off_target_rgs' is not reached will be filtered out
    n_off_target_rgs_in_potential_emrs = get_region_start_end_delta(
        m_values=cumsum(oars_off_target),
        indices=potential_emrs_starts_ends_excl,
    )
    print()
    print("n_off_target_rgs_in_potential_emrs")
    print(n_off_target_rgs_in_potential_emrs)
    # 2.b Get which enlarged regions into which the merge will likely create
    # on target row groups.
    creates_on_target_rg_in_pemrs = get_region_start_end_delta(
        m_values=cumsum(oars_likely_meets_target_size),
        indices=potential_emrs_starts_ends_excl,
    ).astype(np_bool)
    print()
    print("creates_on_target_rg_in_pemrs")
    print(creates_on_target_rg_in_pemrs)
    # Keep enlarged merge regions with too many off target atomic regions or
    # with likely creation of on target row groups.
    confirmed_emrs_starts_ends_excl = potential_emrs_starts_ends_excl[
        (n_off_target_rgs_in_potential_emrs > max_n_off_target_rgs) | creates_on_target_rg_in_pemrs
    ]
    print()
    print("is confirmed emrs")
    print(confirmed_emrs_starts_ends_excl)

    # Step 3: Retrieve indices of merge regions which have DataFrame chunks but
    # are not in retained enlarged merge regions.
    confirmed_emrs = set_true_in_regions(
        length=len(oars_has_df_chunk),
        regions=confirmed_emrs_starts_ends_excl,
    )
    # Create an array of length the number of atomic merge regions, with value
    # 1 if the atomic merge region is within a merge regions.
    simple_mrs_starts_ends_excl = get_region_indices_of_true_values(oars_has_df_chunk)
    overlaps_with_confirmed_emrs = get_region_start_end_delta(
        m_values=cumsum(confirmed_emrs),
        indices=simple_mrs_starts_ends_excl,
    ).astype(np_bool)
    simple_mrs_starts_ends_excl = simple_mrs_starts_ends_excl[~overlaps_with_confirmed_emrs]
    print()
    print("simple_mrs_starts_ends_excl")
    print(simple_mrs_starts_ends_excl)
    print("confirmed_emrs_starts_ends_excl")
    print(confirmed_emrs_starts_ends_excl)
    print("after vstack")
    print(vstack((simple_mrs_starts_ends_excl, confirmed_emrs_starts_ends_excl)))

    return vstack((simple_mrs_starts_ends_excl, confirmed_emrs_starts_ends_excl))


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
    max_n_off_target_rgs: Optional[int] = None,
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

    max_n_off_target_rgs : Optional[int]
        Maximum number of off-target size row groups allowed in a contiguous set
        of row groups. This parameter helps limiting fragmentation by limiting
        number of contiguous off target size row groups.
        - ``None`` value induces no coalescing of row groups. If there is no
          drop of duplicates, new data is systematically appended.
        - A value of ``0`` means that new data will be merged to neighbors
          existing row groups to attempt yielding only complete row groups after
          the merge.
        - A value of ``n >= 1`` means that up to n contiguous off-target size
          row groups are allowed before triggering a merge, hereby limiting
          fragmentation.

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

    When 'max_n_off_target_rgs' is specified, the method will analyze contiguous
    neighbor off target size row groups and may adjust the overlap regions to
    include them in the merge and rewrite step.

    """
    # Compute atomic merge regions.
    # An atomic merge region is defined
    # - either as a single existing row group (overlapping or not with a
    #   DataFrame chunk),
    # - or as a DataFrame chunk not overlapping with any existing row groups.
    pf_statistics = pf.statistics
    rg_ordered_on_mins = array(pf_statistics[MIN][ordered_on])
    rg_ordered_on_maxs = array(pf_statistics[MAX][ordered_on])
    df_ordered_on = df.loc[:, ordered_on]
    oars_info = compute_ordered_atomic_regions(
        rg_ordered_on_mins=rg_ordered_on_mins,
        rg_ordered_on_maxs=rg_ordered_on_maxs,
        df_ordered_on=df_ordered_on,
        drop_duplicates=drop_duplicates,
    )
    oar_split_strategy = (
        NRowsSplitStrategy(
            rgs_n_rows=array([rg.num_rows for rg in pf], dtype=int),
            df_n_rows=len(df),
            oars_info=oars_info,
            row_group_target_size=row_group_target_size,
            max_n_off_target_rgs=max_n_off_target_rgs,
        )
        if isinstance(row_group_target_size, int)
        else TimePeriodSplitStrategy(
            rg_ordered_on_mins=rg_ordered_on_mins,
            rg_ordered_on_maxs=rg_ordered_on_maxs,
            df_ordered_on=df_ordered_on,
            oars_info=oars_info,
            row_group_period=row_group_target_size,
        )
    )
    # Compute enlarged merge regions start and end indices.
    # Enlarged merge regions are set of contiguous atomic merge regions,
    # possibly extended with neighbor off target size row groups depending on
    # criteria.
    # It also restricts to the set of atomic merge regions to be yielded.
    oar_idx_emrs_starts_ends_excl = compute_emrs_start_ends_excl(
        oars_has_df_chunk=oars_info[HAS_DF_CHUNK],
        oars_likely_meets_target_size=oar_split_strategy.likely_meets_target_size,
        max_n_off_target_rgs=max_n_off_target_rgs,
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
