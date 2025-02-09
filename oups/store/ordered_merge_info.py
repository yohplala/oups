#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from abc import ABC
from abc import abstractmethod
from typing import List, Optional, Tuple, Union

from fastparquet import ParquetFile
from numpy import any as np_any
from numpy import arange
from numpy import array
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import flatnonzero
from numpy import insert
from numpy import int8
from numpy import nonzero
from numpy import r_
from numpy import searchsorted
from numpy import unique
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range


MIN = "min"
MAX = "max"
LEFT = "left"
RIGHT = "right"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS = 1 / (1 - MAX_ROW_GROUP_SIZE_SCALE_FACTOR)


def get_region_indices_of_true_values(mask: NDArray) -> NDArray:
    """
    Compute the start and end indices of each connected components in `mask`.

    Taken from https://stackoverflow.com/questions/68514880/finding-contiguous-regions-in-a-1d-boolean-array.

    Parameters
    ----------
    mask : NDArray
        A 1d numpy array of dtype `bool`.

    Returns
    -------
    NDArray
        A numpy array containing the list of the start and end indices of each
        connected components in `mask`.

    """
    return flatnonzero(diff(r_[int8(0), mask.astype(int8), int8(0)])).reshape(-1, 2)


def get_region_indices_of_same_values(ints: NDArray) -> NDArray:
    """
    Compute the start and end indices of regions made of same values in `ints`.

    Parameters
    ----------
    ints : NDArray
        A 1d numpy array of dtype `int`.

    Returns
    -------
    NDArray
        A numpy array containing the list of the start and end indices of
        regions made of same values in `ints`.

    """
    boundaries = r_[0, flatnonzero(diff(ints)) + 1, len(ints)]
    return column_stack((boundaries[:-1], boundaries[1:]))


def get_region_start_end_delta(m_values: NDArray, indices: NDArray) -> NDArray:
    """
    Get difference between values at end and start of each region.

    Parameters
    ----------
    m_values : NDArray
        Array of monotonic values, such as coming from a cumulative sum.
    indices : NDArray
        Array of shape (n, 2) containing start included and end excluded indices
        of regions.

    Returns
    -------
    NDArray
        Array of length n containing the difference between values at end and
        start of each region.

    """
    if indices[0, 0] == 0:
        start_values = m_values[indices[:, 0] - 1]
        start_values[0] = 0
        return m_values[indices[:, 1] - 1] - start_values
    else:
        return m_values[indices[:, 1] - 1] - m_values[indices[:, 0] - 1]


class RGSizePattern(ABC):
    """
    Abstract base class for row group size patterns.
    """

    @abstractmethod
    def is_incomplete(self) -> NDArray:
        """
        Check if row groups are incomplete based on size pattern criteria.

        Returns
        -------
        NDArray
            Boolean array where True indicates incomplete row groups.

        """
        pass

    @abstractmethod
    def get_fragmentation_risk(
        self,
        indices_overlap: NDArray,
        indices_enlarged: NDArray,
    ) -> NDArray:
        """
        Check which enlarged regions get fragmentation risk from overlaps.

        Fragmentation in an existing set of row groups is when a set of
        contiguous incomplete row groups gets split by the addition of complete
        row groups in the middle of it.

        Parameters
        ----------
        indices_overlap : NDArray
            Array of shape (m, 2) containing start and end indices of overlap
            regions.
        indices_enlarged : NDArray
            Array of shape (n, 2) containing start included and end excluded
            indices of enlarged regions. Enlarged regions contain one or several
            overlap regions.

        Returns
        -------
        NDArray
            Boolean array of length n indicating which enlarged regions contain
            overlap regions that risk creating fragmentation when merged.

        """
        pass

    @abstractmethod
    def consolidate_merge_plan(
        self,
    ) -> Tuple[List[int], List[int]]:
        """
        Consolidate row groups and DataFrame chunks for merging.

        Returns
        -------
        Tuple[List[int], List[int]]
            - First list: indices in ParquetFile describing where ends (excluded)
              each set of row groups.
            - Second list: indices in DataFrame describing where ends (excluded)
              each chunk.

        """
        pass


class NRowsPattern(RGSizePattern):
    """
    Row group size pattern based on a target number of rows per row group.
    """

    def __init__(
        self,
        rg_n_rows: NDArray,
        df_n_rows: int,
        df_idx_tmrg_starts: NDArray,
        df_idx_tmrg_ends_excl: NDArray,
        row_group_size_target: int,
        irgs_allowed: bool,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        rg_n_rows : NDArray
            Number of rows in each row group.
        df_n_rows : int
            Number of rows in DataFrame.
        df_idx_tmrg_starts : NDArray
            Start indices (inclusive) in DataFrame for each row group overlap.
        df_idx_tmrg_ends_excl : NDArray
            End indices (exclusive) in DataFrame for each row group overlap.
        row_group_size_target : int
            Target number of rows for each row group.
        irgs_allowed : boolean
            Whether incomplete row groups are allowed.

        """
        self.rg_n_rows = rg_n_rows
        self.n_rgs = len(self.rg_n_rows)
        self.df_n_rows = df_n_rows
        self.df_idx_tmrg_starts = df_idx_tmrg_starts
        self.df_idx_tmrg_ends_excl = df_idx_tmrg_ends_excl
        self.rg_size_target = row_group_size_target
        self.min_size = int(row_group_size_target * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        self.irgs_allowed = irgs_allowed

    def is_incomplete(self) -> NDArray:
        """
        Check if row groups are incomplete based on size.

        Returns
        -------
        NDArray
            Boolean array where True indicates incomplete row groups.

        """
        return self.rg_n_rows < self.min_size

    def get_risk_of_exceeding_rgst(
        self,
        indices_overlap: NDArray,
        indices_enlarged: NDArray,
    ) -> NDArray:
        """
        Check which enlarged regions risk fragmentation.

        An overlap region risks fragmentation if the total number of rows (from
        row groups and DataFrame chunk) is above 'min_size', potentially
        creating a complete row group that would split a region of contiguous
        incomplete row groups.

        Rational of this logic is to prevent that the addition of new data after
        existing incomplete row groups creates an isolated set of incomplete row
        groups followed by complete row groups.

        Example:
          - Before addition:
            - rg   [crg] [crg] [irg] [irg]
            - df                        .....
          - Not desired after addition:
            - rg   [crg] [crg] [irg] [crg] [irg]
          - Desired after addition:
            - rg   [crg] [crg] [crg] [crg] [irg]

        Parameters
        ----------
        indices_overlap : NDArray
            Array of shape (o, 2) containing start and end indices of overlap
            regions.
        indices_enlarged : NDArray
            Array of shape (e, 2) containing start and end indices of enlarged
            regions (e <= o).

        Returns
        -------
        NDArray
            Boolean array of shape (e) indicating which enlarged regions contain
            overlap regions that risk creating fragmentation when merged.

        """
        # Step 1: For each overlap region, compute total rows (rg + df chunk)
        # and compare to min_size for risk of fragmentation.
        rg_rows_per_overlap_regions = get_region_start_end_delta(
            m_values=cumsum(self.rg_n_rows),
            indices=indices_overlap,
        )
        print("rg_rows_per_overlap_regions")
        print(rg_rows_per_overlap_regions)
        # TODO:
        # should include all rows of df that are over the overlap regions
        # as well as df chunk neighbor to these regions and incomplete row
        # groups
        # reword fragmentation_risk into "exceed_rgs_target"
        df_rows_per_overlap_regions = get_region_start_end_delta(
            m_values=self.df_idx_tmrg_ends_excl - self.df_idx_tmrg_starts - 1,
            indices=indices_overlap,
        )
        print("df_rows_per_overlap_regions")
        print(df_rows_per_overlap_regions)
        # 'overlap_has_risk' is an array of length the number of overlap
        # regions.
        overlap_has_risk = (
            rg_rows_per_overlap_regions + df_rows_per_overlap_regions >= self.min_size
        )

        # Step 2: Map risks to enlarged regions.
        # For each retained overlap region with fragmentation risk, find which
        # enlarged region contains its start.
        overlap_with_fragmentation_risk = zeros(len(indices_enlarged), dtype=bool)
        overlap_with_fragmentation_risk[
            unique(
                searchsorted(
                    indices_enlarged[:, 0],
                    indices_overlap[overlap_has_risk][:, 0],
                    side="right",
                ),
            )
            - 1
        ] = True
        return overlap_with_fragmentation_risk

    def consolidate_merge_plan(
        self,
    ) -> Tuple[List[int], List[int]]:
        """
        Sequence row groups and DataFrame chunks to be written together .

        Returns
        -------
        Tuple[List[int], List[int]]
            - First list: indices in ParquetFile describing where ends (excluded)
            each set of row groups to merge with corresponding DataFrame chunk.
            - Second list: indices in DataFrame describing where ends (excluded)
            each DataFrame chunk for the corresponding set of row groups.

        """
        if not self.df_idx_tmrg_ends_excl.size:
            # If no row group to merge, then no need to consolidate.
            return [0], [self.df_n_rows]

        min_n_rows = (
            self.min_size
            if self.irgs_allowed
            else int(self.min_size * MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS)
        )
        # Consolidation loop is processed backward.
        # This makes possible to manage a 'max_n_irgs' set to 0 (meaning no
        # incomplete row groups is allowed), by forcing the last chunk to
        # encompass
        # 'MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS * row_group_size_target' rows.
        # Then whatever the number of rows in the remainder of df, it will be
        # possible to yield chunk with a size between 'row_group_size_target' and
        # 'MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_size_target'.
        consolidated_df_idx_tmrg_ends_excl = []
        consolidated_rg_idx_tmrg_ends_excl = []
        chunk_n_rows = 0

        for m_rg_idx_end_excl, (_rg_n_rows, df_idx_end_excl) in enumerate(
            zip(reversed(self.rg_n_rows), reversed(self.df_idx_tmrg_ends_excl)),
            start=-self.n_rgs,
        ):
            # To make sure number of rows in chunk to write is larger than
            # 'row_group_size_target' despite possible duplicates between df and rg,
            # only account for rows in rg.
            chunk_n_rows += _rg_n_rows
            if chunk_n_rows >= min_n_rows:
                consolidated_df_idx_tmrg_ends_excl.append(df_idx_end_excl)
                consolidated_rg_idx_tmrg_ends_excl.append(-m_rg_idx_end_excl)
                chunk_n_rows = 0

        # Force df_idx_merge_end_excl to include all df rows
        consolidated_df_idx_tmrg_ends_excl[-1] = self.df_n_rows

        return (
            list(reversed(consolidated_rg_idx_tmrg_ends_excl)),
            list(reversed(consolidated_df_idx_tmrg_ends_excl)),
        )


class TimePeriodPattern(RGSizePattern):
    """
    Row group size pattern based on a time period target per row group.
    """

    def __init__(self, row_group_period: str):
        """
        Initialize scheme with target period.

        Parameters
        ----------
        row_group_period : str
            Target period for each row group (pandas freqstr).

        """
        self.period = row_group_period

    def is_incomplete(self, rg_maxs: NDArray) -> NDArray:
        """
        Check if row groups are incomplete based on period coverage.
        """
        # Convert maxs to period bounds
        period_bounds = [Timestamp(ts).floor(self.period) for ts in rg_maxs]
        # A row group is incomplete if it doesn't span its full period
        return array(
            [max < bound.ceil(self.period) for max, bound in zip(rg_maxs, period_bounds)],
        )

    def consolidate_merge_plan(
        self,
        rg_stats: NDArray,
        df_ordered_on: Series,
    ) -> Tuple[List[int], List[int]]:
        """
        Consolidate chunks based on time periods.
        """
        # Generate period starts
        start_ts = min(df_ordered_on.iloc[0], rg_stats[0]).floor(self.period)
        end_ts = max(df_ordered_on.iloc[-1], rg_stats[-1]).ceil(self.period)
        period_starts = date_range(start=start_ts, end=end_ts, freq=self.period)[1:]

        # Get indices for row groups and DataFrame chunks
        rg_idx_tmrg_ends_excl = searchsorted(rg_stats, period_starts, side="left")
        df_idx_period_ends_excl = searchsorted(df_ordered_on, period_starts, side="left")

        # Stack indices and find unique consecutive pairs
        pairs = vstack((rg_idx_tmrg_ends_excl, df_idx_period_ends_excl))
        # Find where consecutive pairs are different
        is_different = (pairs[:, 1:] != pairs[:, :-1]).any(axis=0)
        # Get indices where changes occur (including first and last)
        change_indices = r_[0, nonzero(is_different)[0] + 1]

        return (
            rg_idx_tmrg_ends_excl[change_indices].tolist(),
            df_idx_period_ends_excl[change_indices].tolist(),
        )


def get_merge_regions(
    rg_size_pattern: RGSizePattern,
    max_n_irgs: int,
    is_overlap: NDArray,
) -> NDArray:
    """
    Consolidate merge regions.

    Regions made of row groups overlapping with DataFrame chunks are extended to
    include neighbor incomplete row groups depending two conditions,
      - if the total number of existing incomplete row groups in the enlarged
        merge region is greater than `max_n_irgs`.
      - if the resulting row group(s) from the addition of the Dataframe chunks
        and corresponding overlapping row groups and neighbor incomplete row
        groups are possibly complete row groups.

    Parameters
    ----------
    is_overlap : NDArray
        Boolean array of shape (o).
    max_n_irgs : int
        Maximum number of incomplete row groups allowed in a merge region.

    Returns
    -------
    NDArray
        A numpy array containing the list of the start and end indices of each
        merge region.

    """
    # Step 1: assess start indices (included) and end indices (excluded) of
    # regions.
    indices_overlap = get_region_indices_of_true_values(is_overlap)
    is_incomplete = rg_size_pattern.is_incomplete()
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
    fragmentation_risk = rg_size_pattern.get_risk_of_exceeding_rgst(
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


def _get_atomic_merge_regions(
    rg_mins: List,
    rg_maxs: List,
    df_ordered_on: Series,
    drop_duplicates: bool,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Get atomic merge regions.

    An atomic merge region is
      - either defined by an existing row group in
    ParquetFile and if existing, its corresponding overlapping DataFrame chunk,
      - or a DataFrame chunk that is not overlapping with any row group in
        ParquetFile.

    Returned arrays provide the start and end (excluded) indices in row groups
    and end (excluded) indices in DataFrame for each of these atomic merge
    regions.
    These arrays are of same size.
    There is no need to provide the start indices in DataFrame, as they can be
    inferred from the end (excluded) indices in DataFrame of the previous atomic
    merge region (no part of the DataFrame is omitted for the write).

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
    Tuple[NDArray, NDArray, NDArray]
        - First NDArray: indices in ParquetFile containing the starts of each
          row group to merge with corresponding DataFrame chunk.
        - Second NDArray: indices in ParquetFile containing the ends (excluded)
          of each row group to merge with corresponding DataFrame chunk.
        - Third NDArray: indices in DataFrame containing the ends (excluded)
          of each DataFrame chunk to merge with corresponding row group.

    """
    # Find regions in DataFrame overlapping with row groups.
    if drop_duplicates:
        print("drop_duplicates")
        # Determine overlap start/end indices in row groups
        df_idx_tmrg_starts = searchsorted(df_ordered_on, rg_mins, side=LEFT)
        df_idx_tmrg_ends_excl = searchsorted(df_ordered_on, rg_maxs, side=RIGHT)
    else:
        print("no drop_duplicates")
        df_idx_tmrg_starts, df_idx_tmrg_ends_excl = searchsorted(
            df_ordered_on,
            vstack((rg_mins, rg_maxs)),
            side=LEFT,
        )
    print(f"df_idx_tmrg_starts: {df_idx_tmrg_starts}")
    print(f"df_idx_tmrg_ends_excl: {df_idx_tmrg_ends_excl}")
    # Find regions in DataFrame not overlapping with any row group.
    # `amr` for atomic merge region.
    df_idxs_enlarged = r_[
        df_idx_tmrg_starts[0],  # gap at start (0 to first start)
        df_idx_tmrg_starts[1:] - df_idx_tmrg_ends_excl[:-1],
        len(df_ordered_on) - df_idx_tmrg_ends_excl[-1],  # gap at end
    ]
    print(f"df_idxs_enlarged: {df_idxs_enlarged}")
    amr_idx_non_overlapping = flatnonzero(df_idxs_enlarged)
    print(f"amr_idx_non_overlapping: {amr_idx_non_overlapping}")
    rg_idxs = arange(len(rg_mins) + 1)
    print(f"rg_idxs: {rg_idxs}")
    if len(amr_idx_non_overlapping) == 0:
        # No non-overlapping regions in DataFrame
        return rg_idxs[:-1], rg_idxs[1:], df_idx_tmrg_ends_excl
    else:
        # Get insert accounting for previous insertions
        #        insert_positions = amr_idx_non_overlapping + arange(len(amr_idx_non_overlapping))
        #        print(f"insert_positions: {insert_positions}")
        # Fill arrays
        rg_idx_to_insert = rg_idxs[amr_idx_non_overlapping]
        print(f"rg_idx_to_insert: {rg_idx_to_insert}")
        rg_idxs_with_inserts = insert(rg_idxs, amr_idx_non_overlapping, rg_idx_to_insert)
        print(f"rg_idxs_with_inserts: {rg_idxs_with_inserts}")
        if amr_idx_non_overlapping[-1] == len(df_ordered_on):
            df_idx_to_insert = df_idx_tmrg_starts[amr_idx_non_overlapping]
        else:
            df_idx_to_insert = r_[df_idx_tmrg_starts, len(df_ordered_on)][amr_idx_non_overlapping]
        df_idx_with_inserts = insert(
            df_idx_tmrg_ends_excl,
            amr_idx_non_overlapping,
            df_idx_to_insert,
        )
        print(f"df_idx_with_inserts: {df_idx_with_inserts}")
        return rg_idxs_with_inserts[:-1], rg_idxs_with_inserts[1:], df_idx_with_inserts


def _rgst_as_str__irgs_analysis(
    rg_mins: NDArray,
    rg_idx_merge_start: int,
    rg_idx_merge_end_excl: int,
    row_group_size_target: str,
    df_min: Timestamp,
    df_max: Timestamp,
) -> Tuple[int, int, bool]:
    """
    Return neighbor incomplete row groups start and end indices (end excluded).

    Parameters
    ----------
    rg_mins : NDArray
        Minimum value of 'ordered_on' in each row group.
    rg_idx_merge_start : int
        Starting index of the current merge region in row groups.
    rg_idx_merge_end_excl : int
        Ending index (exclusive) of the current merge region in row groups.
    row_group_size_target : str
        Target row group size.
    df_min : Timestamp
        Value of 'ordered_on' at start of DataFrame.
    df_max : Timestamp
        Value of 'ordered_on' at end of DataFrame.

    Returns
    -------
    Tuple[int, int, bool]
        - First int: Start index of neighbor incomplete row groups (to the
          left).
        - Second int: End index (excluded) of neighbor incomplete row groups
          (to the right).
        - Third bool: True if new data is in distinct periods vs overlapping row
          groups. This will trigger coalescing of incomplete row groups.

    """
    # Include incomplete row groups to the left.
    left_neighbor_period_first_ts = Timestamp(rg_mins[rg_idx_merge_start]).floor(
        row_group_size_target,
    )
    while (
        rg_idx_merge_start > 0 and rg_mins[rg_idx_merge_start - 1] >= left_neighbor_period_first_ts
    ):
        rg_idx_merge_start -= 1

    # Include incomplete row groups to the right.
    right_neighbor_period_last_ts_excl = Timestamp(rg_mins[rg_idx_merge_end_excl - 1]).ceil(
        row_group_size_target,
    )
    while (
        rg_idx_merge_end_excl < len(rg_mins)
        and rg_mins[rg_idx_merge_end_excl] < right_neighbor_period_last_ts_excl
    ):
        rg_idx_merge_end_excl += 1

    left_neighbor_period_last_ts_excl = Timestamp(rg_mins[rg_idx_merge_start]).ceil(
        row_group_size_target,
    )
    right_neighbor_period_first_ts = Timestamp(rg_mins[rg_idx_merge_end_excl - 1]).floor(
        row_group_size_target,
    )
    should_coalesce = (
        df_max >= left_neighbor_period_last_ts_excl or df_min < right_neighbor_period_first_ts
    )
    return rg_idx_merge_start, rg_idx_merge_end_excl, should_coalesce


def _rgst_as_int__ensure_complete_rgs_by_including_neighbor_rgs(
    rg_n_rows: list[int],
    rg_idx_merge_start: int,
    rg_idx_merge_end_excl: int,
    n_missing_rows: int,
) -> Tuple[int, int]:
    """
    Walk neighbor row groups to ensure enough rows for complete row groups.

    This function attempts to reach the required number of rows by including
    additional row groups to the right, then to the left if needed.
    If insufficient, it returns indices to include all available row groups.

    Parameters
    ----------
    rg_n_rows : list[int]
        Number of rows in each row group.
    rg_idx_merge_start : int
        Starting index of the current merge region in row groups.
    rg_idx_merge_end_excl : int
        Ending index (exclusive) of the current merge region in row groups.
    n_missing_rows : int
        Number of additional rows needed to ensure complete row groups.

    Returns
    -------
    Tuple[int, int]
        New start and end indices (start, end_excl) for the expanded merge
        region. If insufficient rows are available, returns (0, len(rg_n_rows))
        to include all row groups.

    """
    # Try to include row groups to the right.
    for right_counter, n_rows in enumerate(rg_n_rows[rg_idx_merge_end_excl:], start=1):
        n_missing_rows -= n_rows
        if n_missing_rows <= 0:
            return rg_idx_merge_start, rg_idx_merge_end_excl + right_counter

    rg_idx_merge_end_excl += right_counter
    # Try to include row groups to the left.
    for left_counter, n_rows in enumerate(rg_n_rows[:rg_idx_merge_start:-1], start=1):
        n_missing_rows -= n_rows
        if n_missing_rows <= 0:
            return rg_idx_merge_start - left_counter, rg_idx_merge_end_excl

    # Include all available row groups if still insufficient.
    return 0, len(rg_n_rows)


def _rgst_as_int__merge_plan(
    rg_n_rows: List[int],
    min_n_rows_in_row_groups: int,
    df_n_rows: int,
    df_idx_tmrg_ends_excl: NDArray,
) -> Tuple[List[int], List[int]]:
    """
    Sequence row groups and DataFrame chunks to be written together.

    Parameters
    ----------
    rg_n_rows : list[int]
        Number of rows in each row group.
        This array has been trimmed to the overlapping region.
    min_n_rows_in_row_groups : int
        Number of rows in row groups to reach to gather row groups and DataFrame
        chunks.
    df_n_rows : int
        Number of rows in DataFrame.
    df_idx_tmrg_ends_excl : NDArray[int]
        Indices in DataFrame describing where ends (excluded) each DataFrame
        chunk that overlaps with a corresponding row group.
        This array has been trimmed to the overlapping region.

    Returns
    -------
    Tuple[List[int], List[int]]
        - First list: indices in ParquetFile describing where ends (excluded)
          each set of row groups to merge with corresponding DataFrame chunk.
        - Second list: indices in DataFrame describing where ends (excluded)
          each DataFrame chunk for the corresponding set of row groups.

    """
    if not df_idx_tmrg_ends_excl.size:
        # If no row group to merge, then no need to consolidate.
        return [0], [df_n_rows]

    # Consolidation loop is processed backward.
    # This makes possible to manage a 'max_n_irgs' set to 0 (meaning no
    # incomplete row groups is allowed), by forcing the last chunk to
    # encompass
    # 'MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS * row_group_size_target' rows.
    # Then whatever the number of rows in the remainder of df, it will be
    # possible to yield chunk with a size between 'row_group_size_target' and
    # 'MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_size_target'.
    consolidated_df_idx_tmrg_ends_excl = []
    consolidated_rg_idx_tmrg_ends_excl = []
    chunk_n_rows = 0
    for m_rg_idx_end_excl, (_rg_n_rows, df_idx_end_excl) in enumerate(
        zip(
            reversed(df_idx_tmrg_ends_excl),
            reversed(rg_n_rows),
        ),
        start=-len(rg_n_rows),
    ):
        # To make sure number of rows in chunk to write is larger than
        # 'row_group_size_target' despite possible duplicates between df and rg,
        # only account for rows in rg.
        chunk_n_rows += _rg_n_rows
        if chunk_n_rows >= min_n_rows_in_row_groups:
            consolidated_df_idx_tmrg_ends_excl.append(df_idx_end_excl)
            consolidated_rg_idx_tmrg_ends_excl.append(-m_rg_idx_end_excl)
            chunk_n_rows = 0

    # Force df_idx_merge_end_excl to include all df rows.
    consolidated_df_idx_tmrg_ends_excl[-1] = df_n_rows

    return reversed(consolidated_rg_idx_tmrg_ends_excl), reversed(
        consolidated_df_idx_tmrg_ends_excl,
    )


def _rgst_as_str__merge_plan(
    rg_maxs: NDArray,
    row_group_period: str,
    df_ordered_on: Series,
) -> Tuple[List[int], List[int]]:
    """
    Sequence row groups and DataFrame chunks to be written together.

    Parameters
    ----------
    rg_maxs : NDArray
        Maximum value of 'ordered_on' in each row group.
    row_group_period : str
        Period of row groups.
    df_ordered_on : Series[Timestamp]
        Values of 'ordered_on' column in DataFrame.

    Returns
    -------
    Tuple[List[int], List[int]]
        - First list: indices in ParquetFile describing where ends (excluded)
          each set of row groups to merge with corresponding DataFrame chunk.
        - Second list: indices in DataFrame describing where ends (excluded)
          each DataFrame chunk for the corresponding set of row groups.

    """
    # Generate period starts.
    # Note
    # `floor` is used because we consider valid periods bounds are
    # [inclusive_min, exclusive_max[.
    # Only in `date_range` do we then skip the 1st lower bound.
    # Previous implementation used `ceil` to get the 2nd lower bound directly,
    # but this is incorrect in case ts = ts.ceil = ts.floor.
    start_ts = min(df_ordered_on.iloc[0], rg_maxs[0]).floor(row_group_period)
    end_ts = max(df_ordered_on.iloc[-1], rg_maxs[-1]).ceil(row_group_period)
    period_starts = date_range(start=start_ts, end=end_ts, freq=row_group_period)[1:]
    # Get indices for row groups and DataFrame chunks.
    rg_idx_tmrg_ends_excl = searchsorted(rg_maxs, period_starts, side=LEFT)
    df_idx_period_ends_excl = searchsorted(df_ordered_on, period_starts, side=LEFT)

    # Stack indices and find unique consecutive pairs.
    pairs = vstack((rg_idx_tmrg_ends_excl, df_idx_period_ends_excl))
    # Find where consecutive pairs are different.
    is_different = (pairs[:, 1:] != pairs[:, :-1]).any(axis=0)
    # Get indices where changes occur (including first and last).
    change_indices = r_[0, nonzero(is_different)[0] + 1]

    return (
        rg_idx_tmrg_ends_excl[change_indices].tolist(),
        df_idx_period_ends_excl[change_indices].tolist(),
    )


def analyze_chunks_to_merge(
    pf: ParquetFile,
    df: DataFrame,
    ordered_on: str,
    row_group_size_target: Union[int, str],
    drop_duplicates: bool,
    max_n_irgs: Optional[int] = None,
) -> Tuple[List[int], bool]:
    """
    Describe how DataFrame and ParquetFile chunks can be merged.

    Important: because this function returns indices in input DataFrame, if
    duplicates are dropped, this function should be applied on a DataFrame
    without duplicates.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame. Must contain the 'ordered_on' column.
    pf : ParquetFile
        Input ParquetFile. Must contain statistics for the 'ordered_on'
        column.
    ordered_on : str
        Column name by which data is ordered. Must exist in both DataFrame
        and ParquetFile.
    row_group_size_target : Union[int, str]
        Target row group size.
    drop_duplicates : bool
        Flag impacting how overlap boundaries have to be managed.
        More exactly, 'pf' is considered as first data, and 'df' as second
        data, coming after. In case of 'pf' leading 'df', if last value in
        'pf' is a duplicate of the first in 'df', then
        - If True, at this index, overlap starts
        - If False, no overlap at this index

    max_n_irgs : Optional[int]
        Max allowed number of 'incomplete' row groups.
        - ``None`` value induces no coalescing of row groups. If there is
            no drop of duplicates, new data is systematically appended.
        - A value of ``0`` means that new data will likely be merged to the last
          existing row groups to attempt making complete row groups.
        - A value of ``1`` means that new data will be merged to the last
          existing row group only if it is not 'complete'.

    Returns
    -------
    Tuple[List[int], List[int], bool]
        - First list: indices in DataFrame describing where each DataFrame chunk
          ends (excluded).
        - Second list: indices in ParquetFile describing where each set of row
          groups ends (excluded) to merge with corresponding DataFrame chunk.
        - Boolean: flag indicating if row groups need to be resorted after
          merge.

    Notes
    -----
    When a row in pf shares the same value in 'ordered_on' column as a row
    in df, the row in pf is considered as leading the row in df i.e.
    anterior to it.
    This has an impact in overlap identification in case of not dropping
    duplicates.

    When 'max_n_irgs' is specified, the method will analyze neighbor incomplete
    row groups and may adjust the overlap region to include them for coalescing
    if necessary.

    """
    df_n_rows = len(df)
    # /!\ TODO: remove 2 below variables. Are specific to IntRGSizePattern.
    rg_n_rows = [rg.num_rows for rg in pf.row_groups]
    n_rgs = len(rg_n_rows)
    # df_min = df.loc[:, ordered_on].iloc[0]
    #    # Find overlapping regions in dataframe
    #    rg_mins = pf.statistics[MIN][ordered_on]
    #    rg_maxs = pf.statistics[MAX][ordered_on]
    #    if drop_duplicates:
    #        print("drop_duplicates")
    # Determine overlap start/end indices in row groups
    #        df_idx_tmrg_starts = searchsorted(df.loc[:, ordered_on], rg_mins, side=LEFT)
    #        df_idx_tmrg_ends_excl = searchsorted(df.loc[:, ordered_on], rg_maxs, side=RIGHT)
    #    else:
    #        print("no drop_duplicates")
    #        df_idx_tmrg_starts, df_idx_tmrg_ends_excl = searchsorted(
    #            df.loc[:, ordered_on],
    #            vstack((rg_mins, rg_maxs)),
    #            side=LEFT,
    #        )
    pf_statistics = pf.statistics
    rg_idx_starts, rg_idx_ends_excl, df_idx_ends_excl = _get_atomic_merge_regions(
        rg_mins=pf_statistics[MIN][ordered_on],
        rg_maxs=pf_statistics[MAX][ordered_on],
        df_ordered_on=df.loc[:, ordered_on],
        drop_duplicates=drop_duplicates,
    )

    # Get overlap status per row groups.
    is_overlap = 1
    #    is_overlap = df_idx_tmrg_starts == df_idx_tmrg_ends_excl - 1

    #    if not df_idx_tmrg_ends_excl[-1]:
    #        # df after last row group in pf.
    #        rg_idx_merge_start = rg_idx_merge_end_excl = n_rgs
    #    elif df_idx_tmrg_starts[0] == df_n_rows:
    #        # df before first row group in pf.
    #        rg_idx_merge_start = rg_idx_merge_end_excl = 0
    #    else:
    #        print("df within pf")
    #        rg_idx_merge_start = df_idx_tmrg_ends_excl.astype(bool).argmax()
    #        # df overlaps with pf.
    #        # Then index of last row group to merge is the row group before the first
    #        # row group starting after the end of df.
    #        # Second case, df is overlapping with last row group in pf.
    #        rg_idx_merge_end_excl = (
    #            df_idx_tmrg_starts.argmax()
    #            if df_idx_tmrg_starts[-1] == df_n_rows
    #            else df_idx_tmrg_ends_excl.argmax() + 1
    #        )

    # Get map of incomplete row groups, and if they should be merged.
    is_complete = rg_n_rows >= row_group_size_target * MAX_ROW_GROUP_SIZE_SCALE_FACTOR
    #    is_non_overlapping = (df_idx_tmrg_starts == df_idx_tmrg_ends_excl - 1)
    #    splits_merge_regions = is_complete & is_non_overlapping
    #    splits_merge_regions_starts = nonzero(diff(splits_merge_regions) != 0)[0]#
    #    splits_merge_regions_ends_excl = nonzero(splits_merge_regions_starts)[0] + 1

    #       # Identify areas of non-overlapping complete row groups, which will
    #        # delimit merge regions.
    #        df_idx_tmrg_starts = df_idx_tmrg_starts[rg_idx_merge_start:rg_idx_merge_end_excl]
    #        df_idx_tmrg_ends_excl = df_idx_tmrg_ends_excl[rg_idx_merge_start:rg_idx_merge_end_excl]
    #        #print("df_idx_tmrg_starts")
    #        #print(df_idx_tmrg_starts)
    #        print("df_idx_tmrg_ends_excl")
    #        print(df_idx_tmrg_ends_excl)

    # Get row group start and end indices of overlap regions, merging those that
    # are contiguous by incomplete row groups.
    # Assess if merged overlap region need to be fully rewritten by checking if
    # 'max_nirgs' is reached per overlap region.
    rg_idx_merge_starts, rg_idx_merge_ends_excl = get_merge_regions(
        is_overlap=is_overlap,
        is_complete=is_complete,
    )

    #        # Starts of all potential regions, i.e. indices where the current index
    #        # differs from its predecessor.
    #        is_region_start = r_[True, diff(df_idx_tmrg_starts) != 0]
    #        is_region_end = roll(is_region_start, -1)
    #        is_empty = (df_idx_tmrg_starts == df_idx_tmrg_ends_excl - 1)
    #        empty_interval_starts = nonzero(is_region_start & is_empty)[0]
    #        empty_interval_ends_excl = nonzero(is_region_end & is_empty)[0] + 1

    print("")
    print("before irgs analysis")
    print(f"df_n_rows: {df_n_rows}")
    #    print(f"df_idx_tmrg_starts: {df_idx_tmrg_starts}")
    # print(f"df_idx_tmrg_ends_excl: {df_idx_tmrg_ends_excl}")
    print(f"rg_idx_merge_start: {rg_idx_merge_starts}")
    print(f"rg_idx_merge_end_excl: {rg_idx_merge_ends_excl}")
    # Assess if neighbor incomplete row groups in ParquetFile have to be
    # included in the merge.
    # if max_n_irgs is not None:
    #    rg_idx_merge_start, rg_idx_merge_end_excl = _include_neighbord_irgs(
    #        rg_n_rows=rg_n_rows,
    #        rg_mins=rg_mins,
    #        rg_idx_merge_start=rg_idx_merge_start,
    #        rg_idx_merge_end_excl=rg_idx_merge_end_excl,
    #        row_group_size_target=row_group_size_target,
    #        max_n_irgs=max_n_irgs,
    #        df_n_rows=df_n_rows,
    #        df_min=df.loc[:, ordered_on].iloc[0],
    #        df_max=df.loc[:, ordered_on].iloc[-1],
    #    )
    #    print("")
    #    print("after irgs analysis")
    #    print(f"rg_idx_merge_start: {rg_idx_merge_start}")
    #    print(f"rg_idx_merge_end_excl: {rg_idx_merge_end_excl}")

    # TODO:
    # if areas of complete non overlapping row groups, split df chunk to process
    # each area separately.
    # combine with below to re-extend if needed to maintain complete row groups.

    if max_n_irgs == 0:
        # Specific case where only 'complete' row groups should result from the
        # merge. This is possible if total number of rows in row groups to merge
        # is larger than 1/(1-min_row_group_size) x df_size.
        min_rgs_size_target = row_group_size_target * MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS
        if (
            n_missing_rows := (
                min_rgs_size_target - sum(rg_n_rows[rg_idx_merge_starts:rg_idx_merge_ends_excl])
            )
        ) >= 0:
            (
                rg_idx_merge_start,
                rg_idx_merge_end_excl,
            ) = _rgst_as_int__ensure_complete_rgs_by_including_neighbor_rgs(
                rg_n_rows=rg_n_rows,
                rg_idx_merge_start=rg_idx_merge_starts,
                rg_idx_merge_end_excl=rg_idx_merge_ends_excl,
                n_missing_rows=n_missing_rows,
            )
        print("after max_n_irgs=0 analysis")
        #        print(f"df_idx_tmrg_starts: {df_idx_tmrg_starts}")
        # print(f"df_idx_tmrg_ends_excl: {df_idx_tmrg_ends_excl}")

    # Row groups have to be sorted after write step
    #  - either if there is a merge region (new row groups are written first,
    #    then old row groups are removed).
    #  - or there is no merge region but df is not appended at the tail of
    #    existing data.
    sort_rgs_after_write = rg_idx_merge_start != rg_idx_merge_end_excl or (
        rg_idx_merge_start == rg_idx_merge_end_excl and rg_idx_merge_end_excl != n_rgs
    )

    # TODO:
    # get below into a loop to process each area.

    # Trim row group related lists to the overlapping region.
    print("trimming step")
    print("rg_idx_merge_start")
    print(rg_idx_merge_start)
    print("rg_idx_merge_end_excl")
    print(rg_idx_merge_end_excl)

    rg_idx_tmrg_ends_excl, df_idx_tmrg_ends_excl = (
        _rgst_as_int__merge_plan(
            rg_n_rows=rg_n_rows,
            min_n_rows_in_row_groups=min_rgs_size_target
            if max_n_irgs == 0
            else row_group_size_target,
            df_n_rows=df_n_rows,
            #            df_idx_tmrg_ends_excl=df_idx_tmrg_ends_excl,
            df_idx_tmrg_ends_excl=df_idx_ends_excl,
        )
        if isinstance(row_group_size_target, int)
        else _rgst_as_str__merge_plan(
            #            rg_maxs=rg_maxs,
            row_group_period=row_group_size_target,
            df_ordered_on=df.loc[:, ordered_on],
        )
    )

    return (
        rg_idx_merge_start,
        rg_idx_tmrg_ends_excl,
        df_idx_tmrg_ends_excl,
        sort_rgs_after_write,
    )
