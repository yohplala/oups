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
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import List, Tuple, Union

from numpy import arange
from numpy import array
from numpy import bool_
from numpy import cumsum
from numpy import diff
from numpy import flatnonzero
from numpy import insert
from numpy import int_
from numpy import nonzero
from numpy import ones
from numpy import r_
from numpy import searchsorted
from numpy import unique
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.utils import get_region_start_end_delta


LEFT = "left"
RIGHT = "right"
HAS_ROW_GROUP = "has_row_group"
HAS_DF_CHUNK = "has_df_chunk"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS = 1 / (1 - MAX_ROW_GROUP_SIZE_SCALE_FACTOR)


def compute_atomic_merge_regions(
    rg_mins: List[Union[int, float, Timestamp]],
    rg_maxs: List[Union[int, float, Timestamp]],
    df_ordered_on: Series,
    drop_duplicates: bool,
) -> NDArray:
    """
    Compute atomic merge regions.

    An atomic merge region ('amr') is
      - either defined by an existing row group in ParquetFile and its
        corresponding overlapping DataFrame chunk if any,
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
        - 'has_df_chunk': boolean indicating if this region contains a
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
            ("rg_idx_start", int_),
            ("rg_idx_end_excl", int_),
            ("df_idx_end_excl", int_),
            (HAS_ROW_GROUP, bool_),
            (HAS_DF_CHUNK, bool_),
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


class AMRSplitStrategy(ABC):
    """
    Abstract base class for atomic merge region split strategies.

    This class defines strategies for:
    - evaluating likelihood of region completeness after merge,
    - determining appropriate sizes for new row groups,
    - consolidating merge plans for efficient write operations.

    """

    @cached_property
    @abstractmethod
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which AMRs are likely to meet target size.

        This can be the result of 2 conditions:
        - either a DataFrame chunk and a row group are merged together, and the
          result is likely roght sized (not under-sized, nor over-sized).
        - or if there is only a Dataframe chunk or only a row group, they are
          likely right sized on their own.

        Returns
        -------
        NDArray
            Boolean array of length the number of atomic merge regions.

        Notes
        -----
        The likelihood aspect results from the fact that the final completeness
        of an AMR cannot be determined at analysis time for various reasons
        depending on the split strategy.

        """
        raise NotImplementedError("Subclasses must implement this property")

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
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_row_group_size(self, chunk_len: int, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Parameters
        ----------
        chunk_len : int
            Length of the chunk to process.
        is_last_chunk : bool
            Whether this is the last chunk in the iteration.

        Returns
        -------
        Union[int, List[int]]
            Either a single integer (for uniform splitting) or a list of indices
            where the chunk should be split

        """
        raise NotImplementedError("Subclasses must implement this method")


class NRowsSplitStrategy(AMRSplitStrategy):
    """
    Row group split strategy based on a target number of rows per row group.
    """

    def __init__(
        self,
        amrs_info: NDArray,
        rgs_n_rows: NDArray,
        df_n_rows: int,
        row_group_target_size: int,
        max_n_irgs: int,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        amrs_info : NDArray
            Array of shape (e, 5) containing the information about each atomic
            merge regions.
        rgs_n_rows : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        df_n_rows : int
            Number of rows in DataFrame.
        row_group_target_size : int
            Target number of rows above which a new row group should be created.
        max_n_irgs : int
            Maximum number of incomplete row groups allowed in a merge region.

        """
        # Max number of rows in each atomic merge region. This is a max in case
        # there are duplicates between row groups and DataFrame that will be
        # dropped.
        self.amrs_max_n_rows = zeros(len(amrs_info), dtype=int)
        self.amrs_max_n_rows[amrs_info[HAS_ROW_GROUP]] = rgs_n_rows
        self.amrs_max_n_rows += diff(amrs_info["df_idx_end_excl"], prepend=0)
        self.df_n_rows = df_n_rows
        self.amr_target_size = row_group_target_size
        self.amr_min_size = int(row_group_target_size * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        self.max_n_irgs = max_n_irgs

    @cached_property
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which AMRs are likely to meet target size.

        The likelihood aspect comes from the fact that at analysis time, we don't
        know the exact final size of an AMR. When a row group and DataFrame chunk
        overlap are both present in an AMR, there might be duplicate rows that
        will be removed during the merge operation.
        We know the maximum size (sum of row group and DataFrame chunk sizes)
        We know the minimum size (max of row group size and DataFrame chunk size)
        But we don't know the exact final size until after deduplication

        Returns
        -------
        NDArray
            Boolean array of length the number of atomic merge regions.

        """
        return (self.amrs_max_n_rows >= self.amr_min_size) & (
            self.amrs_max_n_rows <= self.amr_target_size
        )

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
                    side=RIGHT,
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
            if self.max_n_irgs
            else int(self.min_size * MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS)
        )
        # Consolidation loop is processed backward.
        # This makes possible to manage a 'max_n_irgs' set to 0 (meaning no
        # incomplete row groups is allowed), by forcing the last chunk to
        # encompass
        # 'MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS * row_group_target_size' rows.
        # Then whatever the number of rows in the remainder of df, it will be
        # possible to yield chunk with a size between 'row_group_target_size' and
        # 'MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_target_size'.
        consolidated_df_idx_tmrg_ends_excl = []
        consolidated_rg_idx_tmrg_ends_excl = []
        chunk_n_rows = 0

        for m_rg_idx_end_excl, (_rg_n_rows, df_idx_end_excl) in enumerate(
            zip(reversed(self.rg_n_rows), reversed(self.df_idx_tmrg_ends_excl)),
            start=-self.n_rgs,
        ):
            # To make sure number of rows in chunk to write is larger than
            # 'row_group_target_size' despite possible duplicates between df and rg,
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

    def get_row_group_size(self, chunk_len: int, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Logic varies based on `max_n_irgs` setting and whether this is the last
        chunk.

        """
        if self.max_n_irgs == 0 or chunk_len <= self.row_group_target_size:
            # Always uniform splitting for max_n_irgs = 0
            return self.row_group_target_size
        else:
            # When a max_n_irgs is set larger than 0, then
            # calculate split points to create chunks of target size
            # and potentially a smaller final piece.
            return list(range(chunk_len, step=self.row_group_target_size))

    def _rgst_as_int__ensure_complete_rgs_by_including_neighbor_rgs(
        self,
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
        self,
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
        # 'MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS * row_group_target_size' rows.
        # Then whatever the number of rows in the remainder of df, it will be
        # possible to yield chunk with a size between 'row_group_target_size' and
        # 'MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_target_size'.
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
            # 'row_group_target_size' despite possible duplicates between df and rg,
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


class TimePeriodSplitStrategy(AMRSplitStrategy):
    """
    Row group split strategy based on a time period target per row group.
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
        Check if row groups are incomplete based on a time period.

        Returns
        -------
        NDArray
            Boolean array of length the number of row groups where True
            indicates incomplete row groups.

        """
        # Convert maxs to period bounds
        period_bounds = [Timestamp(ts).floor(self.period) for ts in rg_maxs]
        # A row group is incomplete if it doesn't span its full period
        return array(
            [max < bound.ceil(self.period) for max, bound in zip(rg_maxs, period_bounds)],
            dtype=bool,
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

    def _rgst_as_str__irgs_analysis(
        self,
        rg_mins: NDArray,
        rg_idx_merge_start: int,
        rg_idx_merge_end_excl: int,
        row_group_target_size: str,
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
        row_group_target_size : str
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
            row_group_target_size,
        )
        while (
            rg_idx_merge_start > 0
            and rg_mins[rg_idx_merge_start - 1] >= left_neighbor_period_first_ts
        ):
            rg_idx_merge_start -= 1

        # Include incomplete row groups to the right.
        right_neighbor_period_last_ts_excl = Timestamp(rg_mins[rg_idx_merge_end_excl - 1]).ceil(
            row_group_target_size,
        )
        while (
            rg_idx_merge_end_excl < len(rg_mins)
            and rg_mins[rg_idx_merge_end_excl] < right_neighbor_period_last_ts_excl
        ):
            rg_idx_merge_end_excl += 1

        left_neighbor_period_last_ts_excl = Timestamp(rg_mins[rg_idx_merge_start]).ceil(
            row_group_target_size,
        )
        right_neighbor_period_first_ts = Timestamp(rg_mins[rg_idx_merge_end_excl - 1]).floor(
            row_group_target_size,
        )
        should_coalesce = (
            df_max >= left_neighbor_period_last_ts_excl or df_min < right_neighbor_period_first_ts
        )
        return rg_idx_merge_start, rg_idx_merge_end_excl, should_coalesce

    def _rgst_as_str__merge_plan(
        self,
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

    @property
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which AMRs are likely to meet target size.

        The likelihood aspect comes from the temporal nature of the data and how
        it might evolve:
        - An AMR might currently be the only one in its time period, making it
          correctly sized.
        - However, future merges might add new AMRs to the same time period.
        - When multiple AMRs exist in the same period, they become under-sized
          and they may eventually be merged together.

        Therefore, an AMR that appears to meet the target size during one
        analysis might become under-sized in a future analysis when new data
        arrives in the same time period.

        Returns
        -------
        NDArray
            Boolean array of length the number of atomic merge regions.

        """
        return ...  # implementation
