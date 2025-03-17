#!/usr/bin/env python3
"""
Created on Tue Feb 25 18:00:00 2025.

@author: yoh

"""

from abc import ABC
from abc import abstractmethod
from typing import List, Tuple, Union

from numpy import array
from numpy import cumsum
from numpy import diff
from numpy import nonzero
from numpy import r_
from numpy import searchsorted
from numpy import unique
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.atomic_merge_regions import HAS_DF_CHUNK
from oups.store.atomic_merge_regions import HAS_ROW_GROUP
from oups.store.utils import get_region_start_end_delta


LEFT = "left"
RIGHT = "right"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS = 1 / (1 - MAX_ROW_GROUP_SIZE_SCALE_FACTOR)


class MergeRegionSplitStrategy(ABC):
    """
    Abstract base class for merge region split strategies.

    This class defines strategies for:
    - evaluating likelihood of region completeness after merge,
    - determining appropriate sizes for new row groups,
    - consolidating merge plans for efficient write operations.

    """

    @property
    @abstractmethod
    def has_df_chunk(self) -> NDArray:
        """
        Return boolean array indicating which merge regions have DataFrame chunks.

        Returns
        -------
        NDArray
            Boolean array of length the number of atomic merge regions.

        """
        raise NotImplementedError("Subclasses must implement this property")

    @property
    @abstractmethod
    def likely_complete(self) -> NDArray:
        """
        Return boolean array indicating which atomic merge regions is likely complete.

        This can be the result of 2 conditions:
        - either a DataFrame chunk and a row group are merged together, and the
          result is likely complete.
        - or if there is only a DatAframe chunk or only a row group, they are
          complete by themselves.

        Returns
        -------
        NDArray
            Boolean array of length the number of atomic merge regions.

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


class NRowsSplitStrategy(MergeRegionSplitStrategy):
    """
    Row group split strategy based on a target number of rows per row group.
    """

    def __init__(
        self,
        amrs_info: NDArray,
        n_rows_in_rgs: NDArray,
        df_n_rows: int,
        row_group_size_target: int,
        max_n_irgs: int,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        amrs_info : NDArray
            Array of shape (e, 5) containing the information about each atomic
            merge regions.
        n_rows_in_rgs : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        df_n_rows : int
            Number of rows in DataFrame.
        row_group_size_target : int
            Target number of rows above which a new row group should be created.
        max_n_irgs : int
            Maximum number of incomplete row groups allowed in a merge region.

        """
        # Max number of rows in each atomic merge region. This is a max in case
        # there are duplicates between row groups and DataFrame that will be
        # dropped.
        self.max_n_rows_in_amrs = zeros(len(amrs_info), dtype=int)
        self.max_n_rows_in_amrs[amrs_info[HAS_ROW_GROUP]] = n_rows_in_rgs
        self.max_n_rows_in_amrs[amrs_info[HAS_DF_CHUNK]] += diff(
            amrs_info["df_idx_end_excl"],
            prepend=0,
        )
        self.df_n_rows = df_n_rows
        self.rg_size_target = row_group_size_target
        self.min_size = int(row_group_size_target * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        self.max_n_irgs = max_n_irgs

    def is_incomplete(self) -> NDArray:
        """
        Check if row groups are incomplete based on their number of rows.

        Returns
        -------
        NDArray
            Boolean array of length the number of row groups where True
            indicates incomplete row groups.

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

    def get_row_group_size(self, chunk_len: int, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Logic varies based on `max_n_irgs` setting and whether this is the last
        chunk.

        """
        if self.max_n_irgs == 0 or chunk_len <= self.row_group_size_target:
            # Always uniform splitting for max_n_irgs = 0
            return self.row_group_size_target
        else:
            # When a max_n_irgs is set larger than 0, then
            # calculate split points to create chunks of target size
            # and potentially a smaller final piece.
            return list(range(chunk_len, step=self.row_group_size_target))

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


class TimePeriodSplitStrategy(MergeRegionSplitStrategy):
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
            rg_idx_merge_start > 0
            and rg_mins[rg_idx_merge_start - 1] >= left_neighbor_period_first_ts
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
