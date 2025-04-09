#!/usr/bin/env python3
"""
Created on Mon Mar 17 18:00:00 2025.

Ordered atomic regions for Parquet files and DataFrames.

This module defines the base functions for analyzing how DataFrame can be merged
with existing Parquet files when both are ordered on the same column.
An ordered atomic region ('oar') represents the smallest unit for merging, which
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
from numpy import bincount
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
from pandas import DataFrame
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.utils import ceil_ts
from oups.store.utils import floor_ts
from oups.store.utils import get_region_start_end_delta


LEFT = "left"
RIGHT = "right"
RG_IDX_START = "rg_idx_start"
RG_IDX_END_EXCL = "rg_idx_end_excl"
DF_IDX_END_EXCL = "df_idx_end_excl"
HAS_ROW_GROUP = "has_row_group"
HAS_DF_CHUNK = "has_df_chunk"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
MIN_RG_NUMBER_TO_ENSURE_COMPLETE_RGS = 1 / (1 - MAX_ROW_GROUP_SIZE_SCALE_FACTOR)


def compute_ordered_atomic_regions(
    rg_ordered_on_mins: NDArray,
    rg_ordered_on_maxs: NDArray,
    df_ordered_on: Series,
    drop_duplicates: bool,
) -> NDArray:
    """
    Compute ordered atomic regions (OARs) from row groups and DataFrame.

    An ordered atomic region is either:
      - A row group and its overlapping DataFrame chunk (if any)
      - A DataFrame chunk that doesn't overlap with any row group

    Returned arrays provide the start and end (excluded) indices in row groups
    and end (excluded) indices in DataFrame for each of these ordered atomic
    regions. All these arrays are of same size and describe how are composed the
    ordered atomic regions.

    Parameters
    ----------
    rg_ordered_on_mins : NDArray[Timestamp]
        Minimum values of 'ordered_on' in each row group.
    rg_ordered_on_maxs : NDArray[Timestamp]
        Maximum values of 'ordered_on' in each row group.
    df_ordered_on : Series[Timestamp]
        Values of 'ordered_on' column in DataFrame.
    drop_duplicates : bool
        Flag impacting how overlapping boundaries have to be managed.
        More exactly, row groups are considered as first data, and DataFrame as
        second data, coming after. In case of a row group leading a DataFrame
        chunk, if the last value in row group is a duplicate of a value in
        DataFrame chunk, then
        - If True, at this index, overlap starts
        - If False, no overlap at this index

    Returns
    -------
    NDArray
        Structured array with fields:
        - rg_idx_start: Start indices of row groups in each OAR
        - rg_idx_end_excl: End indices (exclusive) of row groups in each OAR
        - df_idx_end_excl: End indices (exclusive) of DataFrame chunks in each OAR
        - has_row_group: Boolean indicating if OAR contains a row group
        - has_df_chunk: Boolean indicating if OAR contains a DataFrame chunk

    Raises
    ------
    ValueError
        If input arrays have inconsistent lengths or unsorted data.

    Notes
    -----
    Start indices in DataFrame are not provided, as they can be inferred from
    the end (excluded) indices in DataFrame of the previous ordered atomic region
    (no part of the DataFrame is omitted for the write).

    In case 'drop_duplicates' is False, and there are duplicate values between
    row group max values and DataFrame 'ordered_on' values, then DataFrame
    'ordered_on' values are considered to be the last occurrences of the
    duplicates in 'ordered_on'. Leading row groups (with duplicate max values)
    will not be in the same ordered atomic region as the DataFrame chunk starting
    at the duplicate 'ordered_on' value. This is an optimization to prevent
    rewriting these leading row groups.

    """
    # Validate 'ordered_on' in row groups and DataFrame.
    if len(rg_ordered_on_mins) != len(rg_ordered_on_maxs):
        raise ValueError("rg_ordered_on_mins and rg_ordered_on_maxs must have the same length.")
    # Check that rg_maxs[i] is less than rg_mins[i+1] (no overlapping row groups).
    if len(rg_ordered_on_mins) > 1 and (rg_ordered_on_maxs[:-1] > rg_ordered_on_mins[1:]).any():
        raise ValueError("row groups must not overlap.")
    # Check that df_ordered_on is sorted.
    if not df_ordered_on.is_monotonic_increasing:
        raise ValueError("'df_ordered_on' must be sorted in ascending order.")

    if drop_duplicates:
        # Determine overlap start/end indices in row groups
        df_idx_oar_starts = searchsorted(df_ordered_on, rg_ordered_on_mins, side=LEFT)
        df_idx_oar_ends_excl = searchsorted(df_ordered_on, rg_ordered_on_maxs, side=RIGHT)
    else:
        df_idx_oar_starts, df_idx_oar_ends_excl = searchsorted(
            df_ordered_on,
            vstack((rg_ordered_on_mins, rg_ordered_on_maxs)),
            side=LEFT,
        )
    # Find regions in DataFrame not overlapping with any row group.
    df_chunk_wo_overlap = r_[
        df_idx_oar_starts[0],  # gap at start (0 to first start)
        df_idx_oar_ends_excl[:-1] - df_idx_oar_starts[1:],
        len(df_ordered_on) - df_idx_oar_ends_excl[-1],  # gap at end
    ]
    # Indices in row groups where a DataFrame chunk is not overlapping with any
    # row group.
    rg_idx_df_orphans = flatnonzero(df_chunk_wo_overlap)
    n_rgs = len(rg_ordered_on_mins)
    rg_idxs_template = arange(n_rgs + 1)
    # Create a structured array to hold all related indices
    # DataFrame orphans are regions in DataFrame that do not overlap with any
    # row group.
    n_df_orphans = len(rg_idx_df_orphans)
    oars_info = ones(
        n_rgs + n_df_orphans,
        dtype=[
            (RG_IDX_START, int_),
            (RG_IDX_END_EXCL, int_),
            (DF_IDX_END_EXCL, int_),
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
        # 'Resize 'df_idx_oar_ends_excl', and duplicate values where there are
        # non-overlapping regions in DataFrame.
        if rg_idx_df_orphans[-1] == len(df_ordered_on):
            df_idx_to_insert = df_idx_oar_starts[rg_idx_df_orphans]
        else:
            df_idx_to_insert = r_[df_idx_oar_starts, len(df_ordered_on)][rg_idx_df_orphans]
        df_idx_oar_ends_excl = insert(
            df_idx_oar_ends_excl,
            rg_idx_df_orphans,
            df_idx_to_insert,
        )

    oars_info[RG_IDX_START] = rg_idxs_template[:-1]
    oars_info[RG_IDX_END_EXCL] = rg_idxs_template[1:]
    oars_info[DF_IDX_END_EXCL] = df_idx_oar_ends_excl
    oars_info[HAS_ROW_GROUP] = oars_info[RG_IDX_START] != oars_info[RG_IDX_END_EXCL]
    oars_info[HAS_DF_CHUNK] = diff(oars_info[DF_IDX_END_EXCL], prepend=0) != 0
    return oars_info


class OARSplitStrategy(ABC):
    """
    Abstract base class for ordered atomic region split strategies.

    This class defines strategies for:
    - evaluating likelihood of region completeness after merge,
    - determining appropriate sizes for new row groups,
    - consolidating merge plans for efficient write operations.

    """

    @cached_property
    @abstractmethod
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to meet target size.

        This can be the result of 2 conditions:
        - either a DataFrame chunk and a row group are merged together, and the
          result is likely right sized (not under-sized, nor over-sized).
        - or if there is only a Dataframe chunk or only a row group, they are
          right sized on their own.

        Returns
        -------
        NDArray
            Boolean array of length the number of ordered atomic regions.

        Notes
        -----
        The likelihood aspect results from the fact that the final completeness
        of an OAR cannot be determined at analysis time for various reasons
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
    def get_row_group_size(self, chunk: DataFrame, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Parameters
        ----------
        chunk : DataFrame
            DataFrame chunk to process.
        is_last_chunk : bool
            Whether this is the last chunk in the iteration.

        Returns
        -------
        Union[int, List[int]]
            Either a single integer (for uniform splitting) or a list of indices
            where the chunk should be split

        """
        raise NotImplementedError("Subclasses must implement this method")


class NRowsSplitStrategy(OARSplitStrategy):
    """
    Row group split strategy based on a target number of rows per row group.

    This strategy ensures that row groups are split when they exceed a target
    size, while maintaining a minimum size to prevent too small row groups. It
    also handles incomplete row groups through the max_n_irgs parameter.

    Attributes
    ----------
    oar_target_size : int
        Target number of rows above which a new row group should be created.
    oar_min_size : int
        Minimum number of rows in an ordered atomic region, computed as
        ``MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_target_size``.
    max_n_irgs : int
        Maximum number of incomplete row groups allowed in an ordered atomic
        region.
    oars_max_n_rows : NDArray
        Array of shape (e) containing the maximum number of rows in each ordered
        atomic region.

    Notes
    -----
    The maximum number of rows in an OAR is calculated as the sum of:
    - The number of rows in the row group (if present)
    - The number of rows in the DataFrame chunk (if present)
    This represents the worst-case scenario where there are no duplicates.

    """

    def __init__(
        self,
        rgs_n_rows: NDArray,
        oars_info: NDArray,
        row_group_target_size: int,
        max_n_irgs: int,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        rgs_n_rows : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        oars_info : NDArray
            Array of shape (e, 5) containing the information about each ordered
            atomic region.
        row_group_target_size : int
            Target number of rows above which a new row group should be created.
        max_n_irgs : int
            Maximum number of incomplete row groups allowed in an ordered atomic
            region.

        """
        self.oar_target_size = row_group_target_size
        self.oar_min_size = int(row_group_target_size * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        self.max_n_irgs = max_n_irgs
        # Max number of rows in each ordered atomic region. This is a max in case
        # there are duplicates between row groups and DataFrame that will be
        # dropped.
        self.oars_max_n_rows = zeros(len(oars_info), dtype=int)
        self.oars_max_n_rows[oars_info[HAS_ROW_GROUP]] = rgs_n_rows
        self.oars_max_n_rows += diff(oars_info[DF_IDX_END_EXCL], prepend=0)
        # self.df_n_rows = oars_info[DF_IDX_END_EXCL][-1]

    @cached_property
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to meet target size.

        An OAR is considered likely to meet target size if its maximum possible
        size (sum of row group and DataFrame chunk sizes) is between the minimum
        and target sizes. This is a conservative estimate since the actual size
        after deduplication could be smaller.

        Returns
        -------
        NDArray
            Boolean array of length equal to the number of ordered atomic regions,
            where True indicates the OAR is likely to meet the target size.

        Notes
        -----
        The likelihood aspect comes from the fact that at analysis time, the
        exact final size of an OAR is not known due to potential duplicates
        between row groups and DataFrame chunks. The maximum size is used as a
        conservative estimate.

        """
        return (self.oars_max_n_rows >= self.oar_min_size) & (
            self.oars_max_n_rows <= self.oar_target_size
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
        # TODO:
        # should include all rows of df that are over the overlap regions
        # as well as df chunk neighbor to these regions and incomplete row
        # groups
        # reword fragmentation_risk into "exceed_rgs_target"
        df_rows_per_overlap_regions = get_region_start_end_delta(
            m_values=self.df_idx_tmrg_ends_excl - self.df_idx_tmrg_starts - 1,
            indices=indices_overlap,
        )
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
              each set of row groups overlapping with corresponding DataFrame
              chunk.
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

    def get_row_group_size(self, chunk: DataFrame, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Logic varies based on `max_n_irgs` setting and whether this is the last
        chunk.

        """
        if self.max_n_irgs == 0 or len(chunk) <= self.row_group_target_size:
            # Always uniform splitting for max_n_irgs = 0
            return self.row_group_target_size
        else:
            # When a max_n_irgs is set larger than 0, then
            # calculate split points to create chunks of target size
            # and potentially a smaller final piece.
            return list(range(len(chunk), step=self.row_group_target_size))


class TimePeriodSplitStrategy(OARSplitStrategy):
    """
    Row group split strategy based on a time period target per row group.

    This strategy ensures that row groups are split based on time periods. Each
    resulting row group will ideally contain data from a single time period
    (e.g., a month, day, etc.).

    Attributes
    ----------
    oar_period : str
        Time period for a row group to be complete (e.g., 'MS' for month start).
    oars_mins_maxs : NDArray
        Array of shape (e, 2) containing the start and end bounds of each
        ordered atomic region.
    single_component_oars : NDArray
        Array of shape (e) containing booleans indicating whether each ordered
        atomic region has exactly one component (either RG or DF chunk, not both).
    period_bounds : DatetimeIndex
        Period bounds over the total time span of the dataset, considering both
        row groups and DataFrame.

    Notes
    -----
    - A row group is considered meeting the target size if it contains data from
      exactly one time period.
    - The strategy tries to maintain temporal locality by keeping data from the same
      time period together.

    """

    def __init__(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        oars_info: NDArray,
        row_group_period: str,
    ):
        """
        Initialize scheme with target period.

        Parameters
        ----------
        rg_ordered_on_mins : NDArray
            Minimum value of 'ordered_on' in each row group.
        rg_ordered_on_maxs : NDArray
            Maximum value of 'ordered_on' in each row group.
        df_ordered_on : Series
            Values of 'ordered_on' column in DataFrame.
        oars_info : NDArray
            Array of shape (e, 5) containing the information about each ordered
            atomic region.
        row_group_period : str
            Target period for each row group (pandas freqstr).

        """
        self.oar_period = row_group_period
        df_ordered_on_np = df_ordered_on.to_numpy()
        self.oars_mins_maxs = ones((len(oars_info), 2)).astype(df_ordered_on_np.dtype)
        # Row groups encompasses Dataframe chunks in an OAR.
        # Hence, start with Dataframe chunks starts and ends.
        oar_idx_df_chunk = flatnonzero(oars_info[HAS_DF_CHUNK])
        df_idx_chunk_starts = zeros(len(oar_idx_df_chunk), dtype=int)
        df_idx_chunk_starts[1:] = oars_info[DF_IDX_END_EXCL][oar_idx_df_chunk[:-1]]
        self.oars_mins_maxs[oar_idx_df_chunk, 0] = df_ordered_on_np[df_idx_chunk_starts]
        self.oars_mins_maxs[oar_idx_df_chunk, 1] = df_ordered_on_np[
            oars_info[DF_IDX_END_EXCL][oar_idx_df_chunk] - 1
        ]
        # Only then add row groups starts and ends. They will overwrite where
        # Dataframe chunks are present.
        oar_idx_row_groups = flatnonzero(oars_info[HAS_ROW_GROUP])
        self.oars_mins_maxs[oar_idx_row_groups, 0] = rg_ordered_on_mins
        self.oars_mins_maxs[oar_idx_row_groups, 1] = rg_ordered_on_maxs
        # Keep track where there is only one component (either RG or DF chunk,
        # not both).
        self.single_component_oars = oars_info[HAS_ROW_GROUP] ^ oars_info[HAS_DF_CHUNK]
        # Generate period bounds.
        start_ts = floor_ts(Timestamp(self.oars_mins_maxs[0, 0]), row_group_period)
        end_ts = ceil_ts(Timestamp(self.oars_mins_maxs[-1, 1]), row_group_period)
        self.period_bounds = date_range(start=start_ts, end=end_ts, freq=row_group_period)

    @cached_property
    def likely_meets_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to meet target size.

        An OAR meets target size if and only if:
         - It contains exactly one row group OR one DataFrame chunk (not both)
         - That component fits entirely within a single period bound

        Returns
        -------
        NDArray
            Boolean array of length equal to the number of ordered atomic regions,
            where True indicates the OAR is likely to meet the target size.

        Notes
        -----
        The likelihood aspect comes from the fact that the dataset is mutable,
        and additional chunks may be added at a next run. However, this has no
        impact the logic implemented.

        """
        # Find period indices for each OAR.
        period_idx_oars = searchsorted(
            self.period_bounds,
            self.oars_mins_maxs,
            side=RIGHT,
        )
        # Check if OAR fits in a single period
        single_period_oars = period_idx_oars[:, 0] == period_idx_oars[:, 1]
        # Check if OAR is the only one in its period
        period_counts = bincount(period_idx_oars.ravel())
        # Each period index has to appear only twice (oncee for start, once for end).
        # Since we already checked OARs don't span multiple periods (start == end),
        # the check is then only made on the period start.
        oars_single_in_period = period_counts[period_idx_oars[:, 0]] == 2
        return (
            self.single_component_oars  # Single component check
            & single_period_oars  # Single period check
            & oars_single_in_period  # Single OAR in period check
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
        start_ts = min(df_ordered_on.iloc[0], rg_stats[0]).floor(self.oar_period)
        end_ts = max(df_ordered_on.iloc[-1], rg_stats[-1]).ceil(self.oar_period)
        period_starts = date_range(start=start_ts, end=end_ts, freq=self.oar_period)[1:]

        # Get indices for row groups and DataFrame chunks
        rg_idx_tmrg_ends_excl = searchsorted(rg_stats, period_starts, side=LEFT)
        df_idx_period_ends_excl = searchsorted(df_ordered_on, period_starts, side=LEFT)

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

    def get_row_group_size(self, chunk: DataFrame, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Parameters
        ----------
        chunk : DataFrame
            DataFrame chunk to process.
        is_last_chunk : bool
            Whether this is the last chunk in the iteration.

        Returns
        -------
        Union[int, List[int]]
            Either a single integer (for uniform splitting) or a list of indices
            where the chunk should be split

        """
        raise NotImplementedError("Subclasses must implement this method")
