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
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import flatnonzero
from numpy import insert
from numpy import linspace
from numpy import maximum
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


LEFT = "left"
RIGHT = "right"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
# MIN_RG_NUMBER_TO_ENSURE_ON_TARGET_RGS = 1 / (1 - MAX_ROW_GROUP_SIZE_SCALE_FACTOR)


class OARSplitStrategy(ABC):
    """
    Abstract base class for ordered atomic region split strategies.

    This class defines strategies for:
    - evaluating likelihood of row groups being on target size after merge,
    - determining appropriate sizes for new row groups,
    - consolidating merge plans for efficient write operations.

    An OAR is considered 'on target size' if it meets the target size criteria
    (either in terms of number of rows or time period). Otherwise it is
    considered 'off target size'.

    Attributes
    ----------
    oars_rg_idx_starts : NDArray[int_]
        Start indices of row groups in each OAR.
    oars_cmpt_idx_ends_excl : NDArray[int_, int_]
        End indices (excluded) of row groups and DataFrame chunks in each OAR.
    oars_has_row_group : NDArray[bool_]
        Boolean array indicating if OAR contains a row group.
    oars_df_n_rows : NDArray[int_]
        Number of rows in each DataFrame chunk in each OAR.
    oars_has_df_chunk : NDArray[bool_]
        Boolean array indicating if OAR contains a DataFrame chunk.
    n_oars : int
        Number of ordered atomic regions.

    """

    def __init__(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        drop_duplicates: bool,
    ):
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
        n_rgs = len(rg_ordered_on_mins)
        n_df_rows = len(df_ordered_on)
        if n_rgs != len(rg_ordered_on_maxs):
            raise ValueError("rg_ordered_on_mins and rg_ordered_on_maxs must have the same length.")
        # Check that rg_maxs[i] is less than rg_mins[i+1] (no overlapping row groups).
        if n_rgs > 1 and (rg_ordered_on_maxs[:-1] > rg_ordered_on_mins[1:]).any():
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
        # DataFrame orphans are regions in DataFrame that do not overlap with
        # any row group. Find indices in row groups of DataFrame orphans.
        rg_idx_df_orphans = flatnonzero(
            r_[
                df_idx_oar_starts[0],  # gap at start (0 to first start)
                df_idx_oar_ends_excl[:-1] - df_idx_oar_starts[1:],
                n_df_rows - df_idx_oar_ends_excl[-1],  # gap at end
            ],
        )
        n_df_orphans = len(rg_idx_df_orphans)
        rg_idxs_template = arange(n_rgs + 1)
        if n_df_orphans != 0:
            # Case of non-overlapping regions in DataFrame.
            # Resize 'rg_idxs', and duplicate values where there are
            # non-overlapping regions in DataFrame.
            rg_idx_to_insert = rg_idxs_template[rg_idx_df_orphans]
            rg_idxs_template = insert(
                rg_idxs_template,
                rg_idx_df_orphans,
                rg_idx_to_insert,
            )
            # 'Resize 'df_idx_oar_ends_excl', and duplicate values where there
            # are non-overlapping regions in DataFrame.
            df_idx_to_insert = r_[df_idx_oar_starts, n_df_rows][rg_idx_df_orphans]
            df_idx_oar_ends_excl = insert(
                df_idx_oar_ends_excl,
                rg_idx_df_orphans,
                df_idx_to_insert,
            )

        self.oars_rg_idx_starts = rg_idxs_template[:-1]
        self.oars_cmpt_idx_ends_excl = column_stack((rg_idxs_template[1:], df_idx_oar_ends_excl))
        self.oars_has_row_group = rg_idxs_template[:-1] != rg_idxs_template[1:]
        self.oars_df_n_rows = diff(df_idx_oar_ends_excl, prepend=0)
        self.oars_has_df_chunk = self.oars_df_n_rows.astype(bool)
        self.n_oars = len(self.oars_rg_idx_starts)

    @abstractmethod
    def specialized_init(self, **kwargs):
        """
        Initialize specialized attributes.

        This method provides a base method to initialize specialized attributes
        for testing purpose of other methods of the OARSPlitStrategy abstract
        class.
        It is intended to be overridden by subclasses to initialize additional
        attributes specific to the strategy.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to initialize specialized attributes.

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_oars_desc(
        cls,
        oars_rg_idx_starts: NDArray,
        oars_cmpt_idx_ends_excl: NDArray,
        oars_has_row_group: NDArray,
        **kwargs,
    ) -> "OARSplitStrategy":
        """
        Create a strategy instance with a given OARs description.

        This is primarily for testing purposes, allowing tests to directly set
        the OARSplitStrategy base attributes without having to compute it from
        row groups and DataFrame.

        Parameters
        ----------
        oars_rg_idx_starts : NDArray
            Start indices of row groups in each OAR.
        oars_cmpt_idx_ends_excl : NDArray
            End indices (excluded) of row groups and DataFrame chunks in each
            OAR.
        oars_has_row_group : NDArray
            Boolean array indicating if OAR contains a row group.
        drop_duplicates : bool
            Whether to drop duplicates between row groups and DataFrame.
        **kwargs
            Additional arguments needed by specific strategy implementations.

        Returns
        -------
        OARSplitStrategy
            An instance of the strategy with the given OARs description.

        """
        instance = cls.__new__(cls)
        instance.oars_rg_idx_starts = oars_rg_idx_starts
        instance.oars_cmpt_idx_ends_excl = oars_cmpt_idx_ends_excl
        instance.oars_has_row_group = oars_has_row_group
        instance.oars_df_n_rows = diff(oars_cmpt_idx_ends_excl[:, 1], prepend=0)
        instance.oars_has_df_chunk = instance.oars_df_n_rows.astype(bool)
        instance.n_oars = len(oars_rg_idx_starts)
        instance.specialized_init(**kwargs)
        return instance

    @cached_property
    @abstractmethod
    def likely_on_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to be on target size.

        This can be the result of 2 conditions:
        - either a single DataFrame chunk or the merge of a Dataframe chunk and
          a row group, with a (resulting) size that is on target size
          (not under-sized, but over-sized is accepted).
        - or if there is only a row group, it is on target size on its own.

        Returns
        -------
        NDArray
            Boolean array of length the number of ordered atomic regions.

        Notes
        -----
        The logic implements an asymmetric treatment of OARs with and without
        DataFrame chunks to prevent fragmentation and ensure proper compliance
        with split strategy, including off target existing row groups.

        1. For OARs containing a DataFrame chunk:
            - Writing is always triggered (systematic).
            - If oversized, considered on target to force rewrite of neighbor
              already existing off target row groups.
            - If undersized, considered off target to be properly accounted for
              when comparing to 'max_n_off_target_rgs'.
           This ensures that writing a new on target row group will trigger
           the rewrite of adjacent off target row groups when
           'max_n_off_target_rgs' is set.

        2. For OARs containing only row groups:
            - Writing is triggered only if:
               * The OAR is within a set of contiguous off target OARs
                 (under or over sized) and neighbors an OAR with a DataFrame
                 chunk.
               * Either the number of off target OARs exceeds
                 'max_n_off_target_rgs',
               * or an OAR to be written (with DataFrame chunk) will induce
                 writing of a row group likely to be on target.
            - Considered off target if either under or over sized to ensure
              proper accounting when comparing to 'max_n_off_target_rgs'.

        This approach ensures:
        - All off target row groups are captured for potential rewrite.
        - Writing an on target new row group forces rewrite of all adjacent
          already existing off target row groups (under or over sized).
        - Fragmentation is prevented by consolidating off target regions when
          such *full* rewrite is triggered.

        """
        raise NotImplementedError("Subclasses must implement this property")

    @abstractmethod
    def partition_merge_regions(
        self,
        oar_idx_mrs_starts_ends_excl: NDArray,
    ) -> List[Tuple[int, NDArray]]:
        """
        Partition merge regions (MRs) into optimally sized chunks for writing.

        Returns
        -------
        List[Tuple[int, NDArray]]
            List of tuples, where each tuple contains for each merge sequence:
            - First element: Start index of the first row group in the merge
              sequence.
            - Second element: Array of shape (m, 2) containing end indices
              (excluded) for row groups and DataFrame chunks in the merge
              sequence.

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
    also handles off target size row groups through the 'max_n_off_target_rgs'
    parameter.

    Attributes
    ----------
    row_group_target_size : int
        Target number of rows above which a new row group should be created.
    row_group_min_size : int
        Minimum number of rows in an ordered atomic region, computed as
        ``MAX_ROW_GROUP_SIZE_SCALE_FACTOR * row_group_target_size``.
    oars_max_n_rows : NDArray
        Array of shape (e) containing the maximum number of rows in each ordered
        atomic region, obtained by summing the number of rows in a row group
        (if present) and the number of rows in its corresponding DataFrame chunk
        (if present).
    oars_min_n_rows : NDArray
        Array of shape (e) containing the likely  minimum number of rows in each
        ordered atomic region. It is equal to ``oars_max_n_rows`` if
        ``drop_duplicates`` is False.

    Notes
    -----
    The maximum number of rows in an OAR is calculated as the sum of:
    - The number of rows in the row group (if present)
    - The number of rows in the DataFrame chunk (if present)
    This represents the worst-case scenario where there are no duplicates.

    """

    def __init__(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        drop_duplicates: bool,
        rgs_n_rows: NDArray,
        row_group_target_size: int,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        rg_ordered_on_mins : NDArray
            Array of shape (r) containing the minimum values of the ordered
            row groups.
        rg_ordered_on_maxs : NDArray
            Array of shape (r) containing the maximum values of the ordered
            row groups.
        df_ordered_on : Series
            Series of shape (d) containing the ordered DataFrame.
        drop_duplicates : bool
            Whether to drop duplicates between row groups and DataFrame.
        rgs_n_rows : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        row_group_target_size : int
            Target number of rows above which a new row group should be created.

        """
        super().__init__(
            rg_ordered_on_mins,
            rg_ordered_on_maxs,
            df_ordered_on,
            drop_duplicates,
        )
        self.specialized_init(
            rgs_n_rows=rgs_n_rows,
            row_group_target_size=row_group_target_size,
            drop_duplicates=drop_duplicates,
        )

    def specialized_init(
        self,
        rgs_n_rows: NDArray,
        row_group_target_size: int,
        drop_duplicates: bool,
    ):
        """
        Initialize scheme with target size.

        Parameters
        ----------
        rgs_n_rows : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        row_group_target_size : int
            Target number of rows above which a new row group should be created.
        drop_duplicates : bool
            Whether to drop duplicates between row groups and DataFrame.

        """
        self.row_group_target_size = row_group_target_size
        self.row_group_min_size = int(row_group_target_size * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        # Max number of rows in each ordered atomic region. This is a max in case
        # there are duplicates between row groups and DataFrame that will be
        # dropped.
        self.oars_max_n_rows = zeros(self.n_oars, dtype=int)
        self.oars_max_n_rows[self.oars_has_row_group] = rgs_n_rows
        if drop_duplicates:
            # Assuming each DataFrame chunk and each row group have no
            # duplicates within themselves, 'oars_min_n_rows' is set assuming
            # that all rows in the smallest component are duplicates of rows
            # in the largest component.
            self.oars_min_n_rows = maximum(self.oars_max_n_rows, self.oars_df_n_rows)
        else:
            self.oars_min_n_rows = self.oars_max_n_rows
        self.oars_max_n_rows += self.oars_df_n_rows

    @cached_property
    def likely_on_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to be on target size.

        An OAR is considered likely to be on target size if:
        - for OARs containing a DataFrame chunk, its maximum possible size is
          above the minimum size. This is an estimate since the actual size
          after deduplication could be smaller.
        - for OARs containing only row groups, their maximum possible size is
          between the minimum and target sizes.

        Returns
        -------
        NDArray
            Boolean array of length equal to the number of ordered atomic
            regions, where True indicates the OAR is likely to be on target
            size.

        Notes
        -----
        The logic implements an asymmetric treatment of OARs with and without
        DataFrame chunks to prevent fragmentation and ensure proper compliance
        with split strategy, including off target existing row groups.

        1. For OARs containing a DataFrame chunk:
            - Writing is always triggered (systematic).
            - If oversized, considered on target to force rewrite of neighbor
              already existing off target row groups.
            - If undersized, considered off target to be properly accounted for
              when comparing to 'max_n_off_target_rgs'.
           This ensures that writing a new on target row group will trigger
           the rewrite of adjacent off target row groups when
           'max_n_off_target_rgs' is set.

        2. For OARs containing only row groups:
            - Writing is triggered only if:
               * The OAR is within a set of contiguous off target OARs
                 (under or over sized) and neighbors an OAR with a DataFrame
                 chunk.
               * Either the number of off target OARs exceeds
                 'max_n_off_target_rgs',
               * or an OAR to be written (with DataFrame chunk) will induce
                 writing of a row group likely to be on target.
            - Considered off target if either under or over sized to ensure
              proper accounting when comparing to 'max_n_off_target_rgs'.

        This approach ensures:
        - All off target row groups are captured for potential rewrite.
        - Writing an on target new row group forces rewrite of all adjacent
          already existing off target row groups (under or over sized).
        - Fragmentation is prevented by consolidating off target regions when
          such *full* rewrite is triggered.

        """
        return self.oars_has_df_chunk & (  # OAR containing a DataFrame chunk.
            self.oars_max_n_rows >= self.row_group_min_size
        ) | ~self.oars_has_df_chunk & (  # OAR containing only row groups.
            (self.oars_max_n_rows >= self.row_group_min_size)
            & (self.oars_max_n_rows <= self.row_group_target_size)
        )

    def partition_merge_regions(
        self,
        oar_idx_mrs_starts_ends_excl: NDArray,
    ) -> List[Tuple[int, NDArray]]:
        """
        Partition merge regions (MRs) into optimally sized chunks for writing.

        For each merge region (MR) defined in 'oar_idx_mrs_starts_ends_excl',
        this method:
        1. Accumulates row counts using self.oars_min_n_rows
        2. Determines split points where accumulated rows reach
           'self.row_group_target_size'
        3. Creates consolidated chunks by filtering the original OARs indices to
           ensure optimal row group loading.

        This ensures each consolidated chunk approaches the target row size
        while minimizing the number of row groups loaded into memory at once.

        Parameters
        ----------
        oar_idx_mrs_starts_ends_excl : NDArray
            Array of shape (e, 2) containing start and end indices (excluded)
            for each merge region to be consolidated.

        Returns
        -------
        List[Tuple[int, NDArray]]
            List of tuples, where each tuple contains for each merge sequence:
            - First element: Start index of the first row group in the merge
              sequence.
            - Second element: Array of shape (m, 2) containing end indices
              (excluded) for row groups and DataFrame chunks in the merge
              sequence.

        Notes
        -----
        The partitioning optimizes memory usage by loading only the minimum
        number of row groups needed to create complete chunks of approximately
        target size rows. The returned indices may be a subset of the original
        OARs indices, filtered to ensure efficient memory usage during the
        write process.

        """
        oars_merge_sequences = []
        for oar_idx_start, oar_idx_end_excl in oar_idx_mrs_starts_ends_excl:
            rg_idx_start = self.oars_rg_idx_starts[oar_idx_start]
            # Cumulate number of rows.
            cumsum_rows = cumsum(self.oars_min_n_rows[oar_idx_start:oar_idx_end_excl])
            n_target_size_multiples = cumsum_rows[-1] // self.row_group_target_size
            # Get indices where multiples of target size are crossed.
            if self.row_group_target_size <= cumsum_rows[-1]:
                target_size_crossings = unique(
                    searchsorted(
                        cumsum_rows,
                        # Using linspace instead of arange to include endpoint.
                        linspace(
                            self.row_group_target_size,
                            self.row_group_target_size * n_target_size_multiples,
                            n_target_size_multiples,
                            endpoint=True,
                        ),
                        side=LEFT,
                    ),
                )
            else:
                target_size_crossings = zeros(1, dtype=int)
            # Force last index to be length of cumsum_rows
            target_size_crossings[-1] = oar_idx_end_excl - oar_idx_start - 1
            # Create a structured array with the filtered indices
            oars_merge_sequences.append(
                (
                    rg_idx_start,
                    self.oars_cmpt_idx_ends_excl[oar_idx_start:oar_idx_end_excl][
                        target_size_crossings
                    ],
                ),
            )

        return oars_merge_sequences

    def get_row_group_size(self, chunk: DataFrame, is_last_chunk: bool) -> Union[int, List[int]]:
        """
        Define the appropriate row group size or split points for a chunk.

        This size is defined in terms of number of rows. Result is to be used
        as `row_group_size` parameter in `iter_dataframe` method.

        Logic varies based on `max_n_off_target_rgs` setting and whether this is
        the last chunk.

        """
        if self.max_n_off_target_rgs == 0 or len(chunk) <= self.row_group_target_size:
            # Always uniform splitting for max_n_off_target_rgs = 0
            return self.row_group_target_size
        else:
            # When a max_n_off_target_rgs is set larger than 0, then
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
    row_group_time_period : str
        Time period for a row group to be on target size (e.g., 'MS' for month
        start).
    oars_mins_maxs : NDArray
        Array of shape (e, 2) containing the start and end bounds of each
        ordered atomic region.
    period_bounds : DatetimeIndex
        Period bounds over the total time span of the dataset, considering both
        row groups and DataFrame.
    oars_period_idx : NDArray
        Array of shape (e, 2) containing the start and end indices of each
        ordered atomic region in the period bounds.

    Notes
    -----
    - A row group is considered meeting the target size if it contains data from
      exactly one time period.
    - A point in time is within a time period if it is greater than or equal to
      period start and strictly less than period end.

    """

    def __init__(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        drop_duplicates: bool,
        row_group_time_period: str,
    ):
        """
        Initialize scheme with time size.

        Parameters
        ----------
        rg_ordered_on_mins : NDArray
            Array of shape (r) containing the minimum values of the ordered
            row groups.
        rg_ordered_on_maxs : NDArray
            Array of shape (r) containing the maximum values of the ordered
            row groups.
        df_ordered_on : Series
            Series of shape (d) containing the ordered DataFrame.
        drop_duplicates : bool
            Whether to drop duplicates between row groups and DataFrame.
        row_group_time_period : str
            Target period for each row group (pandas freqstr).

        """
        super().__init__(
            rg_ordered_on_mins,
            rg_ordered_on_maxs,
            df_ordered_on,
            drop_duplicates,
        )
        self.specialized_init(
            rg_ordered_on_mins,
            rg_ordered_on_maxs,
            df_ordered_on,
            row_group_time_period,
        )

    def specialized_init(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        row_group_time_period: str,
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
        row_group_time_period : str
            Expected time period for each row group (pandas freqstr).

        """
        self.row_group_time_period = row_group_time_period
        df_ordered_on_np = df_ordered_on.to_numpy()
        self.oars_mins_maxs = ones((self.n_oars, 2)).astype(df_ordered_on_np.dtype)
        # Row groups encompasses Dataframe chunks in an OAR.
        # Hence, start with Dataframe chunks starts and ends.
        oar_idx_df_chunk = flatnonzero(self.oars_has_df_chunk)
        df_idx_chunk_starts = zeros(len(oar_idx_df_chunk), dtype=int)
        df_idx_chunk_starts[1:] = self.oars_cmpt_idx_ends_excl[oar_idx_df_chunk[:-1], 1]
        self.oars_mins_maxs[oar_idx_df_chunk, 0] = df_ordered_on_np[df_idx_chunk_starts]
        self.oars_mins_maxs[oar_idx_df_chunk, 1] = df_ordered_on_np[
            self.oars_cmpt_idx_ends_excl[oar_idx_df_chunk, 1] - 1
        ]
        # Only then add row groups starts and ends. They will overwrite where
        # Dataframe chunks are present.
        oar_idx_row_groups = flatnonzero(self.oars_has_row_group)
        self.oars_mins_maxs[oar_idx_row_groups, 0] = rg_ordered_on_mins
        self.oars_mins_maxs[oar_idx_row_groups, 1] = rg_ordered_on_maxs
        # Generate period bounds.
        start_ts = floor_ts(Timestamp(self.oars_mins_maxs[0, 0]), row_group_time_period)
        end_ts = ceil_ts(Timestamp(self.oars_mins_maxs[-1, 1]), row_group_time_period)
        self.period_bounds = date_range(start=start_ts, end=end_ts, freq=row_group_time_period)
        # Find period indices for each OAR.
        self.oars_period_idx = searchsorted(
            self.period_bounds,
            self.oars_mins_maxs,
            side=RIGHT,
        )

    @cached_property
    def likely_on_target_size(self) -> NDArray:
        """
        Return boolean array indicating which OARs are likely to be on target size.

        An OAR meets target size if and only if:
         - It contains exactly one row group OR one DataFrame chunk (not both)
         - That component fits entirely within a single period bound

        Returns
        -------
        NDArray
            Boolean array of length equal to the number of ordered atomic
            regions, where True indicates the OAR is likely to be on target
            size.

        Notes
        -----
        The logic implements an asymmetric treatment of OARs with and without
        DataFrame chunks to prevent fragmentation and ensure proper compliance
        with split strategy, including off target existing row groups.

        1. For OARs containing a DataFrame chunk:
            - Writing is always triggered (systematic).
            - If oversized, considered on target to force rewrite of neighbor
              already existing off target row groups.
            - If undersized, considered off target to be properly accounted for
              when comparing to 'max_n_off_target_rgs'.
           This ensures that writing a new on target row group will trigger
           the rewrite of adjacent off target row groups when
           'max_n_off_target_rgs' is set.

        2. For OARs containing only row groups:
            - Writing is triggered only if:
               * The OAR is within a set of contiguous off target OARs
                 (under or over sized) and neighbors an OAR with a DataFrame
                 chunk.
               * Either the number of off target OARs exceeds
                 'max_n_off_target_rgs',
               * or an OAR to be written (with DataFrame chunk) will induce
                 writing of a row group likely to be on target size.
            - Considered off target if either under or over sized to ensure
              proper accounting when comparing to 'max_n_off_target_rgs'.

        This approach ensures:
        - All off target row groups are captured for potential rewrite.
        - Writing an on target new row group forces rewrite of all adjacent
          already existing off target row groups (under or over sized).
        - Fragmentation is prevented by consolidating off target regions when
          such *full* rewrite is triggered.

        """
        # Check if OAR fits in a single period
        single_period_oars = self.oars_period_idx[:, 0] == self.oars_period_idx[:, 1]
        # Check if OAR is the only one in its period
        # period_counts = bincount(period_idx_oars.ravel())
        # Each period index has to appear only twice (oncee for start, once for end).
        # Since we already checked OARs don't span multiple periods (start == end),
        # the check is then only made on the period start.
        # oars_single_in_period = period_counts[period_idx_oars[:, 0]] == 2
        return (  # Over-sized OAR containing a DataFrame chunk.
            self.oars_has_df_chunk & ~single_period_oars
        ) | (  # OAR with or wo DataFrame chunk, single in period and within a single period.
            single_period_oars
            & (bincount(self.oars_period_idx.ravel())[self.oars_period_idx[:, 0]] == 2)
        )

    def partition_merge_regions(
        self,
        oar_idx_mrs_starts_ends_excl: NDArray,
    ) -> List[Tuple[int, NDArray]]:
        """
        Partition merge regions (MRs) into optimally sized chunks for writing.

        For each merge region (MR) defined in 'oar_idx_mrs_starts_ends_excl',
        this method:
        1. Determines split points where 'ordered_on' valuee is equal or larfer
           than corresponding time period lower bound and strictly lower than
           time period lower bound.
        2. Creates consolidated chunks by filtering the original OARs indices to
           ensure optimal row group loading.

        This ensures each consolidated chunk approaches row group time period
        while minimizing the number of row groups loaded into memory at once.

        Parameters
        ----------
        oar_idx_mrs_starts_ends_excl : NDArray
            Array of shape (e, 2) containing start and end indices (excluded)
            for each merge region to be consolidated.

        Returns
        -------
        List[Tuple[int, NDArray]]
            List of tuples, where each tuple contains for each merge sequence:
            - First element: Start index of the first row group in the merge
              sequence.
            - Second element: Array of shape (m, 2) containing end indices
              (excluded) for row groups and DataFrame chunks in the merge
              sequence.

        Notes
        -----
        The partitioning optimizes memory usage by loading only the minimum
        number of row groups needed to create complete chunks of approximately
        row group time period. The returned indices may be a subset of the
        original OARs indices, filtered to ensure efficient memory usage
        during the write process.

        """
        oars_merge_sequences = []
        for oar_idx_start, oar_idx_end_excl in oar_idx_mrs_starts_ends_excl:
            rg_idx_start = self.oars_rg_idx_starts[oar_idx_start]
            print()
            print("self.oars_period_idx[oar_idx_start:oar_idx_end_excl, 0]")
            print(self.oars_period_idx[oar_idx_start:oar_idx_end_excl, 0])
            # Find all OARs in this merge region that start a new period
            oar_idx_last_periods = (
                oar_idx_end_excl
                - oar_idx_start
                - 1
                - unique(
                    self.oars_period_idx[oar_idx_start:oar_idx_end_excl, 0][::-1],
                    return_index=True,
                )[1]
            )
            print()
            print("oar_idx_last_periods")
            print(oar_idx_last_periods)
            period_component_ends = self.oars_cmpt_idx_ends_excl[oar_idx_start:oar_idx_end_excl][
                oar_idx_last_periods
            ]
            print()
            print("period_component_ends")
            print(period_component_ends)
            oars_merge_sequences.append((rg_idx_start, period_component_ends))

        return oars_merge_sequences

    #        [
    #    (
    #        self.oars_rg_idx_starts[oar_idx_start],
    #        self.oars_cmpt_idx_ends_excl[oar_idx_start:oar_idx_end_excl][
    #            unique(self.oars_period_idx[oar_idx_start:oar_idx_end_excl, 0], return_index=True)[1]
    #        ]
    #    )
    #    for oar_idx_start, oar_idx_end_excl in oar_idx_mrs_starts_ends_excl
    # ]

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
