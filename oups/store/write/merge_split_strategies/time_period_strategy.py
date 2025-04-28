#!/usr/bin/env python3
"""
Created on Mon Mar 17 18:00:00 2025.

Concrete implementation of OARMergeSplitStrategy based on time period.

@author: yoh

"""
from functools import cached_property
from typing import List, Optional, Tuple

from numpy import bincount
from numpy import dtype
from numpy import flatnonzero
from numpy import ones
from numpy import searchsorted
from numpy import unique
from numpy import zeros
from numpy.typing import NDArray
from pandas import Series
from pandas import Timestamp
from pandas import date_range

from oups.store.utils import ceil_ts
from oups.store.utils import floor_ts
from oups.store.write.merge_split_strategies.base import OARMergeSplitStrategy


LEFT = "left"
RIGHT = "right"
DTYPE_DATETIME64 = dtype("datetime64[ns]")


class TimePeriodMergeSplitStrategy(OARMergeSplitStrategy):
    """
    OAR merge and split strategy based on a time period target per row group.

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
        row_group_time_period: str,
        drop_duplicates: Optional[bool] = False,
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
        drop_duplicates : Optional[bool], default False
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
        self._specialized_init(
            rg_ordered_on_mins,
            rg_ordered_on_maxs,
            df_ordered_on,
            row_group_time_period,
        )

    def _specialized_init(
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
        if not df_ordered_on.empty and df_ordered_on.dtype != DTYPE_DATETIME64:
            raise TypeError(
                "if 'row_group_target_size' is a pandas 'freqstr', dtype"
                f" of column {df_ordered_on.name} has to be 'datetime64[ns]'.",
            )
        self.row_group_time_period = row_group_time_period
        df_ordered_on_np = df_ordered_on.to_numpy()
        self.oars_mins_maxs = ones((self.n_oars, 2)).astype(df_ordered_on_np.dtype)
        # Row groups encompasses Dataframe chunks in an OAR.
        # Hence, start with Dataframe chunks starts and ends.
        oar_idx_df_chunk = flatnonzero(self.oars_has_df_overlap)
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
    def oars_likely_on_target_size(self) -> NDArray:
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
        See the parent class documentation for details on the asymmetric
        treatment of OARs with and without DataFrame chunks.

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
            self.oars_has_df_overlap & ~single_period_oars
        ) | (  # OAR with or wo DataFrame chunk, single in period and within a single period.
            single_period_oars
            & (bincount(self.oars_period_idx.ravel())[self.oars_period_idx[:, 0]] == 2)
        )

    def _specialized_compute_merge_sequences(
        self,
    ) -> List[Tuple[int, NDArray]]:
        """
        Sequence merge regions (MRs) into optimally sized chunks for writing.

        For each merge region (MR) defined in 'oar_idx_mrs_starts_ends_excl',
        this method:
        1. Determines split points where 'ordered_on' valuee is equal or larfer
           than corresponding time period lower bound and strictly lower than
           time period lower bound.
        2. Creates consolidated chunks by filtering the original OARs indices to
           ensure optimal row group loading.

        This ensures each consolidated chunk approaches row group time period
        while minimizing the number of row groups loaded into memory at once.

        Returns
        -------
        List[Tuple[int, NDArray]]
            Merge sequences, a list of tuples, where each tuple contains for
            each merge sequence:
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
        # Process each merge region to find period-based split points:
        # 1. For each merge region, identify the starting row group index
        # 2. Find indices of OARs that are the last in each unique time period
        # 3. Extract component end indices at these period boundaries
        # 4. Return a list of tuples with:
        #    - Starting row group index for each merge sequence
        #    - Array of component end indices at period boundaries
        return [
            (
                self.oars_rg_idx_starts[oar_idx_start],
                self.oars_cmpt_idx_ends_excl[oar_idx_start:oar_idx_end_excl][
                    (
                        oar_idx_end_excl
                        - oar_idx_start
                        - 1
                        - unique(
                            self.oars_period_idx[oar_idx_start:oar_idx_end_excl, 0][::-1],
                            return_index=True,
                        )
                    )[1]
                ],
            )
            for oar_idx_start, oar_idx_end_excl in self.oar_idx_mrs_starts_ends_excl
        ]

    def compute_split_sequence(self, df_ordered_on: Series) -> List[int]:
        """
        Define the split sequence for a chunk depending row group target size.

        Result is to be used as `compute_split_sequence` parameter in
        `iter_dataframe` method.

        Parameters
        ----------
        df_ordered_on : Series
            Series by which the DataFrame to be written is ordered.

        Returns
        -------
        List[int]
            A list of indices with the explicit index values to start new row
            groups.

        """
        # Generate period bounds for the chunk.
        # start_ts = floor_ts(Timestamp(df_ordered_on.iloc[0]), self.row_group_time_period)
        # end_ts = ceil_ts(Timestamp(df_ordered_on.iloc[-1]), self.row_group_time_period)
        # period_bounds = date_range(start=start_ts, end=end_ts, freq=self.row_group_time_period)[:-1]
        # Find where each period boundary falls in 'df_ordered_on'.
        return unique(
            searchsorted(
                df_ordered_on,
                date_range(
                    start=floor_ts(Timestamp(df_ordered_on.iloc[0]), self.row_group_time_period),
                    end=ceil_ts(Timestamp(df_ordered_on.iloc[-1]), self.row_group_time_period),
                    freq=self.row_group_time_period,
                )[:-1],
                side=LEFT,
            ),
        ).tolist()
