#!/usr/bin/env python3
"""
Created on Mon Mar 17 18:00:00 2025.

Concrete implementation of OARMergeSplitStrategy based on number of rows.

@author: yoh

"""
from functools import cached_property
from typing import List, Optional, Tuple

from numpy import cumsum
from numpy import int_
from numpy import linspace
from numpy import maximum
from numpy import r_
from numpy import searchsorted
from numpy import unique
from numpy import zeros
from numpy.typing import NDArray
from pandas import Series

from oups.store.write.merge_split_strategies.base import OARMergeSplitStrategy
from oups.store.write.merge_split_strategies.base import get_region_start_end_delta


LEFT = "left"
ROW_GROUP_TARGET_SIZE_SCALE_FACTOR = 0.8  # % of target row group size.
# MIN_RG_NUMBER_TO_ENSURE_ON_TARGET_RGS = 1 / (1 - ROW_GROUP_TARGET_SIZE_SCALE_FACTOR)


class NRowsMergeSplitStrategy(OARMergeSplitStrategy):
    """
    OAR merge and split strategy based on a target number of rows per row group.

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
        ``ROW_GROUP_TARGET_SIZE_SCALE_FACTOR * row_group_target_size``.
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
        rgs_n_rows: NDArray,
        row_group_target_size: int,
        drop_duplicates: Optional[bool] = False,
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
        rgs_n_rows : NDArray
            Array of shape (r) containing the number of rows in each row group
            in existing ParquetFile.
        row_group_target_size : int
            Target number of rows above which a new row group should be created.
        drop_duplicates : Optional[bool], default False
            Whether to drop duplicates between row groups and DataFrame.

        """
        super().__init__(
            rg_ordered_on_mins,
            rg_ordered_on_maxs,
            df_ordered_on,
            drop_duplicates,
        )
        self._specialized_init(
            rgs_n_rows=rgs_n_rows,
            row_group_target_size=row_group_target_size,
            drop_duplicates=drop_duplicates,
        )

    def _specialized_init(
        self,
        rgs_n_rows: NDArray,
        row_group_target_size: int,
        drop_duplicates: Optional[bool] = False,
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
        drop_duplicates : Optional[bool], default False
            Whether to drop duplicates between row groups and DataFrame.

        """
        self.row_group_target_size = row_group_target_size
        self.row_group_min_size = int(row_group_target_size * ROW_GROUP_TARGET_SIZE_SCALE_FACTOR)
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
    def oars_likely_on_target_size(self) -> NDArray:
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
        See the parent class documentation for details on the asymmetric
        treatment of OARs with and without DataFrame chunks.

        """
        return self.oars_has_df_overlap & (  # OAR containing a DataFrame chunk.
            self.oars_max_n_rows >= self.row_group_min_size
        ) | ~self.oars_has_df_overlap & (  # OAR containing only row groups.
            (self.oars_max_n_rows >= self.row_group_min_size)
            & (self.oars_max_n_rows <= self.row_group_target_size)
        )

    def mrs_likely_exceeds_target_size(self, mrs_starts_ends_excl: NDArray) -> NDArray:
        """
        Return boolean array indicating which merge regions likely exceed target size.

        Parameters
        ----------
        mrs_starts_ends_excl : NDArray
            Array of shape (m, 2) containing the start (included) and end
            (excluded) indices of the merge regions.

        Returns
        -------
        NDArray
            Boolean array of length equal to the number of merge regions, where
            True indicates the merge region is likely to exceed target size.

        """
        return (
            get_region_start_end_delta(
                m_values=cumsum(self.oars_min_n_rows),
                indices=mrs_starts_ends_excl,
            )
            >= self.row_group_target_size
        )

    def _specialized_compute_merge_sequences(
        self,
    ) -> List[Tuple[int, NDArray]]:
        """
        Sequence merge regions (MRs) into optimally sized chunks for writing.

        For each merge region (MR) defined in 'oar_idx_mrs_starts_ends_excl',
        this method:
        1. Accumulates row counts using self.oars_min_n_rows
        2. Determines split points where accumulated rows reach
           'self.row_group_target_size'
        3. Creates consolidated chunks by filtering the original OARs indices to
           ensure optimal row group loading.

        This ensures each consolidated chunk approaches the target row size
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
        target size rows. The returned indices may be a subset of the original
        OARs indices, filtered to ensure efficient memory usage during the
        write process.

        """
        # Process each merge region to find optimal split points:
        # 1. For each merge region, accumulate row counts
        # 2. Find indices where accumulated rows reach multiples of target size
        # 3. Include the last index of the region
        # 4. Return a list of tuples with:
        #    - Starting row group index for each merge sequence
        #    - Array of component end indices at split points
        return [
            (
                self.oars_rg_idx_starts[oar_idx_start],
                self.oars_cmpt_idx_ends_excl[oar_idx_start:oar_idx_end_excl][
                    r_[
                        unique(
                            searchsorted(
                                (
                                    cum_rows := cumsum(
                                        self.oars_min_n_rows[oar_idx_start:oar_idx_end_excl],
                                    )
                                ),
                                linspace(
                                    self.row_group_target_size,
                                    self.row_group_target_size
                                    * (n_multiples := cum_rows[-1] // self.row_group_target_size),
                                    n_multiples,
                                    endpoint=True,
                                    dtype=int_,
                                ),
                                side=LEFT,
                            ),
                        )[:-1],
                        oar_idx_end_excl - oar_idx_start - 1,
                    ]
                ],
            )
            for oar_idx_start, oar_idx_end_excl in self.oar_idx_mrs_starts_ends_excl
        ]

    def compute_split_sequence(self, df_ordered_on: Series) -> List[int]:
        """
        Define the split sequence for a chunk depending row group target size.

        Result is to be used as `row_group_offsets` parameter in
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
        return list(range(0, len(df_ordered_on), self.row_group_target_size))
