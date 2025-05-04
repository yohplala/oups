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
from typing import List, Optional, Tuple, Union

from numpy import arange
from numpy import array
from numpy import bool_
from numpy import column_stack
from numpy import cumsum
from numpy import diff
from numpy import empty
from numpy import flatnonzero
from numpy import insert
from numpy import int8
from numpy import int_
from numpy import isin
from numpy import ones
from numpy import r_
from numpy import searchsorted
from numpy import vstack
from numpy import zeros
from numpy.typing import NDArray
from pandas import Series


LEFT = "left"
RIGHT = "right"
FILTERED_MERGE_SEQUENCES = "filtered_merge_sequences"


def get_region_indices_of_true_values(mask: NDArray[bool_]) -> NDArray[int_]:
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


def set_true_in_regions(length: int, regions: NDArray[int_]) -> NDArray[bool_]:
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
    return cumsum(changes[:-1]).astype(bool_)


def get_region_start_end_delta(m_values: NDArray, indices: NDArray) -> NDArray:
    """
    Get difference between values at end and start of each region.

    For regions where the start index is 0, the start value is considered 0.
    For all other regions, the start value is m_values[start_index - 1].

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
        start of each region, with special handling for regions starting at
        index 0.

    """
    if not indices.size:
        return empty(0, dtype=int_)
    if indices[0, 0] == 0:
        start_values = m_values[indices[:, 0] - 1]
        start_values[0] = 0
        return m_values[indices[:, 1] - 1] - start_values
    else:
        return m_values[indices[:, 1] - 1] - m_values[indices[:, 0] - 1]


class OARMergeSplitStrategy(ABC):
    """
    Abstract base class for ordered atomic region merge and split strategies.

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
    oars_has_df_overlap : NDArray[bool_]
        Boolean array indicating if OAR contains a DataFrame chunk.
    n_oars : int
        Number of ordered atomic regions.
    oars_likely_on_target_size : NDArray[bool_]
        Boolean array indicating if OAR is likely to be on target size.
    oar_idx_mrs_starts_ends_excl : NDArray[int_]
        Array of shape (e, 2) containing the list of the OARs start and end
        indices for each merge regions.
    rg_idx_ends_excl_not_to_use_as_split_points : Union[NDArray, None]
        Array containing indices of row groups which should not be used as split
        points in 'filtered_merge_sequences'. This ensures these row groups will
        be loaded all together so that duplicate search can be made over all
        relevant row groups.
    n_rgs : int
        Number of existing row groups.
    n_df_rows : int
        Number of rows in DataFrame.
    rg_idx_mrs_starts_ends_excl : List[slice]
        List of slices, each containing the start (included) and end (excluded)
        indices of the row groups in a merge sequence.
    filtered_merge_sequences : List[Tuple[int, NDArray]]
        List of merge sequences, each containing a tuple of two items:
        - the first item is the row group index starting the merge sequence,
        - the second item is a numpy array of shape (n, 2) containing the
          successive end (excluded) indices of row groups and DataFrame row of
          the merge sequence.

    """

    def __init__(
        self,
        rg_ordered_on_mins: NDArray,
        rg_ordered_on_maxs: NDArray,
        df_ordered_on: Series,
        drop_duplicates: Optional[bool] = False,
    ):
        """
        Compute ordered atomic regions (OARs) from row groups and DataFrame.

        An ordered atomic region is either:
        - A row group and its overlapping DataFrame chunk (if any)
        - A DataFrame chunk that doesn't overlap with any row group

        Returned arrays provide the start and end (excluded) indices in row
        groups and end (excluded) indices in DataFrame for each of these ordered
        atomic regions. All these arrays are of same size and describe how are
        composed the ordered atomic regions.

        Parameters
        ----------
        rg_ordered_on_mins : NDArray[Timestamp]
            Minimum values of 'ordered_on' in each row group.
        rg_ordered_on_maxs : NDArray[Timestamp]
            Maximum values of 'ordered_on' in each row group.
        df_ordered_on : Series[Timestamp]
            Values of 'ordered_on' column in DataFrame.
        drop_duplicates : Optional[bool], default False
            Flag impacting how overlapping boundaries have to be managed.
            More exactly, row groups are considered as first data, and DataFrame
            as second data, coming after. In case of a row group leading a
            DataFrame chunk, if the last value in row group is a duplicate of a
            value in DataFrame chunk, then
            - If True, at this index, overlap starts
            - If False, no overlap at this index

        Attributes
        ----------
        rg_idx_ends_excl_not_to_use_as_split_points : Union[NDArray, None]
            Array containing indices of row groups which should not be used as
            split points in 'filtered_merge_sequences'. This ensures these row
            groups will be loaded all together so that duplicate search can be
            made over all relevant row groups.

        Raises
        ------
        ValueError
            If input arrays have inconsistent lengths or unsorted data.

        Notes
        -----
        Start indices in DataFrame are not provided, as they can be inferred
        from the end (excluded) indices in DataFrame of the previous ordered
        atomic region (no part of the DataFrame is omitted for the write).

        In case 'drop_duplicates' is False, and there are duplicate values
        between row group max values and DataFrame 'ordered_on' values, then
        DataFrame 'ordered_on' values are considered to be the last occurrences
        of the duplicates in 'ordered_on'. Leading row groups (with duplicate
        max values) will not be in the same ordered atomic region as the
        DataFrame chunk starting at the duplicate 'ordered_on' value. This is
        an optimization to prevent rewriting these leading row groups.

        On the opposite, if 'drop_duplicates' is True, then the row group with
        the last occurrence of a duplicate 'ordered_on' value will be considered
        to be the one corresponding to the DataFrame chunk with this value.
        But all row groups with a this duplicate 'ordered_on' value will be
        considered to have an overlap with the DataFrame chunk with this value,
        i.e. 'self.oars_has_df_overlap' will be True for these row groups.

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
        # Check use of OARSplitStrategy with no row groups.
        if n_df_rows and not n_rgs:
            self.oars_rg_idx_starts = zeros(1, dtype=int_)
            self.oars_cmpt_idx_ends_excl = array([[0, n_df_rows]], dtype=int_)
            self.oars_has_row_group = zeros(1).astype(bool_)
            self.oars_df_n_rows = array([n_df_rows], dtype=int_)
            self.oars_has_df_overlap = ones(1).astype(bool_)
            self.n_oars = 1
            self.rg_idx_ends_excl_not_to_use_as_split_points = None
            self.n_rgs = 0
            self.n_df_rows = n_df_rows
            return

        if drop_duplicates:
            # Determine overlap start/end indices in row groups
            df_idx_rgs_starts = searchsorted(df_ordered_on, rg_ordered_on_mins, side=LEFT)
            df_idx_rgs_ends_excl = searchsorted(df_ordered_on, rg_ordered_on_maxs, side=RIGHT)
        else:
            df_idx_rgs_starts, df_idx_rgs_ends_excl = searchsorted(
                df_ordered_on,
                vstack((rg_ordered_on_mins, rg_ordered_on_maxs)),
                side=LEFT,
            )
        # Keep track of which row groups have an overlap with a DataFrame chunk.
        rgs_has_df_overlap = df_idx_rgs_starts != df_idx_rgs_ends_excl
        # 'rg_idx_ends_excl_not_to_use_as_split_points' keeps track of row group
        # indices which should not be used as split points.
        self.rg_idx_ends_excl_not_to_use_as_split_points = None
        if any(
            rgs_min_equ_max := (rg_ordered_on_mins[1:] == rg_ordered_on_maxs[:-1])
            & rgs_has_df_overlap[1:],
        ):
            # In case rg_maxs[i] is a duplicate of rg_mins[i+1],
            # then df_idx_rg_ends_excl for rg[i] should be set to
            # df_idx_rg_starts of rg[i+1], so that the overlapping df chunk is
            # not in several row groups.
            # Restrict the correction to row groups that overlap with a
            # DataFrame chunk.
            rg_idx_maxs_to_correct = flatnonzero(rgs_min_equ_max)
            df_idx_rgs_ends_excl[rg_idx_maxs_to_correct] = df_idx_rgs_starts[
                rg_idx_maxs_to_correct + 1
            ]
            self.rg_idx_ends_excl_not_to_use_as_split_points = rg_idx_maxs_to_correct + 1
        # DataFrame orphans are regions in DataFrame that do not overlap with
        # any row group. Find indices in row groups of DataFrame orphans.
        rg_idx_df_orphans = flatnonzero(
            r_[
                df_idx_rgs_starts[0],  # gap at start (0 to first start)
                df_idx_rgs_ends_excl[:-1] - df_idx_rgs_starts[1:],
                n_df_rows - df_idx_rgs_ends_excl[-1],  # gap at end
            ],
        )
        n_df_orphans = len(rg_idx_df_orphans)
        rg_idxs_template = arange(n_rgs + 1)
        if n_df_orphans != 0:
            # Case of non-overlapping regions in DataFrame.
            # Resize 'rg_idxs', and duplicate values where there are
            # non-overlapping regions in DataFrame.
            # These really become now the OARs.
            rg_idx_to_insert = rg_idxs_template[rg_idx_df_orphans]
            rg_idxs_template = insert(
                rg_idxs_template,
                rg_idx_df_orphans,
                rg_idx_to_insert,
            )
            # 'Resize 'df_idx_oar_ends_excl', and duplicate values where there
            # are non-overlapping regions in DataFrame.
            df_idx_to_insert = r_[df_idx_rgs_starts, n_df_rows][rg_idx_df_orphans]
            df_idx_rgs_ends_excl = insert(
                df_idx_rgs_ends_excl,
                rg_idx_df_orphans,
                df_idx_to_insert,
            )
            rgs_has_df_overlap = insert(
                rgs_has_df_overlap,
                rg_idx_df_orphans,
                True,
            )

        self.oars_rg_idx_starts = (
            rg_idxs_template[:-1] if len(rg_idxs_template) > 1 else zeros(1, dtype=int_)
        )
        self.oars_cmpt_idx_ends_excl = column_stack((rg_idxs_template[1:], df_idx_rgs_ends_excl))
        self.oars_has_row_group = rg_idxs_template[:-1] != rg_idxs_template[1:]
        self.oars_df_n_rows = diff(df_idx_rgs_ends_excl, prepend=0)
        self.oars_has_df_overlap = rgs_has_df_overlap
        self.n_oars = len(self.oars_rg_idx_starts)
        self.n_rgs = n_rgs
        self.n_df_rows = n_df_rows

    @abstractmethod
    def _specialized_init(self, **kwargs):
        """
        Initialize specialized attributes.

        This method initializes attributes specific to strategy concrete
        implementation. It is kept apart from class constructor to allow for
        reuse by 'self.from_oars_desc' class method.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to initialize specialized attributes.

        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def from_oars_desc(
        cls,
        oars_rg_idx_starts: NDArray,
        oars_cmpt_idx_ends_excl: NDArray,
        oars_has_row_group: NDArray,
        oars_has_df_overlap: NDArray,
        rg_idx_ends_excl_not_to_use_as_split_points: Union[NDArray, None],
        **kwargs,
    ) -> "OARMergeSplitStrategy":
        """
        Create a strategy instance with a given OARs description.

        This is primarily for testing purposes, allowing tests to directly set
        the 'OARMergeSplitStrategy' base attributes without having to compute it
        from row groups and DataFrame.

        Parameters
        ----------
        oars_rg_idx_starts : NDArray
            Start indices of row groups in each OAR.
        oars_cmpt_idx_ends_excl : NDArray
            End indices (excluded) of row groups and DataFrame chunks in each
            OAR.
        oars_has_row_group : NDArray
            Boolean array indicating if OAR contains a row group.
        oars_has_df_overlap : NDArray
            Boolean array indicating if OAR overlaps with a DataFrame chunk.
        rg_idx_ends_excl_not_to_use_as_split_points : Union[NDArray, None]
            Array of indices for row group not to use as split points. There are
            filtered out from 'filtered_merge_sequences'.
        **kwargs
            Additional arguments needed by specific strategy implementations.
            For NRowsMergeSplitStrategy, this should include 'rgs_n_rows',
            'row_group_target_size', and optionally 'drop_duplicates'.
            For TimePeriodMergeSplitStrategy, this should include
            'rg_ordered_on_mins', 'rg_ordered_on_maxs', 'df_ordered_on', and
            'row_group_time_period'.

        Returns
        -------
        OARMergeSplitStrategy
            An instance of the strategy with the given OARs description.

        """
        instance = cls.__new__(cls)
        instance.oars_rg_idx_starts = oars_rg_idx_starts
        instance.oars_cmpt_idx_ends_excl = oars_cmpt_idx_ends_excl
        instance.oars_has_row_group = oars_has_row_group
        instance.oars_df_n_rows = diff(oars_cmpt_idx_ends_excl[:, 1], prepend=0)
        instance.oars_has_df_overlap = oars_has_df_overlap
        instance.rg_idx_ends_excl_not_to_use_as_split_points = (
            rg_idx_ends_excl_not_to_use_as_split_points
        )
        instance.n_oars = len(oars_rg_idx_starts)
        instance._specialized_init(**kwargs)
        return instance

    @abstractmethod
    def oars_likely_on_target_size(self) -> NDArray:
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
        raise NotImplementedError("Subclasses must implement this method")

    def _compute_merge_regions_start_ends_excl(
        self,
        max_n_off_target_rgs: Optional[int] = None,
    ) -> NDArray[int_]:
        """
        Aggregate ordered atomic regions into merge regions.

        Sets of contiguous ordered atomic regions with DataFrame chunks are
        possibly extended with neighbor regions that are off target size
        depending on two conditions,
        - if the atomic merge region with DataFrame chunks is found to result in
          a row group potentially on target size.
        - if the total number of atomic merge regions off target size in a
          given enlarged merge region is greater than `max_n_off_target_rgs`.

        Parameters
        ----------
        max_n_off_target_rgs : Optional[int_], default None
            Maximum number of off-target size row groups allowed in a contiguous
            set of row groups. It cannot be set to 0. This parameter helps
            limiting fragmentation by limiting number of contiguous row groups
            off target size.
            A ``None`` value induces no merging of off target size row groups
            neighbor to a newly added row groups.

        Attributes
        ----------
        oar_idx_mrs_starts_ends_excl : NDArray[int_]
            A numpy array of shape (e, 2) containing the list of the OARs start
            and end indices for each merge regions.

        Notes
        -----
        Reason for including off target size OARs contiguous to a newly added
        OAR likely to be on target size is to prevent that the addition of new
        data creates isolated sets of off target size row groups followed by on
        target size row groups. This most notably applies when new data is
        appended at the tail of the DataFrame.

        This method relies on the abstract property
        'oars_likely_on_target_size' which must be implemented by concrete
        subclasses before calling this method.

        """
        simple_mrs_starts_ends_excl = get_region_indices_of_true_values(self.oars_has_df_overlap)
        if max_n_off_target_rgs is None:
            self.oar_idx_mrs_starts_ends_excl = simple_mrs_starts_ends_excl
            return
        elif max_n_off_target_rgs == 0:
            raise ValueError("'max_n_off_target_rgs' cannot be 0.")

        # If 'max_n_off_target_rgs' is not None, then we need to compute the
        # merge regions.
        # Step 1: assess start indices (included) and end indices (excluded) of
        # enlarged merge regions.
        oars_off_target = ~self.oars_likely_on_target_size
        potential_emrs_starts_ends_excl = get_region_indices_of_true_values(
            self.oars_has_df_overlap | oars_off_target,
        )
        if self.n_df_rows:
            # Filter out emrs without overlap with a DataFrame chunk.
            # If there is no DataFrame overlap, then all enlarged merge regions
            # are accepted. This allows for resize of row groups if desired.
            # As of this point, potential enlarged merge regions are those which
            # have an overlap with a simple merge region (has a DataFrame
            # chunk).
            potential_emrs_starts_ends_excl = potential_emrs_starts_ends_excl[
                get_region_start_end_delta(
                    m_values=cumsum(self.oars_has_df_overlap),
                    indices=potential_emrs_starts_ends_excl,
                ).astype(bool_)
            ]
        # Step 2: Filter out enlarged candidates based on multiple criteria.
        # 2.a - Get number of off target size OARs per enlarged merged region.
        # Those where 'max_n_off_target_rgs' is not reached will be filtered out
        n_off_target_oars_in_pemrs = get_region_start_end_delta(
            m_values=cumsum(oars_off_target),
            indices=potential_emrs_starts_ends_excl,
        )
        # 2.b Get which enlarged regions into which the merge will likely create
        # on target row groups (not combining OARs together).
        # TODO: test if this condition is really needed by commenting it out
        # and checking if test results are changed?
        # 'mrs_likely_exceeds_target_size' is probably more restrictive than
        # this condition, by assessing this on merge regions rather than
        # individual OARs.
        creates_on_target_rg_in_pemrs = get_region_start_end_delta(
            m_values=cumsum(self.oars_likely_on_target_size),
            indices=potential_emrs_starts_ends_excl,
        ).astype(bool_)
        # Keep enlarged merge regions with too many off target atomic regions or
        # with likely creation of on target row groups.
        confirmed_emrs_starts_ends_excl = potential_emrs_starts_ends_excl[
            (n_off_target_oars_in_pemrs > max_n_off_target_rgs)
            | creates_on_target_rg_in_pemrs
            | self.mrs_likely_exceeds_target_size(
                mrs_starts_ends_excl=potential_emrs_starts_ends_excl,
            )
        ]
        if not self.n_df_rows:
            # If there is no DataFrame overlap, no need for subsequent steps.
            self.oar_idx_mrs_starts_ends_excl = confirmed_emrs_starts_ends_excl
            return

        # Step 3: Retrieve indices of merge regions which have overlap with a
        # DataFrame chunk but are not in retained enlarged merge regions.
        oars_confirmed_emrs = set_true_in_regions(
            length=self.n_oars,
            regions=confirmed_emrs_starts_ends_excl,
        )
        # Create an array of length the number of simple merge regions, with
        # value 1 if the simple merge region is within an enlarged merge
        # regions.
        smrs_overlaps_with_confirmed_emrs = get_region_start_end_delta(
            m_values=cumsum(oars_confirmed_emrs),
            indices=simple_mrs_starts_ends_excl,
        ).astype(bool_)
        n_simple_mrs_in_enlarged_mrs = sum(smrs_overlaps_with_confirmed_emrs)
        if n_simple_mrs_in_enlarged_mrs == 0:
            # Case there is no simple merge regions in enlarged merge regions.
            # This means there is no enlarged merge regions.
            self.oar_idx_mrs_starts_ends_excl = simple_mrs_starts_ends_excl
        elif n_simple_mrs_in_enlarged_mrs == len(simple_mrs_starts_ends_excl):
            # Case all simple merge regions are encompassed in enlarged merge
            # regions.
            self.oar_idx_mrs_starts_ends_excl = confirmed_emrs_starts_ends_excl
        else:
            # Case in-between.
            self.oar_idx_mrs_starts_ends_excl = vstack(
                (
                    simple_mrs_starts_ends_excl[~smrs_overlaps_with_confirmed_emrs],
                    confirmed_emrs_starts_ends_excl,
                ),
            )
            # Sort along 1st column.
            # Sorting is required to ensure that DataFrame chunks are enumerated
            # correctly (end indices excluded of a Dataframe chunk is the start
            # index of the next Dataframe chunk).
            self.oar_idx_mrs_starts_ends_excl = self.oar_idx_mrs_starts_ends_excl[
                self.oar_idx_mrs_starts_ends_excl[:, 0].argsort()
            ]

    @abstractmethod
    def _specialized_compute_merge_sequences(
        self,
    ) -> List[Tuple[int, NDArray]]:
        """
        Sequence merge regions (MRs) into optimally sized chunks for writing.

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

        """
        raise NotImplementedError("Subclasses must implement this method")

    def compute_merge_sequences(
        self,
        max_n_off_target_rgs: Optional[int] = None,
    ) -> List[Tuple[int, NDArray]]:
        """
        Compute merge sequences.

        This method is a wrapper to the chain of methods:
        - '_compute_merge_regions_start_ends_excl'
        - '_specialized_compute_merge_sequences'
        Additionally, row group indices listed in
        'rg_idx_ends_excl_not_to_use_as_split_points' are filtered out from the
        output returned by child '_specialized_compute_merge_sequences'. This
        filtering ensures that in case 'drop_duplicates' is True, prior
        existing row groups with a max 'ordered_on' value equals to next row
        group's min 'ordered_on' value are merged. This approach guarantees that
        duplicate search is made over all relevant priorly existing row groups.

        Parameters
        ----------
        max_n_off_target_rgs : Optional[int], default None
            Maximum number of off target row groups to merge.

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
        The return value is also stored in 'self.filtered_merge_sequences'.

        """
        self._compute_merge_regions_start_ends_excl(max_n_off_target_rgs=max_n_off_target_rgs)
        self.filtered_merge_sequences = (
            self._specialized_compute_merge_sequences()
            if self.rg_idx_ends_excl_not_to_use_as_split_points is None
            else [
                (
                    rg_idx_start,
                    cmpt_ends_excl[
                        isin(
                            cmpt_ends_excl[:, 0],
                            self.rg_idx_ends_excl_not_to_use_as_split_points,
                            invert=True,
                        )
                    ],
                )
                for rg_idx_start, cmpt_ends_excl in self._specialized_compute_merge_sequences()
            ]
        )
        return self.filtered_merge_sequences

    @cached_property
    def rg_idx_mrs_starts_ends_excl(self) -> List[slice]:
        """
        Get the start and end indices of row groups for each merge regions.

        Returns
        -------
        List[slice]
            List of slices, where each slice contains the start (included) and
            end (excluded) indices of row groups for each merge region.

        """
        if not hasattr(self, "oar_idx_mrs_starts_ends_excl"):
            raise AttributeError(
                "not possible to return 'rg_idx_mrs_starts_ends_excl' value if "
                "'compute_merge_sequences()' has not been run beforehand.",
            )
        return [
            slice(rg_idx_start, rg_idx_end_excl)
            for oar_idx_start, oar_idx_end_excl in self.oar_idx_mrs_starts_ends_excl
            if (
                (rg_idx_start := self.oars_rg_idx_starts[oar_idx_start])
                != (rg_idx_end_excl := self.oars_cmpt_idx_ends_excl[oar_idx_end_excl - 1, 0])
            )
        ]

    @cached_property
    def sort_rgs_after_write(self) -> bool:
        """
        Whether to sort row groups after writing.

        Row groups witten may be so in the middle of existing row groups.
        It is then required to sort them so that order is maintained between
        row groups.

        Returns
        -------
        bool
            Whether to sort row groups after writing.

        """
        try:
            return (
                (
                    len(self.filtered_merge_sequences) > 1
                    # 'filtered_merge_sequences[0][1][-1,0]' is 'rg_idx_ends_excl'
                    # of the last row group in the first merge sequence.
                    or self.filtered_merge_sequences[0][1][-1, 0] < self.n_rgs
                )
                if self.filtered_merge_sequences
                else False
            )
        except AttributeError:
            raise AttributeError(
                "not possible to return 'sort_rgs_after_write' value if "
                "'compute_merge_sequences()' has not been run beforehand.",
            )

    @abstractmethod
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
        raise NotImplementedError("Subclasses must implement this method")
