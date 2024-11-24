#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from dataclasses import dataclass
from typing import List, Optional, Union

from fastparquet import ParquetFile
from numpy import searchsorted
from numpy import vstack
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Timestamp
from pandas import date_range


MIN = "min"
MAX = "max"
LEFT = "left"
RIGHT = "right"
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.9


def _incomplete_row_groups_start(
    row_group_size_target: Union[int, str],
    df_max: Union[int, float, Timestamp],
    df_n_rows: int,
    max_n_irgs: int,
    n_rgs: int,
    rg_n_rows: List[int],
    rg_mins: NDArray,
    rg_maxs: NDArray,
) -> Optional[int]:
    """
    Walk backward through row groups to identify start of incomplete row groups.

    To evaluate number of 'incomplete' row groups, only those at the end
    of an existing dataset are accounted for. 'Incomplete' row groups in
    the middle of 'complete' row groups are not accounted for (they can
    be created by insertion of new data 'in the middle' of existing
    data).

    Parameters
    ----------
    row_group_size_target : Union[int, str]
        Target row group size.
    df_max : Union[int, float, Timestamp]
        Value of 'ordered_on' at end of DataFrame.
    df_n_rows : int
        Number of rows in DataFrame.
    max_n_irgs : int
        Max allowed number of 'incomplete' row groups.

        - ``None`` value induces no coalescing of row groups. If there is
            no drop of duplicates, new data is systematically appended.
        - A value of ``0`` or ``1`` means that new data should be
            systematically merged to the last existing one to 'complete' it
            (if it is not 'complete' already).

    n_rgs : int
        Total number of row groups in ParquetFile.
    rg_n_rows : List[int]
        Number of rows in each row group.
    rg_mins : ndarray
        Minimum value of 'ordered_on' in each row group.
    rg_maxs : ndarray
        Maximum value of 'ordered_on' in each row group.

    Returns
    -------
    Optional[int]
        Index of first incomplete row group, if any.
        Returns None if no coalescing is needed.

    Notes
    -----
    Coalescing is triggered when either:
    - The actual number of incomplete row groups exceeds 'max_n_irgs'.
    - The boundary of the last incomplete row group is exceeded by new data

    """
    df_connected_to_set_of_incomplete_rgs = False
    last_row_group_boundary_exceeded = False
    # Walking parquet file row groups backward to identify where the set of
    # incomplete row groups starts.
    irg_idx_start = n_rgs - 1
    if isinstance(row_group_size_target, int):
        # Case 'row_group_size_target' is an 'int'.
        # Number of incomplete row groups at end of recorded data.
        total_rows_in_irgs = 0
        min_row_group_size = int(row_group_size_target * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
        while rg_n_rows[irg_idx_start] <= min_row_group_size and irg_idx_start >= 0:
            total_rows_in_irgs += rg_n_rows[irg_idx_start]
            irg_idx_start -= 1
        if df_max > rg_maxs[irg_idx_start]:
            # If new data is located in the set of incomplete row groups,
            # add length of new data to the number of rows of incomplete row
            # groups.
            # In the 'if' above, 'irg_idx_start' is the index of the
            # last complete row groups.
            total_rows_in_irgs += df_n_rows
            df_connected_to_set_of_incomplete_rgs = True
        if total_rows_in_irgs >= row_group_size_target:
            # Necessary condition to coalesce is that total number of rows
            # in incomplete row groups is larger than target row group size.
            last_row_group_boundary_exceeded = True
    else:
        # Case 'row_group_size_target' is a str.
        # Get the 1st timestamp allowed in the last open period.
        # All row groups previous to this timestamp are considered complete.
        last_period_first_ts, next_period_first_ts = date_range(
            start=Timestamp(rg_mins[-1]).floor(row_group_size_target),
            freq=row_group_size_target,
            periods=2,
        )
        while rg_mins[irg_idx_start] >= last_period_first_ts and irg_idx_start >= 0:
            irg_idx_start -= 1
        if df_max >= next_period_first_ts and (n_rgs - irg_idx_start > 2):
            # Necessary conditions to coalesce are that last recorded row
            # group is incomplete (more than 1 row group) and the new data
            # exceeds first timestamp of next period.
            last_row_group_boundary_exceeded = True
        if df_max >= last_period_first_ts:
            df_connected_to_set_of_incomplete_rgs = True

    # Whatever the type of 'row_group_size_target', need to confirm or not
    # coalescing of incomplete row groups.
    # 'irg_idx_start' is '-1' with respect to its definition.
    irg_idx_start += 1
    n_irgs = n_rgs - irg_idx_start
    print("n_irgs")
    print(n_irgs)

    if df_connected_to_set_of_incomplete_rgs and (
        last_row_group_boundary_exceeded or n_irgs >= max_n_irgs
    ):
        # Coalesce recorded data only it new data overlaps with it,
        # or if new data is appended at the tail.
        return irg_idx_start


@dataclass
class OrderedMergeInfo:
    """
    Information about how DataFrame and ParquetFile overlap.

    Parameters
    ----------
    has_df_head : bool
        True if DataFrame head is large enough to make a new row group before
        ParquetFile overlap.
        If 'row_group_size_target' is a str, it is True if data in DataFrame
        starts before first period in ParquetFile.
        If 'row_group_size_target' is an int, it is True if there are more rows
        in DataFrame than 'row_group_size_target' before ParquetFile overlap.
    has_overlap : bool
        True if DataFrame and ParquetFile have overlapping data.
    has_df_tail : bool
        True if DataFrame has rows after ParquetFile overlap.
    df_idx_merge_start : Optional[int]
        Index of first row in DataFrame where merge starts, if any.
    df_idx_merge_end_excl : Optional[int]
        Index of the row after the last row in DataFrame where merge ends, if
        any.
    rg_idx_merge_start : Optional[int]
        Index in the complete ParquetFile of first row group where merge starts,
        if any.
    rg_idx_merge_end_excl : Optional[int]
        Index in the complete ParquetFile of row group after last row group
        where merge ends, if any.
    df_idx_tmrg_starts : ndarray
        Indices of the rows in DataFrame where each to-merge row group starts.
    df_idx_tmrg_ends_excl : ndarray
        Indices of the rows in DataFrame after the rows where each to-merge
        row group ends.
    tmrg_n_rows : ndarray
        Number of rows in each to-merge row group.
    n_tmrgs : int
        Number of to-merge row groups.
    sort_rgs_after_rewrite : bool
        Flag indicating if row groups have to be sorted after rewrite.

    """

    # Region flags
    has_df_head: bool  # Enough data before merge region to make a new row group
    has_overlap: bool  # Whether df and pf have overlapping data
    has_df_tail: bool  # Data after merge region
    # Merge region boundaries
    df_idx_merge_start: Optional[int]
    df_idx_merge_end_excl: Optional[int]
    rg_idx_merge_start: Optional[int]
    rg_idx_merge_end_excl: Optional[int]
    # To-merge row groups details (tmrg)
    df_idx_tmrg_starts: NDArray
    df_idx_tmrg_ends_excl: NDArray
    tmrg_n_rows: NDArray
    n_tmrgs: int
    # Flag indicating if row groups have to be sorted after rewrite.
    sort_rgs_after_rewrite: bool

    @classmethod
    def analyze(
        cls,
        df: DataFrame,
        pf: ParquetFile,
        ordered_on: str,
        row_group_size_target: Union[int, str],
        drop_duplicates: bool,
        max_n_irgs: Optional[int] = None,
    ) -> "OrderedMergeInfo":
        """
        Analyze how DataFrame and ParquetFile data overlap.

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
              - if True, at this index, overlap starts
              - if False, no overlap at this index
        max_n_irgs : Optional[int]
            Max allowed number of 'incomplete' row groups.

            - ``None`` value induces no coalescing of row groups. If there is
              no drop of duplicates, new data is systematically appended.
            - A value of ``0`` or ``1`` means that new data should be
              systematically merged to the last existing one to 'complete' it
              (if it is not 'complete' already).

        Returns
        -------
        OrderedMergeInfo
            Instance containing merge analysis information.

        Notes
        -----
        When a row in pf shares the same value in 'ordered_on' column as a row
        in df, the row in pf is considered as leading the row in df i.e.
        anterior to it.
        This has an impact in overlap identification in case of not dropping
        duplicates.

        When 'max_n_irgs' is specified, the method will analyze trailing
        incomplete row groups and may adjust the overlap region to include them
        for coalescing if necessary.

        """
        df_n_rows = len(df)
        rg_n_rows = [rg.num_rows for rg in pf.row_groups]
        n_rgs = len(rg_n_rows)
        df_min = df.loc[:, ordered_on].iloc[0]
        # Find overlapping regions in dataframe
        rg_mins = pf.statistics[MIN][ordered_on]
        rg_maxs = pf.statistics[MAX][ordered_on]
        if drop_duplicates:
            print("drop_duplicates")
            # Determine overlap start/end indices in row groups
            df_idx_tmrg_starts = searchsorted(df.loc[:, ordered_on], rg_mins, side=LEFT)
            df_idx_tmrg_ends_excl = searchsorted(df.loc[:, ordered_on], rg_maxs, side=RIGHT)
        else:
            print("no drop_duplicates")
            df_idx_tmrg_starts, df_idx_tmrg_ends_excl = searchsorted(
                df.loc[:, ordered_on],
                vstack((rg_mins, rg_maxs)),
                side=LEFT,
            )

        if df_idx_tmrg_ends_excl[-1] and df_idx_tmrg_starts[0] != df_n_rows:
            rg_idx_merge_start = df_idx_tmrg_ends_excl.astype(bool).argmax()
            rg_idx_merge_end_excl = df_idx_tmrg_ends_excl.argmax() + 1
        elif not df_idx_tmrg_ends_excl[-1]:
            # df after last row group in pf
            rg_idx_merge_start = rg_idx_merge_end_excl = n_rgs
        else:
            # df before first row group in pf
            rg_idx_merge_start = rg_idx_merge_end_excl = 0

        sort_rgs_after_rewrite = df_min <= rg_maxs[-1] if drop_duplicates else df_min < rg_maxs[-1]

        print("before trimming rgs lists")
        print("df_n_rows")
        print(df_n_rows)
        print("df_idx_tmrg_starts")
        print(df_idx_tmrg_starts)
        print("df_idx_tmrg_ends_excl")
        print(df_idx_tmrg_ends_excl)
        print("rg_idx_merge_start")
        print(rg_idx_merge_start)
        print("rg_idx_merge_end_excl")
        print(rg_idx_merge_end_excl)

        # Assess if trailing incomplete row groups in ParquetFile have to be
        # included in the merge.
        if max_n_irgs is not None:
            irg_idx_start = _incomplete_row_groups_start(
                row_group_size_target=row_group_size_target,
                df_max=df.loc[:, ordered_on].iloc[-1],
                df_n_rows=df_n_rows,
                max_n_irgs=max_n_irgs,
                n_rgs=n_rgs,
                rg_n_rows=rg_n_rows,
                rg_mins=rg_mins,
                rg_maxs=rg_maxs,
            )
            if irg_idx_start is not None:
                print("irg_idx_start")
                print(irg_idx_start)
                # If not None, coalescing of trailing row groups is needed.
                # Specific case when df is after last row group in pf, then
                # force integration of incomplete row groups in the merge.
                rg_idx_merge_start = (
                    min(rg_idx_merge_start, irg_idx_start)
                    if rg_idx_merge_start != n_rgs
                    else irg_idx_start
                )
                rg_idx_merge_end_excl = n_rgs
                sort_rgs_after_rewrite = True

        if rg_idx_merge_start != n_rgs:
            # In case DataFrame is leading ParquetFile, but not big enough to make
            # a new row group, force the merge to start at its first row.
            df_idx_merge_start = df_idx_tmrg_starts[rg_idx_merge_start]
            has_df_head = (
                df_idx_merge_start >= row_group_size_target
                if isinstance(row_group_size_target, int)
                else df_min < Timestamp(rg_mins[rg_idx_merge_start]).floor(row_group_size_target)
            )
            if not has_df_head and df_idx_merge_start:
                # If amount of rows at start of df is not enough to make a new
                # row group, but merge is not starting at the first row,
                # force the merge to at least encompass the first row group,
                rg_idx_merge_end_excl = max(1, rg_idx_merge_end_excl)
                # and force the merge to start at the first row in the DataFrame.
                df_idx_merge_start = df_idx_tmrg_starts[rg_idx_merge_start] = 0

            # Trim row group related lists to the overlapping region.
            df_idx_tmrg_starts = df_idx_tmrg_starts[rg_idx_merge_start:rg_idx_merge_end_excl]
            df_idx_tmrg_ends_excl = df_idx_tmrg_ends_excl[rg_idx_merge_start:rg_idx_merge_end_excl]
            tmrg_n_rows = rg_n_rows[rg_idx_merge_start:rg_idx_merge_end_excl]
        else:
            # df after last row group in pf.
            has_df_head = False
            df_idx_tmrg_starts = df_idx_tmrg_ends_excl = tmrg_n_rows = []

        # Analyze overlap patterns
        # Assume no overlap.
        print("after trimming rgs lists")
        print("df_idx_tmrg_starts")
        print(df_idx_tmrg_starts)
        print("df_idx_tmrg_ends_excl")
        print(df_idx_tmrg_ends_excl)
        # 'df_idx_tmrg_starts' is a numpy array. Need to check its length to check
        # if it is empty. bool(array([0])) is indeed False.
        if len(df_idx_tmrg_starts):
            # and (df_idx_overlap_start := df_idx_org_starts[0]) != df_n_rows
            # and (df_idx_overlap_end_excl := df_idx_org_ends_excl[-1]) != 0):
            # Overlap.
            # df_idx_merge_start = df_idx_tmrg_starts[0]
            # has_df_head = (
            #    df_idx_merge_start >= row_group_size_target
            #    if isinstance(row_group_size_target, int)
            #    else df_min < Timestamp(rg_mins[rg_idx_merge_start]).floor(row_group_size_target)
            # )
            # if not has_df_head and df_idx_merge_start:
            # If amount of rows at start of df is not enough to make a new
            # row group, but merge is not starting at the first row, force
            # the merge to start at the 1st row.
            #    df_idx_merge_start = df_idx_tmrg_starts[0] = 0

            has_overlap = True
            if not (df_idx_merge_end_excl := df_idx_tmrg_ends_excl[-1]):
                # Specific case when df is after last row group in pf.
                # Then force df_idx_merge_end_excl to include the first row in
                # df.
                df_idx_merge_end_excl = df_idx_tmrg_ends_excl[-1] = 1
            has_df_tail = df_idx_merge_end_excl < df_n_rows
            print("df_idx_merge_start")
            print(df_idx_merge_start)
            print("df_idx_merge_end_excl")
            print(df_idx_merge_end_excl)

        else:
            # No overlap.
            # Any value of df is ok to check its position with respect to pf.
            df_idx_merge_start = None
            df_idx_merge_end_excl = None
            has_overlap = False
            # has_df_head = df_min < rg_mins[0]
            has_df_tail = not has_df_head
            rg_idx_merge_start = None
            rg_idx_merge_end_excl = None

        return cls(
            has_df_head=has_df_head,
            has_overlap=has_overlap,
            has_df_tail=has_df_tail,
            df_idx_merge_start=df_idx_merge_start,
            df_idx_merge_end_excl=df_idx_merge_end_excl,
            rg_idx_merge_start=rg_idx_merge_start,
            rg_idx_merge_end_excl=rg_idx_merge_end_excl,
            df_idx_tmrg_starts=df_idx_tmrg_starts,
            df_idx_tmrg_ends_excl=df_idx_tmrg_ends_excl,
            tmrg_n_rows=tmrg_n_rows,
            n_tmrgs=len(tmrg_n_rows),
            sort_rgs_after_rewrite=sort_rgs_after_rewrite,
        )
