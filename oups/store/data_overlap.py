#!/usr/bin/env python3
"""
Created on Thu Nov 14 18:00:00 2024.

@author: yoh

"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from fastparquet import ParquetFile
from pandas import DataFrame


MIN = "min"
MAX = "max"
LEFT = "left"
RIGHT = "right"


@dataclass
class DataOverlapInfo:
    """
    Information about how DataFrame and ParquetFile overlap.

    Parameters
    ----------
    has_pf_head : bool
        True if ParquetFile has row groups before DataFrame overlap.
    has_df_head : bool
        True if DataFrame has sufficient rows before ParquetFile overlap.
    has_overlap : bool
        True if DataFrame and ParquetFile have overlapping data.
    has_pf_tail : bool
        True if ParquetFile has row groups after DataFrame overlap.
    has_df_tail : bool
        True if DataFrame has rows after ParquetFile overlap.
    df_idx_overlap_start : Optional[int]
        Index of first overlapping row in DataFrame, if any.
    df_idx_overlap_end_excl : Optional[int]
        Index of the row after the last overlapping row in DataFrame, if any.
    rg_idx_overlap_start : Optional[int]
        Index of first overlapping row group, if any.
    rg_idx_overlap_end_excl : Optional[int]
        Index of row group after last overlapping row group, if any.
    df_idx_rg_starts : ndarray
        Indices where each row group starts in DataFrame.
    df_idx_rg_ends_excl : ndarray
        Indices of the rows after the last overlapping rows in DataFrame.

    """

    df_idx_rg_starts: np.ndarray
    df_idx_rg_ends_excl: np.ndarray
    df_idx_overlap_start: Optional[int]
    df_idx_overlap_end_excl: Optional[int]
    rg_idx_overlap_start: Optional[int]
    rg_idx_overlap_end_excl: Optional[int]
    has_pf_head: bool
    has_df_head: bool
    has_overlap: bool
    has_pf_tail: bool
    has_df_tail: bool

    @classmethod
    def analyze(
        cls,
        df: DataFrame,
        pf: ParquetFile,
        ordered_on: str,
        max_row_group_size: int,
    ) -> "DataOverlapInfo":
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
        max_row_group_size : int
            Maximum number of rows per chunk. Must be positive.
        drop_duplicates : bool
            Flag impacting how overlap boundaries have to be managed.
            More exactly, 'pf' is considered as first data, and 'df' as second
            data, coming after. In case of 'pf' leading 'df', if last value in
            'pf' is a duplicate of the first in 'df', then
              - if True, at this index, overlap starts
              - if False, no overlap at this index

        Returns
        -------
        DataOverlapInfo
            Instance containing overlap analysis information.

        """
        # Find overlapping regions in dataframe
        rg_mins = pf.statistics[MIN][ordered_on]
        rg_maxs = pf.statistics[MAX][ordered_on]
        # Determine overlap start/end indices in row groups
        df_idx_rg_starts = np.searchsorted(df.loc[:, ordered_on], rg_mins, side=LEFT)
        df_idx_rg_ends_excl = np.searchsorted(df.loc[:, ordered_on], rg_maxs, side=RIGHT)
        rg_idx_overlap_start = df_idx_rg_ends_excl.astype(bool).argmax()
        rg_idx_overlap_end_excl = df_idx_rg_ends_excl.argmax() + 1
        # Analyze overlap patterns
        has_pf_head = rg_idx_overlap_start > 0 or df_idx_rg_ends_excl[-1] == 0
        has_df_head = df_idx_rg_starts[0] >= max_row_group_size
        has_pf_tail = rg_idx_overlap_end_excl < len(rg_mins) and df_idx_rg_ends_excl[-1] != 0
        has_df_tail = df_idx_rg_ends_excl[rg_idx_overlap_end_excl - 1] < len(df)

        if df_idx_rg_starts[0] == len(df) or df_idx_rg_ends_excl[-1] == 0:
            # Case no overlap.
            has_overlap = False
            rg_idx_overlap_start = None
            rg_idx_overlap_end_excl = None
            df_idx_overlap_start = None
            df_idx_overlap_end_excl = None
        else:
            # Case overlap.
            has_overlap = True
            df_idx_overlap_start = df_idx_rg_starts[rg_idx_overlap_start]
            df_idx_overlap_end_excl = df_idx_rg_ends_excl[rg_idx_overlap_end_excl - 1]

        return cls(
            has_pf_head=has_pf_head,
            has_df_head=has_df_head,
            has_overlap=has_overlap,
            has_pf_tail=has_pf_tail,
            has_df_tail=has_df_tail,
            df_idx_overlap_start=df_idx_overlap_start,
            df_idx_overlap_end_excl=df_idx_overlap_end_excl,
            rg_idx_overlap_start=rg_idx_overlap_start,
            rg_idx_overlap_end_excl=rg_idx_overlap_end_excl,
            df_idx_rg_starts=df_idx_rg_starts,
            df_idx_rg_ends_excl=df_idx_rg_ends_excl,
        )
