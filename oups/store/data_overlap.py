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
    rg_idx_overlap_end : Optional[int]
        Index of last overlapping row group, if any.
    df_idx_rg_starts : ndarray
        Indices where each row group starts in DataFrame.
    df_idx_rg_ends : ndarray
        Indices where each row group ends in DataFrame.

    """

    df_idx_rg_starts: np.ndarray
    df_idx_rg_ends: np.ndarray
    df_idx_overlap_start: Optional[int]
    df_idx_overlap_end_excl: Optional[int]
    rg_idx_overlap_start: Optional[int]
    rg_idx_overlap_end: Optional[int]
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
            Input DataFrame.
        pf : ParquetFile
            Input ParquetFile.
        ordered_on : str
            Column name by which data is ordered.
        max_row_group_size : int
            Maximum number of rows per chunk.

        Returns
        -------
        DataOverlapInfo
            Instance containing overlap analysis information.

        """
        # Find overlapping regions in dataframe
        rg_mins = pf.statistics["min"][ordered_on]
        rg_maxs = pf.statistics["max"][ordered_on]
        df_idx_rg_starts = np.searchsorted(df.loc[:, ordered_on], rg_mins, side="left")
        df_idx_rg_ends = np.searchsorted(df.loc[:, ordered_on], rg_maxs, side="right")

        # Determine overlap start/end indices in row groups
        rg_idx_overlap_start = df_idx_rg_ends.astype(bool).argmax()
        rg_idx_overlap_end = df_idx_rg_ends.argmax()
        # Analyze overlap patterns
        has_pf_head = rg_idx_overlap_start > 0 or df_idx_rg_ends[-1] == 0
        has_df_head = df_idx_rg_starts[0] >= max_row_group_size
        has_pf_tail = rg_idx_overlap_end + 1 < len(rg_mins) and df_idx_rg_ends[-1] != 0
        has_df_tail = df_idx_rg_ends[rg_idx_overlap_end] < len(df)
        if rg_idx_overlap_start != rg_idx_overlap_end:
            has_overlap = True
            df_idx_overlap_start = df_idx_rg_starts[rg_idx_overlap_start]
            df_idx_overlap_end_excl = df_idx_rg_ends[rg_idx_overlap_end]
        else:
            has_overlap = False
            rg_idx_overlap_start = None
            rg_idx_overlap_end = None
            df_idx_overlap_start = None
            df_idx_overlap_end_excl = None

        return cls(
            has_pf_head=has_pf_head,
            has_df_head=has_df_head,
            has_overlap=has_overlap,
            has_pf_tail=has_pf_tail,
            has_df_tail=has_df_tail,
            df_idx_overlap_start=df_idx_overlap_start,
            df_idx_overlap_end_excl=df_idx_overlap_end_excl,
            rg_idx_overlap_start=rg_idx_overlap_start,
            rg_idx_overlap_end=rg_idx_overlap_end,
            df_idx_rg_starts=df_idx_rg_starts,
            df_idx_rg_ends=df_idx_rg_ends,
        )
