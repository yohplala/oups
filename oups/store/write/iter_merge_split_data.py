#!/usr/bin/env python3
"""
Created on Fri Nov  8 22:30:00 2024.

@author: yoh

"""
from typing import Callable, List, Optional, Tuple, Union

from fastparquet import ParquetFile
from numpy.typing import NDArray
from pandas import DataFrame
from pandas import Series
from pandas import concat


def iter_merge_split_data(
    opd: ParquetFile,
    ordered_on: Union[str, Tuple[str]],
    df: DataFrame,
    merge_sequences: List[Tuple[int, NDArray]],
    split_sequence: Callable[[Series], List[int]],
    drop_duplicates: bool,
    subset: Optional[Union[str, List[str]]] = None,
):
    """
    Yield merged and ordered chunks of data from DataFrame and ParquetFile.

    Parameters
    ----------
    opd : ParquetFile
        Ordered parquet dataset to merge with dataframe. Must be ordered by
        'ordered_on' column.
    ordered_on : Union[str, Tuple[str]]
        Column name by which data is ordered. Data must be in ascending order.
    df : Union[DataFrame, None]
        DataFrame (pandas) ordered by 'ordered_on' column.
    merge_sequences : List[Tuple[int, NDArray]]
        Merge sequences defining how to merge data from parquet and dataframe,
        as produced by 'NRowsMergeSplitStrategy' or
        'TimePeriodMergeSplitStrategy', with 'compute_merge_sequences' method.
    split_sequence : Callable[[Series], Iterable[int]]
        Method 'compute_split_sequence' from 'NRowsMergeSplitStrategy' or
        'TimePeriodMergeSplitStrategy' to compute split sequence from
        'ordered_on' column. Used to determine where to split the merged data
        into row groups.
    drop_duplicates : bool
        If ``True``, duplicates are removed from both the pandas DataFrame and
        the corresponding Parquet data overlapping with it.
    subset : Optional[Union[str, List[str]]], default None
        Column(s) to check for duplicates. If ``None``, all columns are used.

    Yields
    ------
    DataFrame
        Chunks of merged and ordered data according 'merge_sequences' and
        'split_sequence'.

    Raises
    ------
    ValueError
        If second item in first merge sequence is not a 2D numpy array.
        This format applies to all merge sequences, but check is managed
        only on first merge sequence.

    Notes
    -----
    If 'duplicates_on' is ``True````, duplicates are removed from both the
    pandas DataFrame and the corresponding Parquet data overlapping with it.
    In case there would be duplicates only within the Parquet data, while there
    is no overlap with the pandas DataFrame, then duplicates are not removed
    from Parquet data.

    Parquet data is supposed to be anterior to data in pandas DataFrame. It
    means that in case of duplicate values in 'ordered_on' columns of both data,
    values from the pandas DataFrame will be positioned behind after the merge.

    """
    if len(merge_sequences) == 0:
        return
    elif merge_sequences[0][1].ndim != 2:
        # Check shape of 'cmpt_ends_excl'array of 1st merge sequence.
        raise ValueError(
            "2nd item in merge sequences should be 2D numpy array, got ndim "
            f"{merge_sequences[0][1].ndim}.",
        )
    df_idx_start = 0
    for rg_idx_start, cmpt_ends_excl in merge_sequences:
        # 1st loop over merge sequences.
        # Between each merge sequence, remainder is reset.
        remainder = None
        for chunk_countdown, (rg_idx_end_excl, df_idx_end_excl) in enumerate(
            cmpt_ends_excl,
            start=-len(cmpt_ends_excl) + 1,
        ):
            # 2nd loop over row group to merge, with overlapping Dataframe
            # chunks.
            opd_chunk = (
                None
                if rg_idx_start == rg_idx_end_excl
                else opd[rg_idx_start:rg_idx_end_excl].to_pandas()
            )
            df_chunk = (
                None
                if (df_idx_start == df_idx_end_excl or df is None)
                else df.iloc[df_idx_start:df_idx_end_excl]
            )
            chunk = concat(
                [
                    remainder,
                    opd_chunk,
                    df_chunk,
                ],
                ignore_index=True,
            ).sort_values(ordered_on, ignore_index=True)
            if drop_duplicates:
                chunk.drop_duplicates(
                    subset=subset,
                    keep="last",
                    ignore_index=True,
                    inplace=True,
                )
            row_group_ends_excl = split_sequence(chunk.loc[:, ordered_on])[1:]
            row_idx_start = 0
            for row_idx_end_excl in row_group_ends_excl:
                # 3rd loop over merged data to split into row groups.
                yield chunk.iloc[row_idx_start:row_idx_end_excl]
                row_idx_start = row_idx_end_excl
            if chunk_countdown == 0:
                # When reaching end of current merge sequence, yield last chunk.
                yield chunk.iloc[row_idx_start:]
            else:
                # Otherwise, keep last chunk as remainder for next merge
                # sequence.
                remainder = chunk.iloc[row_idx_start:].copy(deep=True)
            del chunk
            rg_idx_start = rg_idx_end_excl
            df_idx_start = df_idx_end_excl
