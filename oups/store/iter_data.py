#!/usr/bin/env python3
"""
Created on Fri Nov  8 22:30:00 2024.

@author: yoh

"""
from typing import Iterable, List, Optional, Tuple, Union

from fastparquet import ParquetFile
from numpy import searchsorted
from pandas import DataFrame
from pandas import concat

from oups.store.data_overlap import DataOverlapInfo


def _validate_duplicate_on_param(
    duplicates_on: Union[str, List[str]],
    ordered_on: str,
    distinct_bounds: bool,
    columns: Iterable[str],
) -> List[str]:
    """
    Validate and normalize duplicate parameters.

    Parameters
    ----------
    duplicates_on : Union[str, List[str]]
        Column(s) to check for duplicates. If empty list, all columns are used.
    ordered_on : str
        Column name by which data is ordered.
    distinct_bounds : bool
        If True, ensures row group boundaries do not split duplicate rows.
    columns : List[str]
        Available columns in the DataFrame.

    Returns
    -------
    List[str]
        Normalized list of columns to check for duplicates, including ordered_on.

    Raises
    ------
    ValueError
        If distinct_bounds is not set while duplicates_on is provided.

    """
    if duplicates_on is not None:
        if not distinct_bounds:
            raise ValueError(
                "distinct bounds must be enabled when setting 'duplicates_on'.",
            )
        if isinstance(duplicates_on, list):
            if duplicates_on == []:
                return list(columns)
            if not all(col in columns for col in duplicates_on):
                raise ValueError("one or more duplicate columns not found in input DataFrame.")
            if ordered_on not in duplicates_on:
                duplicates_on.append(ordered_on)
        else:
            if duplicates_on not in columns:
                raise ValueError(f"column '{duplicates_on}' not found in input DataFrame.")
            if duplicates_on != ordered_on:
                return [duplicates_on, ordered_on]
            return [ordered_on]
    return duplicates_on


def _get_next_chunk(
    df: DataFrame,
    start_idx: int,
    size: int,
    distinct_bounds: Optional[bool] = False,
    ordered_on: Optional[str] = None,
) -> Tuple[DataFrame, int]:
    """
    Get the next chunk of data, optionally respecting distinct boundaries.

    Parameters
    ----------
    df : DataFrame
        Source DataFrame.
    start_idx : int
        Starting index for the chunk.
    size : int
        Maximum number of rows in the chunk.
    distinct_bounds : Optional[bool]
        If True, ensures that the chunk does not split duplicates in the 'ordered_on' column.
    ordered_on : Optional[str]
        Column name by which data is ordered. Required if distinct_bounds is True.

    Returns
    -------
    Tuple[DataFrame, int]
        The chunk and the next starting index.

    """
    end_idx = min(start_idx + size, len(df))
    if distinct_bounds and end_idx < len(df):
        val_at_end = df[ordered_on].iloc[end_idx]
        # Find the leftmost index to not split duplicates.
        end_idx = searchsorted(df[ordered_on].to_numpy(), val_at_end)
        if end_idx == start_idx:
            # All values in chunk are duplicates.
            # Return chunk of data that will be larger than 'size', but complies
            # with distinct bounds.
            end_idx = searchsorted(df[ordered_on].to_numpy(), val_at_end, side="right")
    return df.iloc[start_idx:end_idx], end_idx


def _iter_df(
    ordered_on: str,
    max_row_group_size: int,
    df: Union[DataFrame, List[DataFrame]],
    distinct_bounds: bool = False,
    duplicates_on: Optional[Union[str, List[str]]] = None,
    yield_remainder: bool = False,
) -> Iterable[DataFrame]:
    """
    Split pandas DataFrame into row groups.

    Parameters
    ----------
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    max_row_group_size : int
        Maximum number of rows per row group.
    df : Union[DataFrame, List[DataFrame]]
        Pandas DataFrame to split. If a list, they are merged and sorted back by
        'ordered_on' column.
    distinct_bounds : bool, default False
        If True, ensures that row group boundaries do not split duplicate rows.
    duplicates_on : Optional[Union[str, List[str]]], default None
        Column(s) to check for duplicates. If provided, duplicates will be
        removed keeping last occurrence.
    yield_remainder : bool, default False
        If True, yields the last chunk of data even if it is smaller than
        'max_row_group_size'.

    Yields
    ------
    DataFrame
        Chunks of the DataFrame, each with size <= max_row_group_size, except
        if distinct_bounds is True and there are more duplicates in the
        'ordered_on' column than max_row_group_size.

    Returns
    -------
    Optional[DataFrame]
        Remaining data if yield_remainder is False and final chunk is smaller
        than max_row_group_size.

    """
    if isinstance(df, list):
        df = concat(df, ignore_index=True).sort_values(ordered_on, ignore_index=True)

    if duplicates_on:
        df.drop_duplicates(duplicates_on, keep="last", ignore_index=True, inplace=True)

    start_idx = 0
    while len(df) - start_idx >= max_row_group_size:
        chunk, next_idx = _get_next_chunk(
            df=df,
            start_idx=start_idx,
            size=max_row_group_size,
            distinct_bounds=distinct_bounds,
            ordered_on=ordered_on,
        )
        yield chunk
        start_idx = next_idx

    if start_idx < len(df):
        chunk = df.iloc[start_idx:].copy(deep=True)
        del df
        if yield_remainder:
            yield chunk
        else:
            return chunk


def iter_merged_pf_df(
    df: DataFrame,
    pf: ParquetFile,
    ordered_on: str,
    max_row_group_size: int,
    distinct_bounds: Optional[bool] = False,
    duplicates_on: Optional[Union[str, Iterable[str]]] = None,
):
    """
    Yield merged and ordered chunks of data from DataFrame and ParquetFile.

    Parameters
    ----------
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    max_row_group_size : int
        Max number of rows per chunk.
    df : DataFrame
        In-memory pandas DataFrame to process. Must be ordered by 'ordered_on'
        column.
    pf : ParquetFile
        ParquetFile to merge with data. Must be ordered by 'ordered_on' column.
    distinct_bounds : Optional[bool], default False
        If True, ensures that row group boundaries do not split duplicate rows.
    duplicates_on : Optional[Union[str, Iterable[str]]]
        Column(s) to check for duplicates. If empty list, all columns are used.
        If duplicates are found, only the last occurrence is kept.

    Yields
    ------
    DataFrame
        Chunks of merged and ordered data, each with size <= max_row_group_size.

    Raises
    ------
    ValueError
        If ordered_on column is not in data.
        If distinct_bounds is False while duplicates_on is set.

    Notes
    -----
    If 'duplicates_on' is set, duplicates are removed either from the pandas
    DataFrame, or from merged data between pandas DataFrame and ParquetFile.
    In case there would be duplicates only within the ParquetFile, while there
    is no overlap with the pandas DataFrame, then duplicates are not removed
    from ParquetFile.

    Data in ParquetFile is supposed to be anterior to data in pandas DataFrame.
    It means that in case of duplicate values in 'ordered_on' column, the last
    occurrence are those from the pandas DataFrame.

    """
    if df.empty:
        return

    if ordered_on not in df.columns:
        raise ValueError(f"column '{ordered_on}' not found in input DataFrame.")

    if duplicates_on is not None:
        duplicates_on = _validate_duplicate_on_param(
            duplicates_on=duplicates_on,
            distinct_bounds=distinct_bounds,
            ordered_on=ordered_on,
            columns=list(df.columns),
        )

    if duplicates_on:
        df.drop_duplicates(duplicates_on, keep="last", ignore_index=True, inplace=True)

    # Identify overlapping row groups
    overlap_info = DataOverlapInfo.analyze(
        df=df,
        pf=pf,
        ordered_on=ordered_on,
        max_row_group_size=max_row_group_size,
    )

    # Handle data in 'df' before the loop over row groups in 'pf'.
    remainder = None
    df_idx_start = 0
    if overlap_info.has_df_head:
        # Case there is sufficient data in the pandas DataFrame to start a new
        # row group.
        # If 'duplicates_on' is provided, duplicates have been removed already,
        # so no need to remove them again.
        remainder = yield from _iter_df(
            df=df.iloc[: overlap_info.df_idx_overlap_start],
            ordered_on=ordered_on,
            max_row_group_size=max_row_group_size,
            distinct_bounds=distinct_bounds,
            duplicates_on=None,
            yield_remainder=False,
        )
        # Correct then 'df_idx_start' to account for the dataframe head already
        # yielded.
        df_idx_start = overlap_info.df_idx_overlap_start if overlap_info.has_overlap else len(df)

    # Merge possibly overlapping data (full loop over 'pf' row groups).
    rg_idx_start = 0
    buffer_num_rows = 0 if remainder is None else len(remainder)
    for rg_idx_1, (_df_idx_start, df_idx_end) in enumerate(
        zip(overlap_info.df_idx_rg_starts, overlap_info.df_idx_rg_ends_excl),
        start=1,
    ):
        n_data_rows = df_idx_end - _df_idx_start
        buffer_num_rows += pf.row_groups[rg_idx_1 - 1].num_rows + n_data_rows
        if buffer_num_rows >= max_row_group_size or rg_idx_1 == len(pf):
            chunk = pf[rg_idx_start:rg_idx_1].to_pandas()
            if df_idx_start != df_idx_end:
                # Merge with pandas DataFrame chunk.
                # DataFrame chunk is added last in concat, to preserve
                # its values in case of duplicates found with values in
                # ParquetFile chunk.
                chunk = [remainder, chunk, df.iloc[df_idx_start:df_idx_end]]
            elif remainder is not None:
                chunk = [remainder, chunk]
            if buffer_num_rows >= max_row_group_size:
                remainder = yield from _iter_df(
                    df=chunk,
                    ordered_on=ordered_on,
                    max_row_group_size=max_row_group_size,
                    distinct_bounds=distinct_bounds,
                    duplicates_on=duplicates_on,
                    yield_remainder=False,
                )
                df_idx_start = df_idx_end
                rg_idx_start = rg_idx_1
                buffer_num_rows = 0 if remainder is None else len(remainder)
            else:
                remainder = chunk

    # Handle data after the last overlaps.
    if overlap_info.has_df_tail:
        # Case there is remaining data in pandas DataFrame.
        df_idx_overlap_end_excl = overlap_info.df_idx_overlap_end_excl
        yield from _iter_df(
            df=[remainder, df.iloc[df_idx_overlap_end_excl:]],
            ordered_on=ordered_on,
            max_row_group_size=max_row_group_size,
            distinct_bounds=distinct_bounds,
            duplicates_on=None if remainder is None else duplicates_on,
            yield_remainder=True,
        )
    elif remainder is not None:
        # Case there only is a remainder from previous iterations.
        yield remainder
