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

from oups.store.ordered_merge_info import compute_ordered_merge_plan


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
    df_n_rows = len(df)
    end_idx = min(start_idx + size, df_n_rows)
    if distinct_bounds and end_idx < df_n_rows:
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
    row_group_size_target: int,
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
    row_group_size_target : int
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
        'row_group_size_target'.

    Yields
    ------
    DataFrame
        Chunks of the DataFrame, each with size <= row_group_size_target, except
        if distinct_bounds is True and there are more duplicates in the
        'ordered_on' column than row_group_size_target.

    Returns
    -------
    Optional[DataFrame]
        Remaining data if yield_remainder is False and final chunk is smaller
        than row_group_size_target.

    """
    if isinstance(df, list):
        df = concat(df, ignore_index=True).sort_values(ordered_on, ignore_index=True)

    if duplicates_on:
        df.drop_duplicates(duplicates_on, keep="last", ignore_index=True, inplace=True)

    start_idx = 0
    df_n_rows = len(df)
    while df_n_rows - start_idx >= row_group_size_target:
        chunk, next_idx = _get_next_chunk(
            df=df,
            start_idx=start_idx,
            size=row_group_size_target,
            distinct_bounds=distinct_bounds,
            ordered_on=ordered_on,
        )
        yield chunk
        start_idx = next_idx

    if start_idx < df_n_rows:
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
    row_group_size_target: int,
    distinct_bounds: Optional[bool] = False,
    duplicates_on: Optional[Union[str, Iterable[str]]] = None,
):
    """
    Yield merged and ordered chunks of data from DataFrame and ParquetFile.

    Parameters
    ----------
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    row_group_size_target : int
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
        Chunks of merged and ordered data, each with size <= row_group_size_target.

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

    # TODO:
    # correct here!!
    max_n_irgs = None

    # Identify overlapping row groups.
    # If intent is to drop duplicates, 'analyze_chunks_to_merge' has to be
    # applied on a DataFrame without duplicates, so that returned indices stay
    # consistent (versus a scenario duplicates are dropped afterwards).
    # /!\ drop duplicates before calling 'analyze_chunks_to_merge' /!\
    if duplicates_on:
        df.drop_duplicates(duplicates_on, keep="last", ignore_index=True, inplace=True)
    (
        rg_idx_starts,
        rg_idx_ends_excl,
        df_idx_ends_excl,
        row_group_sizer,
        sort_rgs_after_write,
    ) = compute_ordered_merge_plan(
        df=df,
        pf=pf,
        ordered_on=ordered_on,
        row_group_size_target=row_group_size_target,
        drop_duplicates=duplicates_on is not None,
        max_n_irgs=max_n_irgs,
    )

    has_pf = 0
    has_df = 0
    rg_idx_start = rg_idx = 0
    df_idx_start = _df_idx_start = 0
    # if isinstance(row_group_size_target, str):
    #    next_period_start = df.loc[:, ordered_on].iloc[0].ceil(row_group_size_target)
    # is_mixed_chunk = False
    # chunk_n_rows = 0
    remainder = None
    for chunk_countdown, rg_idx_start, rg_idx_end_excl, df_idx_end_excl in enumerate(
        zip(rg_idx_starts, rg_idx_ends_excl, df_idx_ends_excl),
        start=-len(df_idx_ends_excl),
    ):
        # Each step is a write step.
        if df_idx_end_excl > _df_idx_start:
            # Possible DataFrame contribution to chunk.
            has_df = True
        if rg_idx_end_excl > rg_idx_start:
            has_pf = True
        if not has_pf:
            chunk = (
                [remainder, df.iloc[df_idx_start:df_idx_end_excl]]
                if remainder is not None
                else df.iloc[df_idx_start:df_idx_end_excl]
            )
        elif not has_df:
            chunk = (
                [remainder, pf[rg_idx_start:rg_idx].to_pandas()]
                if remainder is not None
                else pf[rg_idx_start:rg_idx].to_pandas()
            )
        else:
            # Both pandas DataFrame and ParquetFile chunks.
            # If 'remainder' is None, it does not raise trouble for concat
            # step.
            chunk = [
                remainder,
                pf[rg_idx_start:rg_idx].to_pandas(),
                df.iloc[df_idx_start:df_idx_end_excl],
            ]
        remainder = yield from _iter_df(
            df=chunk,
            ordered_on=ordered_on,
            # /!\ Here, reuse algo similar to fastparquet when yielding
            # remaining, calculate actual row_group_size to equilibrate
            # number of rows in last rg.
            # Or calculate directly row group offsets?
            # row_group_size_target=list(range(len(chunk), step=row_group_size_target))
            # if max_n_irgs or chunk_countdown
            # else row_group_size_target,
            row_group_size_target=row_group_sizer(chunk, chunk_countdown),
            distinct_bounds=distinct_bounds,
            duplicates_on=duplicates_on,
            yield_remainder=not chunk_countdown,  # yield last chunk
        )
        df_idx_start = df_idx_end_excl
        rg_idx_start = rg_idx_end_excl
        has_pf = has_df = False


#    for chunk_idx, df_idx_end_excl in enumerate(chunk_counter, start=-len(chunk_counter)):
#        # 'chunk_idx' will be 0 when loop is over.
#        if df_idx_end_excl > _df_idx_start:
#            # Possible DataFrame contribution to chunk.
#            chunk_n_rows += df_idx_end_excl - 1 - _df_idx_start
#            has_df += 1
#        if is_mixed_chunk:
#            # Add ParquetFile contribution to chunk.
#            chunk_n_rows += pf[rg_idx].num_rows
#            has_pf += 1
#            rg_idx += 1
#
#        _df_idx_start = df_idx_end_excl
#        is_mixed_chunk = not is_mixed_chunk
#
#        if (
#            chunk_n_rows >= row_group_size_target
#            if isinstance(row_group_size_target, int)
#            else (_chunk_last_ts := df.loc[:, ordered_on].iloc[df_idx_end_excl - 1])
#            >= next_period_start
#        ) or not chunk_idx:
#             if not has_pf:
#                chunk = (
#                    [remainder, df.iloc[df_idx_start:df_idx_end_excl]]
#                    if remainder is not None
#                    else df.iloc[df_idx_start:df_idx_end_excl]
#                )
#            elif not has_df:
#                chunk = (
#                    [remainder, pf[rg_idx_start:rg_idx].to_pandas()]
#                    if remainder is not None
#                    else pf[rg_idx_start:rg_idx].to_pandas()
#                )
#            else:
#                # Both pandas DataFrame and ParquetFile chunks.
#                # If 'remainder' is None, it does not raise trouble for concat
#                # step.
#                chunk = [
#                    remainder,
#                    pf[rg_idx_start:rg_idx].to_pandas(),
#                    df.iloc[df_idx_start:df_idx_end_excl],
#                ]
#            remainder = yield from _iter_df(
#                # /!\ Here, reuse algo similar to fastparquet when yielding
#                # remaining, calculate actual row_group_size to equilibrate
#                # number of rows in last rg.
#                # Or calculate directly row group offsets?
#                df=chunk,
#                ordered_on=ordered_on,
#                row_group_size_target=row_group_size_target,
#                distinct_bounds=distinct_bounds,
#                duplicates_on=duplicates_on,
#                yield_remainder=not chunk_idx,
#            )
#            df_idx_start = df_idx_end_excl
#            if isinstance(row_group_size_target, str):
#                next_period_start = _chunk_last_ts.ceil(row_group_size_target)
#            rg_idx_start = rg_idx
#            chunk_n_rows = 0 if remainder is None else len(remainder)
#            has_pf = has_df = 0
