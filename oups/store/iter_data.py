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


def _validate_duplicate_on_param(
    duplicates_on: Union[str, List[str]],
    distinct_bounds: bool,
    ordered_on: str,
    columns: Iterable[str],
) -> List[str]:
    """
    Validate and normalize duplicate parameters.

    Parameters
    ----------
    duplicates_on : Union[str, List[str]]
        Column(s) to check for duplicates. If empty list, all columns are used.
    distinct_bounds : bool
        If True, ensures row group boundaries do not split duplicate rows.
    ordered_on : str
        Column name by which data is ordered.
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
                "Distinct bounds must be enabled when setting 'duplicates_on'.",
            )
        if isinstance(duplicates_on, list):
            if duplicates_on == []:
                return list(columns)
            if not all(col in columns for col in duplicates_on):
                raise ValueError("one or more duplicate columns not found in input DataFrame")
            if ordered_on not in duplicates_on:
                duplicates_on.append(ordered_on)
        else:
            if duplicates_on not in columns:
                raise ValueError(f"column '{duplicates_on}' not found in input DataFrame")
            if duplicates_on != ordered_on:
                return [duplicates_on, ordered_on]
            return [ordered_on]
    return duplicates_on


def _get_next_chunk(
    df: DataFrame,
    start_idx: int,
    size: int,
    ordered_on: str,
    distinct_bounds: Optional[bool] = False,
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
    ordered_on : str
        Column name by which data is ordered.
    distinct_bounds : Optional[bool]
        If True, ensures that the chunk does not split duplicates in the 'ordered_on' column.

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


def _iter_pandas_dataframe(
    df: DataFrame,
    max_row_group_size: int,
    ordered_on: str,
    start_df: Optional[DataFrame] = None,
    distinct_bounds: bool = False,
    duplicates_on: Optional[Union[str, List[str]]] = None,
    yield_remainder: bool = False,
) -> Iterable[DataFrame]:
    """
    Split pandas DataFrame into row groups.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame to split.
    max_row_group_size : int
        Maximum number of rows per row group.
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    start_df : Optional[DataFrame]
        Data to start the iteration with. Must be ordered by the same column.
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
        Chunks of the DataFrame, each with size <= max_row_group_size.

    Returns
    -------
    Optional[DataFrame]
        Remaining data if yield_remainder is False and final chunk is smaller
        than max_row_group_size.

    """
    start_idx = 0
    if start_df is not None:
        df = concat([start_df, df])
        del start_df

    if duplicates_on:
        df = df.drop_duplicates(duplicates_on, keep="last", ignore_index=True)

    while len(df) - start_idx >= max_row_group_size:
        chunk, next_idx = _get_next_chunk(
            df,
            start_idx,
            max_row_group_size,
            ordered_on,
            distinct_bounds,
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


def _iter_resized_parquet_file(
    pf: ParquetFile,
    max_row_group_size: int,
    ordered_on: str,
    start_df: Optional[DataFrame] = None,
    distinct_bounds: bool = False,
    yield_remainder: bool = False,
):
    """
    Yield resized row groups from ParquetFile.

    Reads row groups from ParquetFile and yields chunks of data that respect
    the maximum row group size. If a starter DataFrame is provided, it will be
    concatenated with the first chunk. Handles distinct bounds to prevent splitting
    of duplicate values.

    Parameters
    ----------
    pf : ParquetFile
        The ParquetFile to iterate over.
    max_row_group_size : int
        Maximum number of rows per row group.
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    start_df : Optional[DataFrame], default None
        Data to start the iteration with. Must be ordered by the same column.
    distinct_bounds : bool, default False
        If True, ensures that row group boundaries do not split duplicate rows
        in the ordered_on column.
    yield_remainder : bool, default False
        If True, yields the last chunk of data even if it is smaller than
        max_row_group_size.

    Yields
    ------
    DataFrame
        Chunks of data with size <= max_row_group_size.

    Returns
    -------
    Optional[DataFrame]
        Remaining data if yield_remainder is False and final chunk is smaller
        than max_row_group_size.

    Notes
    -----
    The function maintains the ordering of data throughout the chunking process
    and ensures that duplicate values in the ordered_on column stay together
    when distinct_bounds is True.

    """
    start_rg_idx = 0
    if start_df is None:
        buffer_num_rows = 0
    else:
        buffer_num_rows = len(start_df)
    remainder = start_df

    for rg_idx, rg in enumerate(pf.row_groups, start=1):
        buffer_num_rows += rg.num_rows
        if buffer_num_rows >= max_row_group_size:
            data = pf[start_rg_idx:rg_idx].to_pandas()
            if remainder is not None:
                data = concat([remainder, data], ignore_index=True, inplace=True)
                del remainder
            chunk, end_idx = _get_next_chunk(
                data,
                0,
                max_row_group_size,
                ordered_on,
                distinct_bounds,
            )
            yield chunk
            remainder = data.iloc[end_idx:].copy(deep=True) if buffer_num_rows > end_idx else None
            del data
            start_rg_idx = rg_idx - 1
            buffer_num_rows = len(remainder) if remainder is not None else 0

    if yield_remainder:
        yield remainder
    else:
        return remainder


def iter_merged_pandas_parquet_file(
    df: DataFrame,
    pf: ParquetFile,
    max_row_group_size: int,
    ordered_on: str,
    distinct_bounds: Optional[bool] = False,
    duplicates_on: Optional[Union[str, Iterable[str]]] = None,
):
    """
    Yield merged and ordered chunks of data from DataFrame and ParquetFile.

    Parameters
    ----------
    df : DataFrame
        In-memory pandas DataFrame to process. Must be ordered by ordered_on column.
    pf : ParquetFile
        ParquetFile to merge with data. Must be ordered by ordered_on column.
    max_row_group_size : int
        Max number of rows per chunk.
    ordered_on : str
        Column name by which data is ordered. Data must be in ascending order.
    distinct_bounds : Optional[bool], default False
        If True, ensures that row group boundaries do not split duplicate rows.
    duplicates_on : Optional[Union[str, Iterable[str]]]
        Column(s) to check for duplicates. If empty list, all columns are used.

    Yields
    ------
    DataFrame
        Chunks of merged and ordered data, each with size <= max_row_group_size.

    Raises
    ------
    ValueError
        If ordered_on column is not in data.
        If distinct_bounds is False while duplicates_on is set.

    """
    if df.empty:
        return

    if ordered_on not in df.columns:
        raise ValueError(f"column '{ordered_on}' not found in input DataFrame")

    if duplicates_on is not None:
        duplicates_on = _validate_duplicate_on_param(
            duplicates_on,
            distinct_bounds,
            ordered_on,
            list(df.columns),
        )

    # Identify overlapping row groups
    rg_mins = pf.statistics["min"][ordered_on]
    rg_maxs = pf.statistics["max"][ordered_on]
    data_values = df[ordered_on].to_numpy()
    rg_df_start_idxs = searchsorted(data_values, rg_mins, side="left")
    rg_df_end_idxs = searchsorted(data_values, rg_maxs, side="right")
    pf_rg_overlap_start_idx = rg_df_end_idxs.astype(bool).argmax() if rg_df_end_idxs[0] == 0 else 0
    pf_rg_overlap_end_idx = (
        rg_df_start_idxs.argmax() if rg_df_start_idxs[-1] == len(df) - 1 else None
    )

    # Handle data before the first overlaps.
    remainder = None
    df_start_idx = 0
    if pf_rg_overlap_start_idx:
        # Case there is parquet file data before the first overlapping row group.
        remainder = yield from _iter_resized_parquet_file(
            pf[:pf_rg_overlap_start_idx],
            max_row_group_size,
            ordered_on,
            distinct_bounds,
            yield_last=False,
        )
    elif (no_overlap_df_last_idx := rg_df_start_idxs[0]) > max_row_group_size:
        # Case there is sufficient data in the pandas DataFrame to start a new row group.
        df_start_idx = no_overlap_df_last_idx + 1
        remainder = yield from _iter_pandas_dataframe(
            df.iloc[:df_start_idx],
            max_row_group_size,
            ordered_on,
            distinct_bounds,
            duplicates_on,
            yield_last=False,
        )

    # Merge overlapping data.
    buffer_num_rows = 0 if remainder is None else len(remainder)
    for rg_idx, (_df_start_idx, _df_end_idx) in enumerate(
        zip(
            rg_df_start_idxs[pf_rg_overlap_start_idx:pf_rg_overlap_end_idx],
            rg_df_end_idxs[pf_rg_overlap_start_idx:pf_rg_overlap_end_idx],
        ),
        start=pf_rg_overlap_start_idx,
    ):
        n_data_rows = _df_end_idx - _df_start_idx
        buffer_num_rows += pf.row_groups[rg_idx].num_rows + n_data_rows

        if buffer_num_rows >= max_row_group_size:
            df_chunk = df.iloc[df_start_idx:_df_end_idx]
            rg_idx_1 = rg_idx + 1
            pf_chunk = pf[pf_rg_overlap_start_idx:rg_idx_1].to_pandas()
            merged = concat([remainder, df_chunk, pf_chunk], ignore_index=True)
            merged.sort_values(ordered_on, inplace=True, ignore_index=True)

            remainder = yield from _iter_pandas_dataframe(
                merged,
                max_row_group_size,
                ordered_on,
                distinct_bounds,
                duplicates_on,
                yield_last=False,
            )
            df_start_idx = _df_end_idx
            pf_rg_overlap_start_idx = rg_idx_1
            buffer_num_rows = 0 if remainder is None else len(remainder)

    # Handle data after the last overlaps.
    if pf_rg_overlap_end_idx:
        # Case there is parquet file data after the last overlapping row group.
        yield from _iter_resized_parquet_file(
            pf[pf_rg_overlap_end_idx:],
            max_row_group_size,
            ordered_on,
            remainder,
            distinct_bounds,
            yield_last=True,
        )
    elif _df_end_idx < len(df):
        yield from _iter_pandas_dataframe(
            df.iloc[_df_end_idx:],
            max_row_group_size,
            ordered_on,
            remainder,
            distinct_bounds,
            duplicates_on,
            yield_last=True,
        )
