#!/usr/bin/env python3
"""
Created on Wed Dec  6 22:30:00 2021.

@author: yoh

"""
from importlib import import_module
from typing import Dict, List, Optional, Tuple, Union

from pandas import DataFrame
from pandas import Series

from oups.defines import KEY_FILE_IDS
from oups.defines import KEY_N_ROWS
from oups.defines import KEY_ORDERED_ON_MAXS
from oups.defines import KEY_ORDERED_ON_MINS
from oups.store.ordered_parquet_dataset.write.iter_merge_split_data import iter_merge_split_data
from oups.store.ordered_parquet_dataset.write.merge_split_strategies import NRowsMergeSplitStrategy
from oups.store.ordered_parquet_dataset.write.merge_split_strategies import (
    TimePeriodMergeSplitStrategy,
)


ROW_GROUP_INT_TARGET_SIZE = 6_345_000


def _validate_duplicate_on_param(
    duplicates_on: Union[str, List[str], List[Tuple[str]]],
    ordered_on: str,
) -> List[str]:
    """
    Validate and normalize duplicate parameters.

    Parameters
    ----------
    duplicates_on : Union[str, List[str], List[Tuple[str]]]
        Column(s) to check for duplicates. If empty list, all columns are used.
    ordered_on : str
        Column name by which data is ordered.

    Returns
    -------
    Tuple[bool, Union[List[str], None]]
        Boolean flag indicating if duplicates are to be dropped, and list of
        columns to check for duplicates, including 'ordered_on' column. If
        duplicates are dropped, a ``None`` indicates to consider all columns.

    Raises
    ------
    ValueError
        If distinct_bounds is not set while duplicates_on is provided.

    """
    if duplicates_on is None:
        return (False, None)
    else:
        if isinstance(duplicates_on, list):
            if duplicates_on == []:
                return (True, None)
            elif ordered_on not in duplicates_on:
                duplicates_on.append(ordered_on)
            return (True, duplicates_on)
        else:
            # 'duplicates_on' is a single column name.
            if duplicates_on != ordered_on:
                return (True, [duplicates_on, ordered_on])
            return (True, ordered_on)


def write(
    dirpath: str,
    ordered_on: Union[str, Tuple[str]],
    df: Optional[DataFrame] = None,
    row_group_target_size: Optional[Union[int, str]] = ROW_GROUP_INT_TARGET_SIZE,
    duplicates_on: Union[str, List[str], List[Tuple[str]]] = None,
    max_n_off_target_rgs: int = None,
    key_value_metadata: Dict[str, str] = None,
    **kwargs,
):
    """
    Write data to disk at location specified by path.

    Parameters
    ----------
    dirpath : Union[str, OrderedParquetDataset]
        If a string, it is the directory where writing pandas dataframe.
        If an OrderedParquetDataset, it is the dataset where writing pandas
        dataframe.
    ordered_on : Union[str, Tuple[str]]
        Name of the column with respect to which dataset is in ascending order.
        If column multi-index, name of the column is a tuple.
        It allows knowing 'where' to insert new data into existing data, i.e.
        completing or correcting past records (but it does not allow to remove
        prior data).
    df : Optional[DataFrame], default None
        Data to write. If None, a resize of Oredered Parquet Dataset may however
        be performed.
    row_group_target_size : Optional[Union[int, str]]
        Target size of row groups. If not set, default to ``6_345_000``, which
        for a dataframe with 6 columns of ``float64`` or ``int64`` results in a
        memory footprint (RAM) of about 290MB.
        It can be a pandas `freqstr` as well, to gather data by timestamp over a
        defined period.
    duplicates_on : Union[str, List[str], List[Tuple[str]]], optional
        Column names according which 'row duplicates' can be identified (i.e.
        rows sharing same values on these specific columns) so as to drop
        them. Duplicates are only identified in new data, and existing
        recorded row groups that overlap with new data.
        If duplicates are dropped, only last is kept.
        To identify row duplicates using all columns, empty list ``[]`` can be
        used instead of all columns names.
        If not set, default to ``None``, meaning no row is dropped.
    max_n_off_target_rgs : int, optional
        Max expected number of 'off target' row groups.
        If 'row_group_target_size' is an ``int``, then a 'complete' row group
        is one which size is 'close to' ``row_group_target_size`` (>=80%).
        If 'row_group_target_size' is a pandas `freqstr`, and if there are several
        row groups in the last period defined by the `freqstr`, then these row
        groups are considered incomplete.
        To evaluate number of 'incomplete' row groups, only those at the end of
        an existing dataset are accounted for. 'Incomplete' row groups in the
        middle of 'complete' row groups are not accounted for (they can be
        created by insertion of new data in the middle of existing data).
        If not set, default to ``None``.

          - ``None`` value induces no coalescing of row groups. If there is no
            drop of duplicates, new data is systematically appended.
          - A value of ``0`` or ``1`` means that new data should systematically
            be merged to the last existing one to 'complete' it (if it is not
            'complete' already).

    key_value_metadata : Dict[str, str], optional
        Key-value metadata to write, or update in dataset. Please see
        fastparquet for updating logic in case of `None` value being used.
    **kwargs :
        Additional parameters to pass to
        'OrderedParquetDataset.write_row_group_files()'.

    Notes
    -----
    - When writing a dataframe with this function,

      - index of dataframe is not written to disk.
      - parquet file scheme is 'hive' (one row group per parquet file).

    - Coalescing off target size row groups is triggered if actual number of off
      target row groups is larger than ``max_n_off_target_rgs``.
      This assessment is however only triggered if ``max_n_off_target_rgs`` is
      set. Otherwise, new data is simply appended, without prior check.
    - When ``duplicates_on`` is set, 'ordered_on' column is added to
      ``duplicates_on`` list, if not already part of it. Purpose is to enable a
      first approximate search for duplicates, to load data of interest only.
    - For simple data appending, i.e. without need to drop duplicates, it is
      advised to keep ``ordered_on`` and ``duplicates_on`` parameters set to
      ``None`` as this parameter will trigger unnecessary evaluations.
    - Off target size row groups are row groups:

      - either not reaching the maximum number of rows if
        'row_group_target_size' is an ``int``,
      - or several row groups lying in the same time period if
        'row_group_target_size' is a pandas 'freqstr'.

    - When incorporating new data within recorded data, existing off target size
      row groups will only be resized if there is intersection with new data.
      Otherwise, new data is only added, without merging with existing off
      target size row groups.

    """
    drop_duplicates, subset = _validate_duplicate_on_param(
        duplicates_on=duplicates_on,
        ordered_on=ordered_on,
    )
    df_ordered_on = Series([]) if df is None else df.loc[:, ordered_on]
    ordered_parquet_dataset = (
        # Case 'dirpath' is a path to a directory.
        import_module("oups.store.ordered_parquet_dataset").OrderedParquetDataset(
            dirpath,
            ordered_on=ordered_on,
        )
        if isinstance(dirpath, str)
        else
        # Case 'dirpath' is already an OrderedParquetDataset.
        dirpath
    )
    if df is not None or len(ordered_parquet_dataset):
        if isinstance(row_group_target_size, int):
            if drop_duplicates and df is not None:
                # Duplicates are dropped a first time in the DataFrame, so that the
                # calculation of merge and split strategy is made with the most
                # correct approximate number of rows in DataFrame.
                df.drop_duplicates(subset=subset, keep="last", ignore_index=True, inplace=True)
            merge_split_strategy = NRowsMergeSplitStrategy(
                rg_ordered_on_mins=ordered_parquet_dataset.row_group_stats.loc[
                    :,
                    KEY_ORDERED_ON_MINS,
                ].to_numpy(),  # rg_ordered_on_mins,
                rg_ordered_on_maxs=ordered_parquet_dataset.row_group_stats.loc[
                    :,
                    KEY_ORDERED_ON_MAXS,
                ].to_numpy(),  # rg_ordered_on_maxs,
                df_ordered_on=df_ordered_on,
                drop_duplicates=drop_duplicates,
                rgs_n_rows=ordered_parquet_dataset.row_group_stats.loc[:, KEY_N_ROWS],
                row_group_target_size=row_group_target_size,
            )
        else:
            merge_split_strategy = TimePeriodMergeSplitStrategy(
                rg_ordered_on_mins=ordered_parquet_dataset.row_group_stats.loc[
                    :,
                    KEY_ORDERED_ON_MINS,
                ].to_numpy(),  # rg_ordered_on_mins,
                rg_ordered_on_maxs=ordered_parquet_dataset.row_group_stats.loc[
                    :,
                    KEY_ORDERED_ON_MAXS,
                ].to_numpy(),  # rg_ordered_on_maxs,
                df_ordered_on=df_ordered_on,
                drop_duplicates=drop_duplicates,
                row_group_time_period=row_group_target_size,
            )
        ordered_parquet_dataset.write_row_group_files(
            dfs=iter_merge_split_data(
                opd=ordered_parquet_dataset,
                ordered_on=ordered_on,
                df=df,
                merge_sequences=merge_split_strategy.compute_merge_sequences(
                    max_n_off_target_rgs=max_n_off_target_rgs,
                ),
                split_sequence=merge_split_strategy.compute_split_sequence,
                drop_duplicates=drop_duplicates,
                subset=subset,
            ),
            write_metadata_file=False,
            **kwargs,
        )
        # Remove row groups of data that is overlapping.
        file_id_col_idx = ordered_parquet_dataset.row_group_stats.columns.get_loc(KEY_FILE_IDS)
        rg_file_ids_to_remove = [
            file_id
            for rg_idx_mr_start_end_excl in merge_split_strategy.rg_idx_mrs_starts_ends_excl
            for file_id in ordered_parquet_dataset.row_group_stats.iloc[
                rg_idx_mr_start_end_excl,
                file_id_col_idx,
            ].to_list()
        ]
        if rg_file_ids_to_remove:
            ordered_parquet_dataset.remove_row_group_files(
                file_ids=rg_file_ids_to_remove,
                sort_row_groups=merge_split_strategy.sort_rgs_after_write,
                key_value_metadata=key_value_metadata,
            )
            # 'remove_row_group_files()' embeds calls to 'sort_row_groups()' and
            # 'align_file_ids()' methods.
            return
        # Rename partition files.
        elif merge_split_strategy.sort_rgs_after_write:
            ordered_parquet_dataset.sort_row_groups()
            ordered_parquet_dataset.align_file_ids()
    # Write metadata.
    ordered_parquet_dataset.write_metadata_file(key_value_metadata=key_value_metadata)
