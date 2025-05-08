#!/usr/bin/env python3
"""
Created on Wed Dec  6 22:30:00 2021.

@author: yoh

"""
from pickle import dumps
from pickle import loads
from typing import Dict, Hashable, List, Optional, Tuple, Union

from fastparquet import ParquetFile
from fastparquet.util import update_custom_metadata
from numpy import array
from numpy import dtype
from pandas import DataFrame
from pandas import Series

from oups.store.defines import OUPS_METADATA_KEY
from oups.store.write.iter_merge_data import iter_merge_data
from oups.store.write.merge_split_strategies import NRowsMergeSplitStrategy
from oups.store.write.merge_split_strategies import TimePeriodMergeSplitStrategy


COMPRESSION = "SNAPPY"
DTYPE_DATETIME64 = dtype("datetime64[ns]")
EMPTY_DATAFRAME = DataFrame()
MIN = "min"
MAX = "max"
MAX_ROW_GROUP_SIZE = 6_345_000
# TODO: remove below KEY_MAX_ROW_GROUP_SIZE
KEY_MAX_ROW_GROUP_SIZE = "row_group_target_size"
KEY_ROW_GROUP_TARGET_SIZE = "row_group_target_size"
KEY_DUPLICATES_ON = "duplicates_on"
# Notes to any dev.
# Store any oups-specific metadata in this dict, such as oups-based application
# metadata.
# When appending new data, use `OUPS_METADATA.update()`.
# `OUPS_METADATA` can be used as a buffer, to keep in memory the metadata to
# be updated, till a write is triggered.
# Metadata itself should be within a nested dict, referred to by a `md_key`.
# By use of a `md_key`, management in parallel of metadata for several keys is
# possible (i.e. several dataset in difference `ParquetFile`).
OUPS_METADATA = {}


def write_metadata(
    pf: ParquetFile,
    metadata: Dict[str, str] = None,
    md_key: Hashable = None,
):
    """
    Write metadata to disk.

    Update oups-specific metadata and merge to user-defined metadata.
    "oups-specific" metadata is retrieved from OUPS_METADATA dict.

    Parameters
    ----------
    pf : ParquetFile
        ParquetFile which metadata are to be updated.
    metadata : Dict[str, str], optional
        User-defined key-value metadata to write, or update in dataset. Please
        see fastparquet for updating logic in case of `None` value being used.
    md_key: Hashable, optional
        Key to retrieve data in ``OUPS_METADATA`` dict, and write it as
        specific oups metadata in parquet file. If not provided, all data
        in ``OUPS_METADATA`` dict are retrieved to be written.
        This parameter is not compulsory. It is needed for instance in case
        data is written at same time for several keys. Then the right metadata
        for each key can be found thanks to this label.

    Notes
    -----
    - Specific oups metadata are available in global variable ``OUPS_METADATA``.
    - Once merged to ``new_metadata``, ``OUPS_METADATA`` is reset.
    - Update strategy of oups specific metadata depends if key found in
      ``OUPS_METADATA``metadata` is also found in already existing metadata,
      as well as its value.

      - If not found in existing, it is added.
      - If found in existing, it is updated.
      - If its value is `None`, it is not added, and if found in existing, it
        is removed from existing.

    """
    if OUPS_METADATA and md_key and md_key in OUPS_METADATA:
        # If 'md_key' is 'None', then no metadata from ``OUPS_METADATA`` is
        # retrieved.
        new_oups_spec_md = OUPS_METADATA[md_key]
        if OUPS_METADATA_KEY in (existing_metadata := pf.key_value_metadata):
            # Case 'append' to existing metadata.
            # oups-specific metadata is expected to be a dict itself.
            # To be noticed, 'md_key' is not written itself in metadata to
            # disk.
            existing_oups_spec_md = loads(existing_metadata[OUPS_METADATA_KEY])
            for key, value in new_oups_spec_md.items():
                if key in existing_oups_spec_md:
                    if value is None:
                        # Case 'remove'.
                        del existing_oups_spec_md[key]
                    else:
                        # Case 'update'.
                        existing_oups_spec_md[key] = value
                elif value:
                    # Case 'add'.
                    existing_oups_spec_md[key] = value
        else:
            existing_oups_spec_md = new_oups_spec_md
        del OUPS_METADATA[md_key]

        if metadata:
            metadata[OUPS_METADATA_KEY] = dumps(existing_oups_spec_md)
        else:
            metadata = {OUPS_METADATA_KEY: dumps(existing_oups_spec_md)}

    if metadata:
        update_custom_metadata(pf, metadata)
    pf._write_common_metadata()


def _validate_duplicate_on_param(
    duplicates_on: Union[str, List[str]],
    ordered_on: str,
    columns: List[str],
) -> List[str]:
    """
    Validate and normalize duplicate parameters.

    Parameters
    ----------
    duplicates_on : Union[str, List[str]]
        Column(s) to check for duplicates. If empty list, all columns are used.
    ordered_on : str
        Column name by which data is ordered.
    columns : List[str]
        Available columns in the DataFrame.
        If an emlpty list, related check is not performed.

    Returns
    -------
    List[str]
        Normalized list of columns to check for duplicates, including
        'ordered_on' column.

    Raises
    ------
    ValueError
        If distinct_bounds is not set while duplicates_on is provided.

    """
    if isinstance(duplicates_on, list):
        if duplicates_on == []:
            if not columns:
                raise ValueError(
                    "not possible to set 'duplicates_on' to '[]' when not "
                    "providing a DataFrame.",
                )
            else:
                return list(columns)
        if columns and not all(col in columns for col in duplicates_on):
            # If columns is an empty list, it means Dataframe is empty.
            # Don't proceed with the check.
            raise ValueError("one or more duplicate columns not found in input DataFrame.")
        if ordered_on not in duplicates_on:
            duplicates_on.append(ordered_on)
        return duplicates_on
    else:
        # 'duplicates_on' is a single column name.
        if columns and duplicates_on not in columns:
            raise ValueError(f"column '{duplicates_on}' not found in input DataFrame.")
        if duplicates_on != ordered_on:
            return [duplicates_on, ordered_on]
        return [ordered_on]


def write_ordered(
    dirpath: str,
    ordered_on: Union[str, Tuple[str]],
    df: Optional[DataFrame] = EMPTY_DATAFRAME,
    row_group_target_size: Optional[Union[int, str]] = MAX_ROW_GROUP_SIZE,
    duplicates_on: Union[str, List[str], List[Tuple[str]]] = None,
    max_n_off_target_rgs: int = None,
    compression: str = COMPRESSION,
    metadata: Dict[str, str] = None,
    md_key: Hashable = None,
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
        It has two effects:

          - it allows knowing 'where' to insert new data into existing data,
            i.e. completing or correcting past records (but it does not allow
            to remove prior data).
          - along with 'sharp_on', it ensures that two consecutive row groups do
            not have duplicate values in column defined by ``ordered_on`` (only
            in row groups to be written). This implies that all possible
            duplicates in ``ordered_on`` column will lie in the same row group.

    df : Optional[DataFrame], default empty DataFrame
        Data to write. If not provided, an empty dataframe is created with a
        single 'ordered_on' column.
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
        If 'max_row_group_size' is an ``int``, then a 'complete' row group
        is one which size is 'close to' ``max_row_group_size`` (>=80%).
        If 'max_row_group_size' is a pandas `freqstr`, and if there are several
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
    compression : str, default SNAPPY
        Algorithm to use for compressing data. This parameter is fastparquet
        specific. Please see fastparquet documentation for more information.
    metadata : Dict[str, str], optional
        Key-value metadata to write, or update in dataset. Please see
        fastparquet for updating logic in case of `None` value being used.
    md_key: Hashable, optional
        Key to retrieve data in ``OUPS_METADATA`` dict, and write it as
        specific oups metadata in parquet file. If not provided, all data
        in ``OUPS_METADATA`` dict are retrieved to be written.

    Returns
    -------
    ParquetHandle
        Instance of ParquetHandle, to be used for further operations.

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

      - either not reaching the maximum number of rows if 'max_row_group_size'
        is an ``int``,
      - or several row groups lying in tge same time period if
        'max_row_group_size' is a pandas 'freqstr'.

    - When incorporating new data within recorded data, existing off target size
      row groups will only be resized if there is intersection with new data.
      Otherwise, new data is only added, without merging with existing off
      target size row groups.

    """
    if duplicates_on is not None:
        duplicates_on = _validate_duplicate_on_param(
            duplicates_on=duplicates_on,
            ordered_on=ordered_on,
            columns=list(df.columns),
        )
        drop_duplicates = True
    else:
        drop_duplicates = False
    if df.empty:
        # At least, start with an empty DataFrame containing 'ordered_on'
        # column.
        df.loc[:, ordered_on] = Series([])
    else:
        try:
            df_ordered_on = df.loc[:, ordered_on]
        except KeyError:
            # Check 'ordered_on' column is within input DataFrame.
            raise ValueError(f"column '{ordered_on}' does not exist in input DataFrame.")
    if isinstance(dirpath, str):
        from oups.store.router import ParquetHandle

        opd = ParquetHandle(dirpath, ordered_on=ordered_on, df_like=df)
    else:
        opd = dirpath
    opd_statistics = opd.statistics
    if isinstance(row_group_target_size, int):
        if drop_duplicates:
            # Duplicates are dropped a first time in the DataFrame, so that the
            # calculation of merge and split strategy is made with the most
            # correct approximate number of rows in DataFrame.
            df.drop_duplicates(duplicates_on, keep="last", ignore_index=True, inplace=True)
        ms_strategy = NRowsMergeSplitStrategy(
            rg_ordered_on_mins=array(opd_statistics[MIN][str(ordered_on)]),
            rg_ordered_on_maxs=array(opd_statistics[MAX][str(ordered_on)]),
            df_ordered_on=df_ordered_on,
            drop_duplicates=drop_duplicates,
            rgs_n_rows=array([rg.num_rows for rg in opd.row_groups], dtype=int),
            row_group_target_size=row_group_target_size,
        )
    else:
        ms_strategy = TimePeriodMergeSplitStrategy(
            rg_ordered_on_mins=array(opd_statistics[MIN][str(ordered_on)]),
            rg_ordered_on_maxs=array(opd_statistics[MAX][str(ordered_on)]),
            df_ordered_on=df_ordered_on,
            drop_duplicates=drop_duplicates,
            row_group_time_period=row_group_target_size,
        )
    opd.write_row_groups(
        data=iter_merge_data(
            opd=opd,
            ordered_on=ordered_on,
            df=df,
            merge_sequences=ms_strategy.compute_merge_sequences(
                max_n_off_target_rgs=max_n_off_target_rgs,
            ),
            split_sequence=ms_strategy.compute_split_sequence,
            duplicates_on=duplicates_on,
        ),
        row_group_offsets=None,
        sort_pnames=False,
        compression=compression,
        write_fmd=False,
    )
    # Remove row groups of data that is overlapping.
    for rg_idx_start_end_excl in ms_strategy.rg_idx_mrs_starts_ends_excl:
        opd.remove_row_groups(opd[rg_idx_start_end_excl].row_groups, write_fmd=False)
    # Rename partition files.
    if ms_strategy.sort_rgs_after_write:
        opd.sort_rgs(ordered_on)
        opd._sort_part_names(write_fmd=False)
    # Manage and write metadata.
    # TODO: when refactoring metadata writing, use straight away
    # 'update_common_metadata' from fastparquet.
    write_metadata(pf=opd, metadata=metadata, md_key=md_key)
