#!/usr/bin/env python3
"""
Created on Wed Dec  6 22:30:00 2021.

@author: yoh

"""
from ast import literal_eval
from os import listdir as os_listdir
from os import path as os_path
from pickle import dumps
from pickle import loads
from typing import Dict, Hashable, List, Optional, Tuple, Union

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from fastparquet.api import filter_row_groups
from fastparquet.api import statistics
from fastparquet.util import update_custom_metadata
from numpy import dtype
from numpy import searchsorted
from numpy import unique
from pandas import DataFrame
from pandas import Index
from pandas import MultiIndex
from pandas import Timestamp
from pandas import concat
from pandas import date_range


DTYPE_DATETIME64 = dtype("datetime64[ns]")
COMPRESSION = "SNAPPY"
MAX_ROW_GROUP_SIZE = 6_345_000
MAX_ROW_GROUP_SIZE_SCALE_FACTOR = 0.9
KEY_MAX_ROW_GROUP_SIZE = "max_row_group_size"
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
# In a fastparquet `ParquetFile`, oups-specific metadata is stored as value for
# key `OUPS_METADATA_KEY`.
OUPS_METADATA_KEY = "oups"


def iter_dataframe(
    data: DataFrame,
    max_row_group_size: Union[int, str],
    sharp_on: str = None,
    duplicates_on: Union[str, List[str]] = None,
):
    """
    Yield dataframe chunks.

    Parameters
    ----------
    data : DataFrame
        Data to split in row groups.
    max_row_group_size : Union[int, str]
        Max size of row groups. It is a max as duplicates are dropped by row
        group to be written, hereby reducing the row group size (if
        ``duplicates_on`` parameter is set).
    sharp_on : str, optional
        Name of column where to check that ends of bins (which split the data
        to be written) do not fall in the middle of duplicate values.
        This parameter is required when ``duplicates_on`` is used.
        If not set, default to ``None``.
    duplicates_on : Union[str, List[str]], optional
        If set, drop duplicates based on list of column names, keeping last.
        Only duplicates within the same row group to be written are identified.
        If not set, default to ``None``.
        If an empty list ``[]``, all columns are used to identify duplicates.

    Yields
    ------
    DataFrame
        Chunk of data.

    Notes
    -----
    - Because duplicates are identified within a same row group, it is
      required to set ``sharp_on`` when using ``duplicates_on``, so that
      duplicates all fall in a same row group. This implies that duplicates
      have to share the same value in ``sharp_on`` column.

    """
    if duplicates_on is not None:
        if not sharp_on:
            raise ValueError(
                "duplicates are looked for row group per group. For this reason, "
                "it is compulsory to set 'sharp_on' while setting 'duplicates_on'.",
            )
        elif isinstance(duplicates_on, list):
            if duplicates_on and sharp_on not in duplicates_on:
                # Case 'not an empty list', and 'ordered_on' not in.
                duplicates_on.append(sharp_on)
        elif duplicates_on != sharp_on:
            # Case 'duplicates_on' is a single column name, but not
            # 'sharp_on'.
            duplicates_on = [duplicates_on, sharp_on]
    if n_rows := len(data):
        # Define bins to split into row groups.
        # Acknowledging this piece of code to be an extract from fastparquet.
        n_parts = (n_rows - 1) // max_row_group_size + 1
        row_group_size = min((n_rows - 1) // n_parts + 1, n_rows)
        starts = list(range(0, n_rows, row_group_size))
        if sharp_on:
            # Adjust bins so that they do not end in the middle of duplicate
            # values in `sharp_on` column.
            val_at_start = data.loc[:, sharp_on].iloc[starts]
            starts = unique(searchsorted(data[sharp_on].to_numpy(), val_at_start)).tolist()
    else:
        # If n_rows=0
        starts = [0]
    ends = starts[1:] + [None]
    if duplicates_on is not None:
        if duplicates_on == []:
            duplicates_on = data.columns
        for start, end in zip(starts, ends):
            yield data[start:end].drop_duplicates(duplicates_on, keep="last")
    else:
        for start, end in zip(starts, ends):
            yield data.iloc[start:end]


def to_midx(idx: Index, levels: List[str] = None) -> MultiIndex:
    """
    Expand a pandas index into a multi-index.

    Parameters
    ----------
    idx : Index
        Pandas index, with values being string representations of tuples, for
        instance, for one column, ``"('lev1','lev2')"``.
    levels : List[str], optional
        Names of levels to be used when creating the multi-index.
        If not provided, a generic naming is used, ``[l0, l1, l2, ...]``.
        If provided list is not long enough for the number of levels, it is
        completed using a generic naming, ``[..., l4, l5]``.

    Returns
    -------
    MultiIndex
        Pandas multi-index.

    Notes
    -----
    The accepted string representations of tuples is one typically obtained
    after a roundtrip from pandas dataframe with a column multi-index to vaex
    dataframe and back to pandas. The resulting column index is then a simple
    one, with string representations for tuples.

    If some column names have string representations of smaller tuples
    (resulting in fewer index levels), these column names are appended with
    empty strings '' as required to be of equal levels number than the longest
    column names.

    """
    idx_temp = []
    max_levels = 0
    for val in idx:
        try:
            tup = literal_eval(val)
            # Get max number of levels.
            max_levels = max(len(tup), max_levels)
            idx_temp.append(tup)
        except ValueError:
            # Keep value as string, enclosed in a tuple.
            idx_temp.append(tuple(val))
    # Generate names of levels if required.
    diff = 0
    if levels is None:
        levels = []
        len_lev = 0
        diff = max_levels
    elif (len_lev := len(levels)) < max_levels:
        diff = max_levels - len_lev
    if diff > 0:
        levels.extend([f"l{i}" for i in range(len_lev, max_levels)])
    # Equalize length of tuples.
    tuples = [(*t, *[""] * n) if (n := (max_levels - len(t))) else t for t in idx_temp]
    return MultiIndex.from_tuples(tuples, names=levels)


def check_cmidx(cmidx: MultiIndex):
    """
    Check if column multi-index complies with fastparquet requirements.

    Library fastparquet requires names for each level in a Multiindex.
    Also, column names have to be tuple of string.

    Parameters
    ----------
    cmidx : MultiIndex
        MultiIndex to check.

    """
    # Check level names.
    if None in cmidx.names:
        raise ValueError(
            "not possible to have level name set to None.",
        )  # If an item of the column name is not a string, turn it into a string.
    # Check column names.
    for level in cmidx.levels:
        for name in level:
            if not isinstance(name, str):
                raise TypeError(f"name {name} has to be of type 'string', not '{type(name)}'.")


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


def _indexes_of_overlapping_rrgs(
    new_data: DataFrame,
    recorded_pf: ParquetFile,
    ordered_on: Union[str, Tuple[str]],
    max_row_group_size: Union[int, str],
    drop_duplicates: bool,
    max_nirgs: Union[int, None],
) -> Tuple[int, int, bool]:
    """
    Identify overlaps between recorded row groups and new data.

    Overlaps may occur in the middle of recorded data.
    It may also occur at its tail in case there are incomplete row groups.

    Returns also a flag indicating if the data that will be written is within
    the set of complete row groups or after.

    Parameters
    ----------
    new_data : Dataframe
        New data.
    recorded_pf : ParquetFile
        ParquetFile of recorded data.
    ordered_on : Union[str, Tuple[str]]
        Name of the column with respect to which dataset is in ascending order.
        If column multi-index, name of the column is a tuple.
        It allows knowing 'where' to insert new data into existing data, i.e.
        completing or correcting past records (but it does not allow to remove
        prior data).
    max_row_group_size : Union[int, str]
        Define how to group data, either an ``int`` or a ``str``.
        If an ``int``, define the maximum number of rows allowed.
        If a ``str``, it has to be a pandas `freqstr`, to gather data by
        timestamp over a defined period.
    drop_duplicates : bool
        If duplicates have to be dropped or not.
    max_nirgs : Union[int, None]
        Max expected number of 'incomplete' row groups.
        To evaluate number of 'incomplete' row groups, only those at the end of
        an existing dataset are accounted for. 'Incomplete' row groups in the
        middle of 'complete' row groups are not accounted for (they can be
        created by insertion of new data 'in the middle' of existing data).
        If not set, default to ``None``.

          - ``None`` value induces no coalescing of row groups. If there is no
            drop of duplicates, new data is systematically appended.
          - A value of ``0`` or ``1`` means that new data should systematically
            be merged to the last existing one to 'complete' it (if it is not
            'complete' already).

    Returns
    -------
    int, int, bool
        'rrg_start_idx', 'rrg_end_idx', 'new_data_within_complete_rgs'
        'rrg_start_idx' and 'rrg_end_idx' are indices of recorded row groups
        overlapping with new data.
        If there is no overlapping row group, they are both set to ``None``.
        If there is overlapping with incomplete row groups (by definition of
        incomplete row groups, at the tail of the recorded data), then only
        'rrg_end_idx' is set to ``None``.
        'new_data_within_complete_rgs' is a flag indicating if once written, the
        row groups need to be sorted.

    Notes
    -----
    The function handles two cases for max_row_group_size:

      - Integer case: based on row count
      - String case: based on time periods

    """
    if not isinstance(max_row_group_size, (int, str)):
        raise TypeError("max_row_group_size must be int or str")
    if isinstance(max_row_group_size, int) and max_row_group_size <= 0:
        raise ValueError("max_row_group_size must be positive")
    # 1: assess existing overlaps.
    new_data_first = new_data.loc[:, ordered_on].iloc[0]
    new_data_last = new_data.loc[:, ordered_on].iloc[-1]
    ordered_on_recorded_max_vals = recorded_pf.statistics["max"][ordered_on]
    ordered_on_recorded_min_vals = recorded_pf.statistics["min"][ordered_on]
    # Recorded row group start and end indexes.
    rrg_start_idx, rrg_end_idx = None, None
    n_rrgs = len(recorded_pf.row_groups)
    compare_greater = ">=" if drop_duplicates else ">"
    overlapping_rrgs_idx = filter_row_groups(
        recorded_pf,
        [
            [
                (ordered_on, compare_greater, new_data_first),
                (ordered_on, "<=", new_data_last),
            ],
        ],
        as_idx=True,
    )
    if overlapping_rrgs_idx:
        rrg_start_idx = overlapping_rrgs_idx[0]
        if overlapping_rrgs_idx[-1] + 1 != n_rrgs:
            # For slicing, 'rrg_end_idx' is increased by 1.
            # If 'rrg_end_idx' is the index of the last row group, it keeps its
            # default 'None' value.
            rrg_end_idx = overlapping_rrgs_idx[-1] + 1
    # 2: if incomplete row groups are allowed, and incomplete row groups
    # location is connected to where the new data will be written (be it at the
    # tail of recorded data, or within it).
    full_tail_to_rewrite = False
    if max_nirgs is not None:
        new_data_connected_to_set_of_incomplete_rgs = False
        last_group_boundary_exceeded = False
        rrg_start_idx_tmp = n_rrgs - 1
        if isinstance(max_row_group_size, int):
            # Case 2.a: 'max_row_group_size' is an 'int'.
            # Number of incomplete row groups at end of recorded data.
            total_rows_in_irgs = 0
            min_row_group_size = int(max_row_group_size * MAX_ROW_GROUP_SIZE_SCALE_FACTOR)
            while (
                recorded_pf[rrg_start_idx_tmp].count() <= min_row_group_size
                and rrg_start_idx_tmp >= 0
            ):
                total_rows_in_irgs += recorded_pf[rrg_start_idx_tmp].count()
                rrg_start_idx_tmp -= 1
            if new_data_last > ordered_on_recorded_max_vals[rrg_start_idx_tmp]:
                # If new data is located in the set of incomplete row groups,
                # add length of new data to the number of rows of incomplete row
                # groups.
                # In the 'if' above, 'rrg_start_idx_tmp' is the index of the
                # last complete row groups.
                total_rows_in_irgs += len(new_data)
                new_data_connected_to_set_of_incomplete_rgs = True
            if total_rows_in_irgs >= max_row_group_size:
                last_group_boundary_exceeded = True
        else:
            # Case 2.b: 'max_row_group_size' is a str.
            # Get the 1st timestamp allowed in the last open period.
            # All row groups previous to this timestamp are considered complete.
            # TODO: if solved, select directly last row group in
            # recorded_pf.statistics["min"][ordered_on][-1] ?
            # https://github.com/dask/fastparquet/issues/938
            last_period_first_ts, next_period_first_ts = date_range(
                start=Timestamp(ordered_on_recorded_min_vals[-1]).floor(max_row_group_size),
                freq=max_row_group_size,
                periods=2,
            )
            while (
                ordered_on_recorded_min_vals[rrg_start_idx_tmp] >= last_period_first_ts
                and rrg_start_idx_tmp >= 0
            ):
                rrg_start_idx_tmp -= 1
            if new_data_last >= next_period_first_ts and (n_rrgs - rrg_start_idx_tmp > 2):
                # Coalesce if last recorded row group is incomplete (more than
                # 1) and the new data exceeds first timestamp of next period.
                last_group_boundary_exceeded = True
            if new_data_last >= last_period_first_ts:
                new_data_connected_to_set_of_incomplete_rgs = True
        # Confirm or not coalescing of incomplete row groups.
        # 'rrg_start_idx_tmp' is '-1' with respect to its definition.
        # This account for the new data that will make at least one row group
        # more.
        n_irgs = n_rrgs - rrg_start_idx_tmp
        rrg_start_idx_tmp += 1
        if new_data_connected_to_set_of_incomplete_rgs and (
            last_group_boundary_exceeded or n_irgs >= max_nirgs
        ):
            # Coalesce recorded data only it new data overlaps with it,
            # or if new data is appended at the tail.
            rrg_start_idx = (
                rrg_start_idx_tmp
                if rrg_start_idx is None
                else min(rrg_start_idx_tmp, rrg_start_idx)
            )
            # Force 'rrg_end_idx' to None.
            rrg_end_idx = None
            full_tail_to_rewrite = True
    return (
        rrg_start_idx,
        rrg_end_idx,
        not full_tail_to_rewrite and new_data_last < ordered_on_recorded_max_vals[-1],
    )


def write_ordered(
    dirpath: str,
    data: DataFrame,
    ordered_on: Union[str, Tuple[str]],
    max_row_group_size: Optional[Union[int, str]] = MAX_ROW_GROUP_SIZE,
    compression: str = COMPRESSION,
    cmidx_expand: bool = False,
    cmidx_levels: List[str] = None,
    duplicates_on: Union[str, List[str], List[Tuple[str]]] = None,
    max_nirgs: int = None,
    metadata: Dict[str, str] = None,
    md_key: Hashable = None,
):
    """
    Write data to disk at location specified by path.

    Parameters
    ----------
    dirpath : str
        Directory where writing pandas dataframe.
    data : DataFrame
        Data to write.
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

    max_row_group_size : Optional[Union[int, str]]
        Max row group size. If not set, default to ``6_345_000``, which for a
        dataframe with 6 columns of ``float64`` or ``int64`` results in a
        memory footprint (RAM) of about 290MB.
        It can be a pandas `freqstr` as well, to gather data by timestamp over a
        defined period.
    compression : str, default SNAPPY
        Algorithm to use for compressing data. This parameter is fastparquet
        specific. Please see fastparquet documentation for more information.
    cmidx_expand : bool, default False
        If `True`, expand column index into a column multi-index.
        This parameter is only used at creation of the dataset. Once column
        names are set, they cannot be modified by use of this parameter.
    cmidx_levels : List[str], optional
        Names of levels to be used when expanding column names into a
        multi-index. If not provided, levels are given names 'l1', 'l2', ...
    duplicates_on : Union[str, List[str], List[Tuple[str]]], optional
        Column names according which 'row duplicates' can be identified (i.e.
        rows sharing same values on these specific columns) so as to drop
        them. Duplicates are only identified in new data, and existing
        recorded row groups that overlap with new data.
        If duplicates are dropped, only last is kept.
        To identify row duplicates using all columns, empty list ``[]`` can be
        used instead of all columns names.
        If not set, default to ``None``, meaning no row is dropped.
    max_nirgs : int, optional
        Max expected number of 'incomplete' row groups.
        If 'max_row_group_size' is an ``int``, then a 'complete' row group
        is one which size is 'close to' ``max_row_group_size`` (>=90%).
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
    metadata : Dict[str, str], optional
        Key-value metadata to write, or update in dataset. Please see
        fastparquet for updating logic in case of `None` value being used.
    md_key: Hashable, optional
        Key to retrieve data in ``OUPS_METADATA`` dict, and write it as
        specific oups metadata in parquet file. If not provided, all data
        in ``OUPS_METADATA`` dict are retrieved to be written.

    Notes
    -----
    - When writing a dataframe with this function,

      - index of dataframe is not written to disk.
      - parquet file scheme is 'hive' (one row group per parquet file).

    - Coalescing incomplete row groups is triggered depending 2 conditions,
      either actual number of incomplete row groups is larger than ``max_nirgs``
      or number of rows for all incomplete row groups (at the end of the
      dataset) is enough to make a new complete row group (reaches
      ``max_row_group_size``). This latter assessment is however only triggered
      if ``max_nirgs`` is set. Otherwise, new data is simply appended, without
      prior check.
    - When ``duplicates_on`` is set, duplicate search is made row group to be
      written per row group to be written. A `row group to be written` is made
      from the merge between new data, and existing recorded row groups which
      overlap.
    - As per logic of previous comment, duplicates need to be gathered by
      row group to be identified, they need consequently to share the same
      `index`, defined by the value in ``ordered_on``. Extending this logic,
      ``ordered_on`` is added to ``duplicates_on`` if not already part of it.
    - For simple data appending, i.e. without need to drop duplicates, it is
      advised to keep ``ordered_on`` and ``duplicates_on`` parameters set to
      ``None`` as this parameter will trigger unnecessary evaluations.
    - Incomplete row groups are row groups:

      - either not reaching the maximum number of rows if 'max_row_group_size'
        is an ``int``,
      - or several row groups lying in tge same time period if
        'max_row_group_size' is a pandas freqstr.

    - When incorporating new data within recorded data, the algorithm checking
      for overlapping will not try to complete incomplete row groups that may
      lie within recorded data. If there is no intersection with existing data,
      new data is only added, without merging with existing incomplete row
      groups.
    - The algorithm will try to complete only incomplete row groups at the tail
      of recorded data.

    """
    if not data.empty:
        if cmidx_expand:
            data.columns = to_midx(data.columns, cmidx_levels)
        if ordered_on not in data.columns:
            # Check 'ordered_on' column is within input dataframe.
            raise ValueError(f"column '{ordered_on}' does not exist in input data.")
        if isinstance(max_row_group_size, str) and data.dtypes[ordered_on] != DTYPE_DATETIME64:
            raise TypeError(
                "if 'max_row_group_size' is a pandas freqstr, dtype"
                f" of column {ordered_on} has to be 'datetime64[ns]'.",
            )
    if os_path.isdir(dirpath) and any(file.endswith(".parquet") for file in os_listdir(dirpath)):
        # Case updating an existing dataset.
        if duplicates_on is not None:
            # Enforce 'ordered_on' in 'duplicates_on', as per logic of
            # duplicate identification restricted to the data overlap between new
            # data and existing data. This overlap being identified thanks to
            # 'ordered_on', it implies that duplicate rows can be identified being
            # so at the condition they share the same value in 'ordered_on' (among
            # other columns).
            if isinstance(duplicates_on, list):
                if duplicates_on and ordered_on not in duplicates_on:
                    # Case 'not an empty list', and 'ordered_on' not in.
                    duplicates_on.append(ordered_on)
            elif duplicates_on != ordered_on:
                # Case 'duplicates_on' is a single column name, but not
                # 'ordered_on'.
                duplicates_on = [duplicates_on, ordered_on]
        pf = ParquetFile(dirpath)
        rrg_start_idx, rrg_end_idx, sort_row_groups = _indexes_of_overlapping_rrgs(
            new_data=data,
            recorded_pf=pf,
            ordered_on=ordered_on,
            max_row_group_size=max_row_group_size,
            drop_duplicates=duplicates_on is not None,
            max_nirgs=max_nirgs,
        )
        if rrg_start_idx is None:
            # Case 'appending' (no overlap with recorded data identified).
            # 'coalesce' has possibly been requested but not needed, hence no row
            # groups removal in existing ones.
            iter_data = iter_dataframe(data, max_row_group_size)
            pf.write_row_groups(
                data=iter_data,
                row_group_offsets=None,
                sort_pnames=False,
                compression=compression,
                write_fmd=False,
            )
        else:
            # Case 'updating' (with existing row groups removal).
            # Read row groups that have impacted data.
            overlapping_pf = pf[rrg_start_idx:rrg_end_idx]
            # TODO: should iterate row groups instead of opening them all at
            # once.
            recorded = overlapping_pf.to_pandas()
            data = concat([recorded, data], ignore_index=True)
            data.sort_values(by=ordered_on, ignore_index=True, inplace=True)
            iter_data = iter_dataframe(
                data,
                max_row_group_size=max_row_group_size,
                sharp_on=ordered_on,
                duplicates_on=duplicates_on,
            )
            # Write.
            pf.write_row_groups(
                data=iter_data,
                row_group_offsets=None,
                sort_pnames=False,
                compression=compression,
                write_fmd=False,
            )
            # Remove row groups of data that is overlapping.
            pf.remove_row_groups(overlapping_pf.row_groups, write_fmd=False)
        if sort_row_groups:
            # /!\ TODO error here /!\
            # We want to check if data is in area of incomplete row groups
            # or if it is within the middle of data;
            # 'rrg_end_idx' can be None and the new data being in the middle
            # recorded data.
            # New data has been inserted in the middle of existing row groups.
            # Sorting row groups based on 'max' in 'ordered_on'.
            # TODO: why using 'ordered_on' index?
            # TODO: should sort if row groups are removed, or if df is inserted
            # in the middle of existing data.
            ordered_on_idx = pf.columns.index(ordered_on)
            pf.fmd.row_groups = sorted(
                pf.fmd.row_groups,
                key=lambda rg: statistics(rg.columns[ordered_on_idx])["max"],
            )
        # Rename partition files.
        pf._sort_part_names(write_fmd=False)
    else:
        # Case initiating a new dataset.
        iter_data = iter_dataframe(data, max_row_group_size)
        chunk = next(iter_data)
        # In case multi-index is used, check that it complies with fastparquet
        # limitations.
        if isinstance(chunk.columns, MultiIndex):
            check_cmidx(chunk.columns)
        fp_write(
            dirpath,
            chunk,
            row_group_offsets=max_row_group_size,
            compression=compression,
            file_scheme="hive",
            write_index=False,
            append=False,
        )
        # Re-open to write remaining chunks.
        pf = ParquetFile(dirpath)
        # Appending remaining chunks.
        pf.write_row_groups(
            data=iter_data,
            row_group_offsets=None,
            sort_pnames=False,
            compression=compression,
            write_fmd=False,
        )
    # Manage and write metadata.
    write_metadata(pf=pf, metadata=metadata, md_key=md_key)


# TODO
# - make ordered_on compulsory parameter, and remove all checks if it is provided
#   or not.
# - remove vaex (in all oups module, then from poetry)
# - implement row group split by date
# - in aggstream, use "standard" metadata recording, and remove the use of
#   OUPS_METADATA general dict.
# - remove in store ability to generate vaex dataframe, simplify oups Handle.


# Cas test:
# - test exception ordered_on column not datetime64, but max_row_group_size is str.
# - test specific append case
#    - max_row_group_size an int
#    - new data with length above "max_row_group_size"
#    - last row group is already above "max_row_groups_size"
#    - in this case, there should be a bug:
#       at the end, "rrg_start_idx" should be equal to n_rrgs
# - test case with pandas freqstr:
#    - new data in last recorded row group
#    - new data starting in a new period, with several incomplete row groups in
#      last recorded row group which should get merged
#    - new data starting in a new period, with a single row group in last
#      recorded row group: this one should not get updated. (check timestamp of writing)
