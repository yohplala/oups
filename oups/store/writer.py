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
from typing import Dict, Hashable, List, Tuple, Union

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from fastparquet.api import filter_row_groups
from fastparquet.api import statistics
from fastparquet.util import update_custom_metadata
from numpy import searchsorted as np_searchsorted
from numpy import unique as np_unique
from pandas import DataFrame as pDataFrame
from pandas import Index
from pandas import MultiIndex
from vaex import from_pandas
from vaex import open_many
from vaex.dataframe import DataFrame as vDataFrame


COMPRESSION = "SNAPPY"
MAX_ROW_GROUP_SIZE = 6_345_000
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
    data: Union[pDataFrame, vDataFrame],
    max_row_group_size: int = None,
    sharp_on: str = None,
    duplicates_on: Union[str, List[str]] = None,
):
    """
    Yield dataframe chunks.

    Parameters
    ----------
    data : Union[pDataFrame, vDataFrame]
        Data to split in row groups.
    max_row_group_size : int, optional
        Max size of row groups. It is a max as duplicates are dropped by row
        group to be written, hereby reducing the row group size (if
        ``duplicates_on`` parameter is set).
        If not set, default to ``6_345_000``.
    sharp_on : str, optional
        Name of column where to check that ends of bins (which split the data
        to be written) do not fall in the middle of duplicate values.
        This parameter is required when ``duplicates_on`` is used.
        This option requires a vaex dataframe.
        If not set, default to ``None``.
    duplicates_on : Union[str, List[str]], optional
        If set, drop duplicates based on list of column names, keeping last.
        Only duplicates within the same row group to be written are identified.
        This option requires a vaex dataframe.
        If not set, default to ``None``.
        If an empty list ``[]``, all columns are used to identify duplicates.

    Yields
    ------
    pDataFrame
        Chunk of data.

    Notes
    -----
    - Because duplicates are identified within a same row group, it is
      required to set ``sharp_on`` when using ``duplicates_on``, so that
      duplicates all fall in a same row group. This implies that duplicates
      have to share the same value in ``sharp_on`` column.

    """
    # TODO: implement 'group_as' (use of 'ordered_on' column) to replicate
    #  row group 'boundaries' between 2 different data sets.
    if max_row_group_size is None:
        max_row_group_size = MAX_ROW_GROUP_SIZE
    if isinstance(data, vDataFrame):
        # Drop any possible lazy indexing, to make the length of data equals
        # its filtered length.
        data = data.extract()
    elif sharp_on or isinstance(duplicates_on, list):
        raise TypeError("vaex dataframe required when using 'sharp_on' and/or 'duplicates_on'.")
    # TODO: if dropping duplicates over the full dataset with vaex, then remove
    # this exception and subsequent conditional cases. It is not necessary any
    # longer to relate 'sharp_on' and 'duplicates_on'.
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
    n_rows = len(data)
    if n_rows:
        # Define bins to split into row groups.
        # Acknowledging this piece of code to be an extract from fastparquet.
        n_parts = (n_rows - 1) // max_row_group_size + 1
        row_group_size = min((n_rows - 1) // n_parts + 1, n_rows)
        starts = list(range(0, n_rows, row_group_size))
    else:
        # If n_rows=0
        starts = [0]
    if sharp_on:
        # Adjust bins so that they do not end in the middle of duplicate values
        # in `sharp_on` column.
        # TODO: shorten this piece of code the day vaex accepts lists of
        # integers. Line of code currently taken from
        # vaex-core.vaex.dataframe.__getitem__ (L5230)
        val_at_start = [
            data.evaluate(sharp_on, idx, idx + 1, array_type="numpy")[0] for idx in starts
        ]
        # TODO: vaex searchsorted kind of broken. Re-try when solved?
        # https://github.com/vaexio/vaex/issues/1674
        # Doing with numpy.
        starts = np_unique(np_searchsorted(data[sharp_on].to_numpy(), val_at_start)).tolist()
    ends = starts[1:] + [None]
    if isinstance(data, vDataFrame):
        if duplicates_on is not None:
            if duplicates_on == []:
                columns = data.get_column_names()
                duplicates_on = columns
            for start, end in zip(starts, ends):
                # TODO: possible 'drop_duplicates' directly in vaex as per:
                # https://github.com/vaexio/vaex/pull/1623
                # check if answer from
                # https://github.com/vaexio/vaex/issues/1378
                # If dropping duplicates with vaex, drop before the chunking.
                # This removes then the constraints to have 'sharp_on' set and
                # and in 'duplicates_on'. Duplicates are then dropped over the
                # full dataset to be written.
                yield data[start:end].to_pandas_df().drop_duplicates(duplicates_on, keep="last")
        else:
            for start, end in zip(starts, ends):
                yield data[start:end].to_pandas_df()
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


def write(
    dirpath: str,
    data: Union[pDataFrame, vDataFrame],
    max_row_group_size: int = None,
    compression: str = COMPRESSION,
    cmidx_expand: bool = False,
    cmidx_levels: List[str] = None,
    ordered_on: Union[str, Tuple[str]] = None,
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
    data : Union[pDataFrame, vDataFrame]
        Data to write.
    max_row_group_size : int, optional
        Max row group size. If not set, default to ``6_345_000``, which for a
        dataframe with 6 columns of ``float64`` or ``int64`` results in a
        memory footprint (RAM) of about 290MB.
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
    ordered_on : Union[str, Tuple[str]] optional
        Name of the column with respect to which dataset is in ascending order.
        If column multi-index, name of the column is a tuple.
        This parameter is optional, and required so that data overlaps between
        new data and existing recorded row groups can be identified.
        This parameter is compulsory when ``duplicates_on`` is used.
        If not set, default to ``None``.
        It has two effects:

          - it allows knowing 'where' to insert new data into existing data,
            i.e. completing or correcting past records (but it does not allow
            to remove prior data).
          - it ensures that two consecutive row groups do not have duplicate
            values in column defined by ``ordered_on`` (only in row groups to
            be written). This implies that all possible duplicates in
            ``ordered_on`` column will lie in the same row group.

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
        Max expected number of 'incomplete' row groups. A 'complete' row group
        is one which size is 'close to' ``max_row_group_size`` (>=90%).
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
      overlap. For this reason ``ordered_on`` parameter is compulsory when
      using ``duplicates_on``, so as to be able to position new data with
      respect to existing row groups and also to cluster this data (new and
      overlapping recorded) into row group which have distinct values in
      ``ordered_on`` column. If 2 rows are duplicates according values in
      indicated columns but are not in the same row group, first duplicates
      will not be dropped.
    - As per logic of previous comment, duplicates need to be gathered by
      row group to be identified, they need consequently to share the same
      `index`, defined by the value in ``ordered_on``. Extending this logic,
      ``ordered_on`` is added to ``duplicates_on`` if not already part of it.
    - For simple data appending, i.e. without need to check where to insert
      data and without need to drop duplicates, it is advised to keep
      ``ordered_on`` and ``duplicates_on`` parameters set to ``None`` as these
      parameters will trigger unnecessary evaluations.

    """
    if ordered_on is not None:
        if isinstance(ordered_on, tuple):
            raise TypeError(f"tuple for {ordered_on} not yet supported.")
        # Check 'ordered_on' column is within input dataframe.
        if isinstance(data, pDataFrame):
            # pandas case
            all_cols = data.columns
        else:
            # vaex case
            all_cols = data.get_column_names()
        if ordered_on not in all_cols:
            raise ValueError(f"column '{ordered_on}' does not exist in input data.")
    if os_path.isdir(dirpath) and any(file.endswith(".parquet") for file in os_listdir(dirpath)):
        # Case updating an existing dataset.
        # Identify overlaps in row groups between new data and recorded data.
        # Recorded row group start and end indexes.
        rrg_start_idx, rrg_end_idx = None, None
        pf = ParquetFile(dirpath)
        n_rrgs = len(pf.row_groups)
        if duplicates_on is not None:
            if not ordered_on:
                raise ValueError(
                    "duplicates are looked for over the overlap between new data and existing data. "
                    "This overlap being identified thanks to 'ordered_on', "
                    "it is compulsory to set 'ordered_on' while setting 'duplicates_on'.",
                )
            # Enforce 'ordered_on' in 'duplicates_on', as per logic of
            # duplicate identification restricted to the data overlap between new
            # data and existing data. This overlap being identified thanks to
            # 'ordered_on', it implies that duplicate rows can be identified being
            # so at the condition they share the same value in 'ordered_on' (among
            # other columns).
            elif isinstance(duplicates_on, list):
                if duplicates_on and ordered_on not in duplicates_on:
                    # Case 'not an empty list', and 'ordered_on' not in.
                    duplicates_on.append(ordered_on)
            elif duplicates_on != ordered_on:
                # Case 'duplicates_on' is a single column name, but not
                # 'ordered_on'.
                duplicates_on = [duplicates_on, ordered_on]
        if ordered_on is not None:
            # Get 'rrg_start_idx' & 'rrg_end_idx'.
            if isinstance(data, pDataFrame):
                # Case 'pandas'.
                start = data[ordered_on].iloc[0]
                end = data[ordered_on].iloc[-1]
            else:
                # Case 'vaex'.
                start = data[ordered_on][:0].to_numpy()[0]
                end = data[ordered_on][-1:].to_numpy()[0]
            rrgs_idx = filter_row_groups(
                pf,
                [[(ordered_on, ">=", start), (ordered_on, "<=", end)]],
                as_idx=True,
            )
            if rrgs_idx:
                if len(rrgs_idx) == 1:
                    rrg_start_idx = rrgs_idx[0]
                else:
                    rrg_start_idx = rrgs_idx[0]
                    # For slicing, 'rrg_end_idx' is increased by 1.
                    rrg_end_idx = rrgs_idx[-1] + 1
                    if rrg_end_idx == n_rrgs:
                        rrg_end_idx = None
        if max_nirgs is not None:
            # Number of incomplete row groups at end of recorded data.
            # Initialize number of rows with number to be written.
            total_rows_in_irgs = len(data)
            rrg_start_idx_tmp = n_rrgs - 1
            min_row_group_size = int(max_row_group_size * 0.9)
            while pf[rrg_start_idx_tmp].count() <= min_row_group_size and rrg_start_idx_tmp >= 0:
                total_rows_in_irgs += pf[rrg_start_idx_tmp].count()
                rrg_start_idx_tmp -= 1
            rrg_start_idx_tmp += 1
            # Confirm or not coalescing of incomplete row groups.
            n_irgs = n_rrgs - rrg_start_idx_tmp
            if total_rows_in_irgs >= max_row_group_size or n_irgs >= max_nirgs:
                if rrg_start_idx and (
                    not rrg_end_idx or (rrg_end_idx and rrg_start_idx_tmp < rrg_end_idx)
                ):
                    # 1st case checked: case 'ordered_on' is used with potential
                    # insertion of new data in existing one, in row groups not at
                    # the tail, in which case coalescing of end data would not be
                    # performed.
                    rrg_start_idx = min(rrg_start_idx_tmp, rrg_start_idx)
                elif not rrg_end_idx:
                    # 2nd case checked: case 'ordered_on' is not used. New data is
                    # necessarily appended in this case.
                    rrg_start_idx = rrg_start_idx_tmp
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
            # Read row groups that have impacted data as a vaex dataframe.
            overlapping_rgs = pf[rrg_start_idx:rrg_end_idx].row_groups
            files = [pf.row_group_filename(rg) for rg in overlapping_rgs]
            recorded = open_many(files)
            if isinstance(data, pDataFrame):
                # Convert to vaex.
                data = from_pandas(data)
            # For concatenation of numpy 'timedelta64' to arrow 'time64', check
            # https://github.com/vaexio/vaex/issues/2024
            data = recorded.concat(data)
            if ordered_on:
                data = data.sort(by=ordered_on)
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
            pf.remove_row_groups(overlapping_rgs, write_fmd=False)
            if rrg_end_idx is not None:
                # New data has been inserted in the middle of existing row groups.
                # Sorting row groups based on 'max' in 'ordered_on'.
                ordered_on_idx = pf.columns.index(ordered_on)
                pf.fmd.row_groups = sorted(
                    pf.fmd.row_groups,
                    key=lambda rg: statistics(rg.columns[ordered_on_idx])["max"],
                )
            # Rename partition files, and write fmd.
            pf._sort_part_names(write_fmd=False)
    else:
        # Case initiating a new dataset.
        iter_data = iter_dataframe(data, max_row_group_size)
        chunk = next(iter_data)
        if cmidx_expand:
            chunk.columns = to_midx(chunk.columns, cmidx_levels)
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
