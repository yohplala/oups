#!/usr/bin/env python3
"""
Created on Wed Dec  6 22:30:00 2021.

@author: yoh
"""
from ast import literal_eval
from typing import List, Tuple, Union

from fastparquet import ParquetFile
from fastparquet import write as fp_write
from fastparquet.api import filter_row_groups
from numpy import searchsorted as np_searchsorted
from pandas import DataFrame as pDataFrame
from pandas import Index
from pandas import MultiIndex
from vaex import agg as vx_agg
from vaex import from_pandas
from vaex import open_many
from vaex import vrange
from vaex.dataframe import DataFrame as vDataFrame


COMPRESSION = "SNAPPY"
MAX_ROW_GROUP_SIZE = 6_345_000


def iter_dataframe(
    data: Union[pDataFrame, vDataFrame],
    max_row_group_size: int = None,
    ordered_on: str = None,
    duplicates_on: Union[str, List[str]] = None,
):
    """Yield dataframe chunks.

    Parameters
    ----------
    data : Union[pDataFrame, vDataFrame]
        Data to split in row groups.
    max_row_group_size : int, optional
        Max size of row groups. It is a max as duplicates are dropped by row
        group to be written, hereby reducing the row group size (if
        ``duplicates_on`` parameter is set).
        If not set, default to ``6_345_000``.
    ordered_on : str, optional
        If set, ensures that end value of a row group in column ``ordered_on``
        is not the same than in next row group.
        This parameter is recommended when ``duplicates_on`` is used.
        This option requires a vaex dataframe.
        If not set, default to ``None``.
    duplicates_on : Union[str, List[str]], optional
        If set, drop duplicates based on list of column names, keeping last.
        Only duplicates within the same row group to be written are identified.
        This option requires a vaex dataframe.
        If not set, default to ``None``.
        If an empty list ``[]``, all columns are used to identify duplicates

    Yields
    ------
    pDataFrame
        Chunk of data.

    Notes
    -----
    - Because duplicates are identified within a same row group, it is
      recommended to set ``ordered_on`` when using ``duplicates_on``, so that
      duplicates all fall in a same row group. This assumes that duplicates
      share the same `index` value (i.e. same value in ``ordered_on`` column).
    """
    # TODO: implement 'replicate_groups' (use of 'ordered_on' column).
    if max_row_group_size is None:
        max_row_group_size = MAX_ROW_GROUP_SIZE
    if isinstance(data, vDataFrame):
        # Drop any possible the lazy indexing, to make the length of data
        # equals its filtered length
        data = data.extract()
    elif ordered_on or duplicates_on:
        raise TypeError("vaex dataframe required when using `ordered_on` and/or `duplicates_on`.")
    # Define bins to split into row groups.
    # Acknowledging this piece of code to be an extract from fastparquet.
    n_rows = len(data)
    n_parts = (n_rows - 1) // max_row_group_size + 1
    row_group_size = min((n_rows - 1) // n_parts + 1, n_rows)
    starts = list(range(0, n_rows, row_group_size))
    if ordered_on:
        # Adjust bins so that they do not end in the middle of duplicate values
        # in `ordered_on` column.
        # TODO: shorten this piece of code the day vaex accepts lists of
        # integers. Line of code currently taken from
        # vaex-core.vaex.dataframe.__getitem__ (L5230)
        val_at_start = [
            data.evaluate(ordered_on, idx, idx + 1, array_type="python")[0] for idx in starts
        ]
        # TODO: vaex searchsorted kind of broken. Re-try when solved?
        # https://github.com/vaexio/vaex/issues/1674
        # Doing with numpy.
        starts = np_searchsorted(data[ordered_on].to_numpy(), val_at_start).tolist()
    ends = starts[1:] + [None]
    if isinstance(data, vDataFrame):
        if duplicates_on is not None:
            if duplicates_on == []:
                duplicates_on = data.get_column_names()
            # TODO: 'drop_duplicates' with vaex, keeping 'last'.
            # simplify when possible. Used workarounds:
            # - inverse row index and keep actually first value (which was
            # last value before), as per
            # https://github.com/vaexio/vaex/issues/1378
            data["__row_index"] = vrange(0, len(data))
            agg_last = vx_agg.first(ordered_on, "-__row_index")
            for start, end in zip(starts, ends):
                # 'drop_duplicates' as per:
                # https://github.com/vaexio/vaex/pull/1623
                yield data[start:end].groupby(duplicates_on, agg={"__last": agg_last}).drop(
                    ["__last", "__row_index"]
                ).to_pandas_df()
        else:
            for start, end in zip(starts, ends):
                yield data[start:end].to_pandas_df()
    else:
        for start, end in zip(starts, ends):
            yield data.iloc[start:end]


def to_midx(idx: Index, levels: List[str] = None) -> MultiIndex:
    """Expand a pandas index into a multi-index.

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


def write(
    dirpath: str,
    data: Union[pDataFrame, vDataFrame],
    max_row_group_size: int = None,
    compression: str = COMPRESSION,
    cmidx_expand: bool = False,
    cmidx_levels: List[str] = None,
    ordered_on: Union[str, Tuple[str]] = None,
    duplicates_on: Union[str, List[str]] = None,
    irgs_max: int = None,
):
    """Write data to disk at location specified by path.

    Parameters
    ----------
    dirpath : str
        Directory where writing pandas dataframe.
    data : Union[pDataFrame, vDataFrame]
        Data to write.
    max_row_group_size : int, optional
        Max row group size. If not set, default to ``6_345_000``, which for a
        dataframe with 6 columns of ``float64``/``int64`` results in a memory
        footprint (RAM) of about 290MB.
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
    ordered_on : str, optional
        Name of the column with respect to which dataset is in ascending order.
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

    duplicates_on : Union[str, List[str]], optional
        Column names according which 'row duplicates' can be identified (i.e.
        rows sharing same values on these specific columns) so as to drop
        them. Duplicates are only identified in new data, and existing
        recorded row groups that overlap with new data.
        If duplicates are dropped, only last is kept.
        To identify row duplicates using all columns, empty list ``[]`` can be
        used instead of all columns names.
        If not set, default to ``None``, meaning no row is dropped.
    irgs_max : int, optional
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

    Notes
    -----
    - When writing a dataframe with this function,

      - index of dataframe is not written to disk.
      - parquet file scheme is 'hive' (one row group per parquet file).

    - Coalescing incomplete row groups is triggered depending 2 conditions,
      either actual number of incomplete row groups is larger than ``irgs_max``
      or number of rows for all incomplete row groups (at the end of the
      dataset) is enough to make a new complete row group (reaches
      ``max_row_group_size``). This latter assessment is however only triggered
      if ``irgs_max`` is set. Otherwise, new data is simply appended, without
      prior check.
    - When ``duplicates_on`` is set, duplicate search is made new row group per
      new row group and with existing recorded row groups which overlap. For
      this reason ``ordered_on`` parameter is compulsory when using
      ``duplicates_on``, so as to be able to position new data with respect to
      existing row groups and also to cluster this data (new and overlapping
      recorded) into row group which have distinct values in ``ordered_on``
      column. If 2 rows are duplicates according values in indicated
      columns but are not in the same row group, first duplicates will not be
      dropped.
    - For simple data appending, i.e. without need to check where to insert
      data and without need to drop duplicates, it is advised to keep
      ``ordered_on`` and ``duplicates_on`` parameters set to ``None`` as these
      parameters will trigger unnecessary evaluations.
    """
    try:
        pf = ParquetFile(dirpath)
    except (FileNotFoundError, ValueError):
        # First time writing.
        iter_data = iter_dataframe(data, max_row_group_size)
        chunk = next(iter_data)
        if cmidx_expand:
            chunk.columns = to_midx(chunk.columns, cmidx_levels)
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
        # Appending
        # TODO: remove 'sort_pnames=False' when set to False by default in
        # fastparquet.
        pf.write_row_groups(
            data=iter_data, row_group_offsets=None, sort_pnames=False, compression=compression
        )
        return
    # Not first time writing.
    # Identify overlaps in row groups between new data and recorded data.
    # Recorded row group start and end indexes.
    rrg_start_idx, rrg_end_idx = None, None
    num_rrgs = len(pf.row_groups)
    if duplicates_on is not None:
        if not ordered_on:
            raise ValueError(
                "not possible to set ``duplicates_on`` without setting ``ordered_on``."
            )
    if ordered_on is not None:
        # Get 'rrg_start_idx' & 'rrg_end_idx'.
        if isinstance(data, pDataFrame):
            # Case 'pandas'.
            start = data.loc[ordered_on].iloc[0]
            end = data.loc[ordered_on].iloc[-1]
        else:
            # Case 'vaex'.
            start = data[ordered_on][:0].to_numpy()[0]
            end = data[ordered_on][-1:].to_numpy()[0]
        rrgs_idx = filter_row_groups(
            pf, [(ordered_on, ">=", start), (ordered_on, "<=", end)], as_idx=True
        )
        if rrgs_idx:
            rrg_start_idx, rrg_end_idx = rrgs_idx[0], rrgs_idx[-1]
        # R
        print(f"rrg_start_idx: {rrg_start_idx} - rrg_end_idx: {rrg_end_idx}")
    if irgs_max is not None:
        # Number of incomplete row groups at end of recorded data.
        total_rows_in_irgs = 0
        rrg_start_idx_tmp = num_rrgs - 1
        min_row_group_size = int(max_row_group_size * 0.9)
        while pf[rrg_start_idx_tmp].count() < min_row_group_size and rrg_start_idx_tmp >= 0:
            total_rows_in_irgs += pf[rrg_start_idx_tmp].count()
            rrg_start_idx_tmp -= 1
        rrg_start_idx_tmp += 1
        # Confirm or not coalescing of incomplete row groups.
        num_irgs = num_rrgs - rrg_start_idx_tmp
        # R
        print(f"num_irgs: {num_irgs}")
        if total_rows_in_irgs >= max_row_group_size or num_irgs > irgs_max:
            if rrg_end_idx and rrg_start_idx_tmp <= rrg_end_idx:
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
        # Case 'appending'.
        # 'coalesce' has possibly been requested but not needed, hence no row
        # groups removal in existing ones.
        iter_data = iter_dataframe(data, max_row_group_size)
        pf.write_row_groups(
            data=iter_data,
            row_group_offsets=None,
            sort_pnames=False,
            compression=compression,
            write_fmd=True,
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
        data = recorded.concat(data)
        if ordered_on:
            data = data.sort(by=ordered_on)
        iter_data = iter_dataframe(data, max_row_group_size, ordered_on, duplicates_on)
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
        # Rename partition files, and write fmd.
        pf._sort_part_names(write_fmd=True)
