#!/usr/bin/env python3
"""
Created on Wed Mar  9 21:30:00 2022.

@author: yoh
"""
from copy import copy
from dataclasses import dataclass
from multiprocessing import cpu_count
from os import path as os_path
from typing import Callable, Dict, Generator, List, Tuple, Union

import numpy as np
from fastparquet import ParquetFile
from joblib import Parallel
from joblib import delayed
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat
from pandas import date_range
from pandas import read_json
from vaex import from_arrays
from vaex import from_pandas
from vaex import vrange
from vaex.agg import first as vfirst
from vaex.agg import last as vlast
from vaex.agg import max as vmax
from vaex.agg import min as vmin
from vaex.agg import sum as vsum
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet
from oups.router import ParquetHandle
from oups.utils import tcut
from oups.writer import MAX_ROW_GROUP_SIZE
from oups.writer import OUPS_METADATA
from oups.writer import OUPS_METADATA_KEY
from oups.writer import write


VDATAFRAME_ROW_GROUP_SIZE = 6_345_000
# Aggregation functions.
FIRST = "first"
LAST = "last"
MIN = "min"
MAX = "max"
SUM = "sum"
ACCEPTED_AGG_FUNC = {FIRST, LAST, MIN, MAX, SUM}
VAEX_AGG = {FIRST: vfirst, LAST: vlast, MIN: vmin, MAX: vmax, SUM: vsum}
VAEX_SORT = "vaex_sort"
# List of keys to metadata of aggregation results.
MD_KEY_CHAINAGG = "chainagg"
MD_KEY_LAST_SEED_INDEX = "last_seed_index"
MD_KEY_BINNING_BUFFER = "binning_buffer"
MD_KEY_LAST_AGGREGATION_ROW = "last_aggregation_row"
MD_KEY_POST_BUFFER = "post_buffer"
# Config. for pandas dataframe serialization / de-serialization.
PANDAS_SERIALIZE = {"orient": "table", "date_unit": "ns", "double_precision": 15}
PANDAS_DESERIALIZE = {"orient": "table", "date_unit": "ns", "precise_float": True}
# Misc.
REDUCTION_BIN_COL_PREFIX = "bin_"
VAEX = "vaex"


def _is_chainagg_result(handle: ParquetHandle) -> bool:
    """Check if input handle is that of a dataset produced by streamaag.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file.

    Returns
    -------
    bool
        `True` if parquet file contains metadata as produced by
        ``oups.chainagg``, which confirms this dataset has been produced with
        this latter function.
    """
    # As oups specific metadata is a string produced by json library, the last
    # 'in' condition is checking if the set of characters defined by
    # 'MD_KEY_CHAINAGG' is in a string.
    pf = handle.pf
    return (
        OUPS_METADATA_KEY in pf.key_value_metadata
        and MD_KEY_CHAINAGG in pf.key_value_metadata[OUPS_METADATA_KEY]
    )


def _get_chainagg_md(handle: ParquetHandle) -> tuple:
    """Retrieve and deserialize chainagg metadata from previous aggregation.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file from which extracting metadata.

    Returns
    -------
    tuple
        Data recorded from previous aggregation to allow pursuing it with
        new seed data. 3 variables are returned.

          - ``last_seed_index``, last value in 'ordered-on' column in seed data.
          - ``last_agg_row``, last row from previously aggregated results.
          - ``binning_buffer``, a dict to be forwarded to ``by`` if a callable.
          - ``post_buffer``, a dict to be forwarded to ``post`` callable.

    """
    # Retrieve corresponding metadata to re-start aggregations.
    # Get seed index value to start new aggregation.
    # It is a value to be included when filtering seed data.
    # Trim accordingly head of seed data in this case.
    chainagg_md = handle._oups_metadata[MD_KEY_CHAINAGG]
    # De-serialize 'last_seed_index'.
    last_seed_index = chainagg_md[MD_KEY_LAST_SEED_INDEX]
    last_seed_index = read_json(last_seed_index, **PANDAS_DESERIALIZE).iloc[0, 0]
    # 'last_agg_row' for stitching with new aggregation results.
    last_agg_row = read_json(chainagg_md[MD_KEY_LAST_AGGREGATION_ROW], **PANDAS_DESERIALIZE)
    # Metadata related to binning process from past binnings on prior data.
    # It is used in case 'by' is a callable.
    if chainagg_md[MD_KEY_BINNING_BUFFER]:
        binning_buffer = (
            read_json(chainagg_md[MD_KEY_BINNING_BUFFER], **PANDAS_DESERIALIZE).iloc[0].to_dict()
        )
    else:
        binning_buffer = {}
    # Metadata related to post-processing of prior aggregation results, to be
    # used by 'post'.
    if chainagg_md[MD_KEY_POST_BUFFER]:
        post_buffer = (
            read_json(chainagg_md[MD_KEY_POST_BUFFER], **PANDAS_DESERIALIZE).iloc[0].to_dict()
        )
    else:
        post_buffer = {}
    return last_seed_index, last_agg_row, binning_buffer, post_buffer


def _set_chainagg_md(
    key: str,
    last_seed_index,
    last_agg_row: pDataFrame,
    binning_buffer: dict = None,
    post_buffer: dict = None,
):
    """Serialize and record chainagg metadata from last aggregation and post.

    Parameters
    ----------
    key : str
        Key of data in oups store for which metadata is to be written.
    last_seed_index : default None
        Last index in seed data. Can be numeric type, timestamp...
    last_agg_row : pDataFrame
        Last row from last aggregation results, required for stitching with
        aggregation results from new seed data.
    binning_buffer : dict
        User-chosen values from previous binning process, that can be required
        when restarting the binning process with new seed data.
    post_buffer : dict
        Last values from post-processing, that can be required when restarting
        post-processing of new aggregation results.
    """
    # Setup metadata for a future 'chainagg' execution.
    # Store a json serialized pandas series, to keep track of 'whatever the
    # object' the index is.
    last_seed_index = pDataFrame({MD_KEY_LAST_SEED_INDEX: [last_seed_index]}).to_json(
        **PANDAS_SERIALIZE
    )
    last_agg_row = last_agg_row.to_json(**PANDAS_SERIALIZE)
    if binning_buffer:
        binning_buffer = pDataFrame(binning_buffer, index=[0]).to_json(**PANDAS_SERIALIZE)
    if post_buffer:
        post_buffer = pDataFrame(post_buffer, index=[0]).to_json(**PANDAS_SERIALIZE)
    # Set oups metadata.
    metadata = {
        MD_KEY_CHAINAGG: {
            MD_KEY_LAST_SEED_INDEX: last_seed_index,
            MD_KEY_BINNING_BUFFER: binning_buffer,
            MD_KEY_LAST_AGGREGATION_ROW: last_agg_row,
            MD_KEY_POST_BUFFER: post_buffer,
        }
    }
    OUPS_METADATA[key] = metadata


def _post_n_write_agg_chunks(
    chunks: List[pDataFrame],
    dirpath: str,
    key: str,
    write_config: dict,
    index_name: Union[str, None] = None,
    post: Callable = None,
    isfbn: bool = None,
    post_buffer: dict = None,
    other_metadata: tuple = None,
):
    """Write list of aggregation row groups with optional post, then reset it.

    Parameters
    ----------
    chunks : List[pandas.DataFrame]
        List of chunks resulting from aggregation (pandas dataframes).
    dirpath : str
        Path to which recording aggregation results.
    key : str
        Key for retrieving corresponding metadata.
    index_name : str, default None
        If a string, name index of dataframe resulting from aggregation with
        this value.
    write_config : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Compulsory parameter defining at least `ordered_on`
        and `duplicates_on` columns.
    post : Callable, default None
        User-defined function accepting 3 parameters.

          - First, the pandas dataframe resulting from the aggregations defined
            by ``agg`` parameter, with first row already corrected with last
            row of previous streamed aggregation.
          - Second, a boolean which indicates if first row of aggregation
            result is a new row, or is the 'same' that last row of aggregation
            result. If 'same', all values may not be the same, but the
            aggregation bin is the same.
          - Third, a dict to be used as data buffer, that can be necessary for
            some user-defined post-processing requiring data assessed in
            previous post-processing iteration.

        It has then to return a pandas dataframe that will be recorded.
        This optional post-processing is intended for use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), or dataframe formatting before results are finally recorded.
    isfbn : boolean, default None
        Boolean indicating if first row of aggregation result is that of a new
        bin, or is the same bin started from yet earlier iteration of
        aggregation. If 'same', aggregation results (values in aggregation
        columns) may have been updated with last aggregation iteration, but the
        aggregation bin (group key) is the same.
    post_buffer : dict, default None
        Buffer to keep track of data that can be processed during previous
        iterations. This pointer should not be re-initialized in 'post' or
        data from previous iterations will be lost.
        This dict has to contain data that can be serialized, as data is then
        kept in parquet file metadata.
    other_metadata : tuple, default None
        Metadata to be recorded in parquet file. Data has to be serializable.
        If `None`, no metadata is recorded.
        If some metadata is defined, ``post_buffer`` also gets its way in
        metadata.
    """
    # Keep last row as there might be not further iteration.
    if len(chunks) > 1:
        agg_res = pconcat(chunks)
    else:
        agg_res = chunks[0]
    if index_name:
        # In case 'by' is a callable, index may have no name, but user may have
        # defined one with 'bin_on' parameter.
        agg_res.index.name = index_name
    # Keep group keys as a column before post-processing.
    agg_res.reset_index(inplace=True)
    # Reset (in place) buffer.
    chunks.clear()
    if post:
        # Post processing if any.
        # 'post_buffer' has to be modified in-place.
        agg_res = post(agg_res, isfbn, post_buffer)
    if other_metadata:
        # Set oups metadata.
        _set_chainagg_md(key, *other_metadata, post_buffer)
    # Record data.
    write(dirpath=dirpath, data=agg_res, md_key=key, **write_config)


def _setup_binning(
    ordered_on: str,
    reduction: bool,
    key_idx: int,
    by: Union[Callable, Grouper, None] = None,
    bin_on: Union[str, Tuple[str, str]] = None,
    reduction_bin_col: Union[str, None] = None,
):
    """Specifically operate binning setup.

    Parameters
    ----------
    ordered_on : str
        Name of column according which seed dataset is ordered.
    reduction : bool
        Flag indicating if reduction step will be performed on seed chunk or
        not.
    key_idx : int
        Index of key.
    by : Callable, pd.Grouper or str, default None
        Defines the binning logic.
    bin_on : str, or Tuple[str, str], default None
        Name of the column from which deriving bins.
    reduction_bin_col : str
        Bin column name to be used between the common reduction step and the
        individual groupby step for each key.

    Returns
    -------
    by, cols_to_by, bins, bin_out_col

          - ``by`` Callable, or pandas Grouper, or `None`. It is used to define
            the bins for the reduction step, or if no reduction step, it still
            defines a bin array when is a Callable for the individual binning.
          - ``cols_to_by`` str, column name required for ``by`` is ``by`` is a
            Callable.
          - ``reduction_bin_col``, str. Name of column to be used for this key
            for binning at reduction step.
          -  ``bins`` str or pandas Grouper. It is the parameter used to
            achieve the individual binning for each key, that there is a
            reduction step or not.
          - ``bin_out_col`` str. Bin column is renamed to this value when
            writing results. If no value is defined, and ``bin_on`` itself is
            ``None``, then the default value it takes is 'index' (set by
            default by pandas when resetting an index without name).

    """
    # Case 1 / Bin according an existing column.
    # 'reduction' is `False`
    # Before setup   'by'                : None
    #                'bin_on'            : str (existing column name)
    #                'reduction_bin_col' : None
    # After setup    'by'                : None
    #                'cols_to_by'        : None
    #                'reduction_bin_col' : None
    #                'bins'              : str (existing column name)
    # 'reduction' is `True`
    # Before setup   'by'                : None
    #                'bin_on'            : str (existing column name)
    #                'reduction_bin_col' : 'key_xx'
    # After setup    'by'                : None
    #                'cols_to_by'        : None
    #                'reduction_bin_col' : str (existing column name)
    #                'bins'              : str (existing column name)
    #
    # Case 2 / 'by' is a Callable.
    # 'reduction' is `False`
    # Before setup   'by'                : callable
    #                'bin_on'            : None or str (column name)
    #                'reduction_bin_col' : None
    # After setup    'by'                : callable
    #                'cols_to_by'        : None or str ('bin_on')
    #                'reduction_bin_col' : None
    #                'bins'              : None (array-like defined on-the-fly)
    # 'reduction' is `True`
    # Before setup   'by'                : callable
    #                'bin_on'            : None or str ('bin_on')
    #                'reduction_bin_col' : 'key_xx'
    # After setup    'by'                : callable
    #                'cols_to_by'        : str (existing column name)
    #                'reduction_bin_col' : 'key_xx'
    #                'bins'              : 'key_xx'
    #
    # Case 3 / 'by' is a pd.Grouper.
    # 'reduction' is `False`
    # Before setup   'by'                : pd.Grouper
    #                'bin_on'            : None or str (column name)
    #                'reduction_bin_col' : None
    # After setup    'by'                : None
    #                'cols_to_by'        : None
    #                'bin_on'            : str (existing column name)
    #                'reduction_bin_col' : None
    #                'bins'              : pd.Grouper
    # 'reduction' is `True`
    # Before setup   'by'                : pd.Grouper
    #                'bin_on'            : None or str (column name)
    #                'reduction_bin_col' : 'key_xx'
    # After setup    'by'                : pd.Grouper
    #                'cols_to_by'        : None
    #                'bin_on'            : str (existing column name)
    #                'reduction_bin_col' : 'key_xx'
    #                'bins'              : pd.Grouper ('key' modified)
    if bin_on:
        if isinstance(bin_on, tuple):
            # 'bin_out_col' is name of column containing group keys in 'agg_res'.
            bin_on, bin_out_col = bin_on
        else:
            bin_out_col = bin_on
    else:
        bin_out_col = None
    if reduction:
        if by is None:
            # Case bin on existing column.
            reduction_bin_col = bin_on
        else:
            reduction_bin_col = f"{REDUCTION_BIN_COL_PREFIX}{key_idx}"
    else:
        reduction_bin_col = None
    if isinstance(by, Grouper):
        # Case pandas Grouper.
        # https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
        cols_to_by = None
        if bin_on and not by.key:
            by.key = bin_on
        elif bin_on and by.key and bin_on != by.key:
            raise ValueError(
                "two different columns are defined for achieving binning,"
                " both by `bin_on` and `by` parameters, pointing to"
                f" '{bin_on}' and '{by.key}' columns respectively."
            )
        elif not bin_on and by.key:
            bin_on = by.key
            bin_out_col = bin_on
        elif not (bin_on or by.key):
            # Nor 'by.key', nor 'bin_on' defined.
            raise ValueError(f"no column name defined to bin with provided pandas grouper '{by}'.")
        if not bin_out_col:
            bin_out_col = by.key
        if reduction:
            # When 'by' is a Grouper, 'bins' is a modified Grouper, with a
            # different key value.
            bins = copy(by)
            bins.key = reduction_bin_col
        else:
            bins = by
            by = None
    elif callable(by):
        # Case Callable.
        if bin_on:
            cols_to_by = [ordered_on, bin_on]
        else:
            cols_to_by = ordered_on
        # If no reduction, 'bins' is updated on the fly within the loop
        # with bin label for each row.
        # Otherwise 'bins' is column name that will get the bin labels.
        bins = reduction_bin_col if reduction else None
    elif by is None and bin_on:
        # Case 'bin_on' is a column name, and 'by' is undefined.
        cols_to_by = None
        if reduction:
            # 'bins' is column name that will get the bin labels.
            bins = reduction_bin_col
        else:
            bins = bin_on
    elif by:
        raise TypeError(f"not possible to have `by` of type {type(by)}.")
    else:
        raise ValueError("at least one among `by` and `bin_on` is required.")
    return by, cols_to_by, bin_on, reduction_bin_col, bins, bin_out_col


def _setup(
    store: ParquetSet,
    keys: Dict[dataclass, dict],
    ordered_on: str,
    trim_start: bool,
    reduction: bool,
    **kwargs,
):
    """Consolidate settings for parallel aggregations.

    Parameters
    ----------
    store : ParquetSet
        Store to which recording aggregation results.
    keys : Dict[dataclass, dict]
        Dict of keys for recording aggregation results in the form
        ``{key: {'agg': agg, 'by': by, 'bin_on': bin_on, 'post': post, **kwargs}}``
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order.
    trim_start : bool
        Flag possibly modified to indicate if trimming seed head is possible
        or not
    reduction : bool
        Flag indicating if reduction step will be performed on seed chunk or
        not.

    Other parameters
    ----------------
    kwargs : dict
        Settings considered as default ones if not specified within ``keys``.
        Default values for parameters related to aggregation can be set this
        way. (``agg``, ``by``, ``bin_on``, and ``post``)
        Parameters related to writing data are added to ``write_config``, that
        is then forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Can define for instance custom `max_row_group_size`
        parameter.

    Returns
    -------
    tuple
        Settings for chained aggregation.

          - ``all_cols_in``, list specifying all columns to loaded from seed
            data, required to perform the aggregation.
          - ``trim_start``, bool indicating if seed head is to be trimmed or
            not.
          - ``seed_index_restart_set``, set of ``seed_index_restart`` from each
            keys, if aggregation results are already available.
          - ``reduction_bin_cols``, list of column to use for binning at
            reduction step.
          - ``reduction_seed_chunk_cols``, list of columns already in seed, to
            be kept in 'seed_chunk' for reduction step.
          - ``vaex_sort``, str, column name to use for sorting groupby result
            from reduction step, in step reduction is performed with vaex.
          - ``reduction_agg``, dict specifying the minimal config to perform
            the reduction aggregation step, still containing all required
            aggregation functions. It is in the form:
            ``{"input_col__agg_function_name":("input_col", "agg_function_name")}``
            ``'input_col__agg_function_name'`` is used as generic name.
          - ``keys_config``, dict of keys config. A config is also a dict in
            the form:
            ``{key: {'dirpath': str, where to record agg res,
                     'agg_n_rows' : 0,
                     'agg_mean_row_group_size' : 0,
                     'agg_res' : None,
                     'agg_res_len' : None,
                     'isfbn' : True,
                     'by' : pandas Grouper, Callable or str (column name),
                     'cols_to_by' : list or None,
                     'reduction_bin_col' : str, col name of bins for reduction,
                     'bins' : pandas grouper or str (column name),
                     'bin_out_col' : str,
                     'agg' : dict,
                     'self_agg' : dict,
                     'post' : Callable or None,
                     'max_agg_row_group_size' : int,
                     'agg_chunk_buffer' : agg_chunk_buffer,
                     'last_agg_row' : empty pandas dataframe,
                     'binning_buffer' : dict, possibly empty,
                     'post_buffer' : dict, possibly empty,
                     'write_config' : {'ordered_on' : str,
                                       'duplicates_on' : str or list,
                                       ...
                                       },
                     },
               }``
            To be noticed:

              - key of dict ``keys_config`` is a string.
              - in case of a reduction step, 'agg' is modified to use as input
                column generic names from reduction aggregation.
              - 'self_agg' is the aggregation step required for stitching
                aggregations.

    """
    keys_config = {}
    seed_index_restart_set = set()
    all_cols_in = {ordered_on}
    reduction_agg = {}
    reduction_bin_cols = []
    reduction_seed_chunk_cols = set()
    # Some default values for keys.
    # 'agg_n_rows' : number of rows in aggregation result.
    # 'isfbn': is first row (from aggregation result) a new bin?
    #          For 1st iteration it is necessarily a new one.
    # 'last_agg_row' : initialized to an empty pandas dataframe, to allow using
    #                  'empty' attribute in subsequent loop.
    last_agg_row_dft = pDataFrame()
    key_default = {
        "agg_n_rows": 0,
        "agg_mean_row_group_size": 0,
        "agg_res": None,
        "agg_res_len": None,
        "isfbn": True,
        "by": None,
        "bins": None,
    }
    for i, (key, key_conf_in) in enumerate(keys.items()):
        # Parameters in 'key_conf_in' take precedence over those in 'kwargs'.
        key_conf_in = kwargs | key_conf_in
        # Step 1 / Process parameters.
        # Step 1.1 / 'by' and 'bin_on'.
        # Initialize 'bins' and 'bin_out_col' from 'by' and 'bin_on'.
        bin_on = key_conf_in.pop("bin_on", None)
        by = key_conf_in.pop("by", None)
        (by, cols_to_by, bin_on, reduction_bin_col, bins, bin_out_col) = _setup_binning(
            ordered_on=ordered_on, reduction=reduction, key_idx=i, by=by, bin_on=bin_on
        )
        # 'cols_to_by' defines columns to be loaded and sent to 'by' when 'by'
        # is a Callable. It has to be completed with 'ordered_on'.
        if bin_on:
            # Make sure it is in the columns to be loaded in seed.
            all_cols_in.add(bin_on)
        if reduction:
            # List of column onto which binning at reduction step.
            reduction_bin_cols.append(reduction_bin_col)
            if by is None:
                # Binning is achieved on 'bin_on' column directly.
                reduction_seed_chunk_cols.add(bin_on)
        # 'agg' required right at this step.
        # Making a copy as 'agg' may possibly be a dict coming from default
        # config. Because it get modified in below code, it is necessary to
        # modify the copy, and not the reference.
        agg = key_conf_in.pop("agg")
        if bin_out_col in agg:
            # Check that this name is not already that of an output column
            # from aggregation.
            raise ValueError(
                f"not possible to have {bin_out_col} as column name in"
                " aggregated results as it is also for column containing group"
                " keys."
            )
        # Step 1.2 / 'agg' and 'post'.
        # Initialize 'self_agg', 'agg' and update 'reduction_agg', 'all_cols_in'.
        self_agg = {}
        # 'agg' is in the form:
        # {"output_col":("input_col", "agg_function_name")}
        key_agg = {}
        for col_out, (col_in, agg_func) in agg.items():
            # Check if aggregation functions are allowed.
            if agg_func not in ACCEPTED_AGG_FUNC:
                raise ValueError(f"aggregation function '{agg_func}' is not tested yet.")
            # Update 'all_cols_in', list of columns from seed to be loaded.
            all_cols_in.add(col_in)
            if reduction:
                # Update 'reduction_agg', required for reduction step.
                # 'reduction_agg' is in the form:
                # {("input_col_name__agg_function_name") : ("input_col", "agg_function_name")}
                reduction_agg_col = f"{col_in}__{agg_func}"
                if reduction_agg_col not in reduction_agg:
                    reduction_agg[reduction_agg_col] = (
                        VAEX_AGG[agg_func](col_in) if reduction == VAEX else (col_in, agg_func)
                    )
                # Modify consequently 'agg' so that it can operate on agg results from
                # reduction step.
                key_agg[col_out] = (reduction_agg_col, agg_func)
                # Update list of columns to keep in seed chunk for reduction
                # step.
                reduction_seed_chunk_cols.add(col_in)
            # Update 'self_agg', required for stitching step.
            self_agg[col_out] = (col_out, agg_func)
        if not reduction:
            key_agg = agg
        # Initialize 'post'.
        post = key_conf_in.pop("post")
        # Step 1.3 / 'max_agg_row_group' and 'write_config'.
        # Initialize aggregation result max size before writing to disk.
        # If present, keep 'max_row_group_size' within 'key_conf_in' as it
        # is a parameter to be forwarded to the writer.
        max_agg_row_group_size = (
            key_conf_in["max_row_group_size"]
            if "max_row_group_size" in key_conf_in
            else MAX_ROW_GROUP_SIZE
        )
        # Initialize 'write_config', which are parameters remaining in
        # 'key_conf_in' and some adjustments.
        # Forcing 'ordered_on' for write.
        key_conf_in["ordered_on"] = ordered_on
        # Adding 'bin_out_col' to 'duplicates_on' except if 'duplicates_on' is set
        # already. In this case, if 'bin_out_col' is not in 'duplicates_on', it is
        # understood as a voluntary user choice to not have 'bin_on' in
        # 'duplicates_on'.
        # For all other cases, 'duplicates_on' has been set by user.
        # If 'bin_out_col' is not in 'duplicates_on', it is understood as a
        # voluntary choice by the user.
        if "duplicates_on" not in key_conf_in or key_conf_in["duplicates_on"] is None:
            if bin_out_col:
                # Force 'bin_out_col'.
                key_conf_in["duplicates_on"] = bin_out_col
            else:
                key_conf_in["duplicates_on"] = ordered_on
        # Step 2 / Process metadata if already existing aggregation results.
        # Initialize variables.
        if key in store:
            # Prior chainagg results already in store.
            # Retrieve corresponding metadata to re-start aggregations.
            prev_agg_res = store[key]
            if not _is_chainagg_result(prev_agg_res):
                raise ValueError(f"provided key '{key}' is not that of 'chainagg' results.")
            seed_index_restart, last_agg_row, binning_buffer, post_buffer = _get_chainagg_md(
                prev_agg_res
            )
            seed_index_restart_set.add(seed_index_restart)
            # Comment: pandas does not care about index name for concat. If
            # - prior agg results are obtained with 'reduction', hence index
            #   name of 'last_agg_row' has been modified,
            # - and that now, 'reduction' is not used,
            # it is not necessary to modify index name of 'last_agg_row'.
        else:
            last_agg_row = last_agg_row_dft
            # Because 'binning_buffer' and 'post_buffer' are modified in-place
            # for each key, they are created separately for each key.
            binning_buffer = {}
            post_buffer = {}
        # 'agg_chunks_buffer' is a buffer to keep aggregation chunks
        # before a concatenation to record. Because it is appended in-place
        # for each key, it is created separately for each key.
        keys_config[str(key)] = key_default | {
            "dirpath": os_path.join(store._basepath, key.to_path),
            "by": by,
            "cols_to_by": cols_to_by,
            "bins": bins,
            "reduction_bin_col": reduction_bin_col,
            "bin_out_col": bin_out_col,
            "agg": key_agg,
            "self_agg": self_agg,
            "post": post,
            "max_agg_row_group_size": max_agg_row_group_size,
            "agg_chunks_buffer": [],
            "last_agg_row": last_agg_row,
            "binning_buffer": binning_buffer,
            "post_buffer": post_buffer,
            "write_config": key_conf_in,
        }
    if not seed_index_restart_set:
        # No aggregation result existing yet. Whatever 'trim_start' value, no
        # trimming is possible.
        trim_start = False
    # In case of a reduction performed with vaex, result from reduction loses
    # its initial order. It is then necessary to re-order bins in the order of
    # appearance of 1st occurrence. For this, either we use an aggregation
    # already requested on 'ordered_on' column if relevant.
    # Or we will create a temporary index column on-the-fly, just before
    # reduction.
    vaex_sort = None
    if reduction == VAEX and (vaex_sort := f"{ordered_on}__{FIRST}") not in reduction_agg:
        # If not requested, request a new agg that will behave similarly.
        # Byt using 2 underscores, this column is considered hidden, and will
        # not be exported when translating the results back to pandas.
        vaex_sort = VAEX_SORT
        reduction_agg[vaex_sort] = VAEX_AGG[FIRST](ordered_on)
    return (
        list(all_cols_in),
        trim_start,
        seed_index_restart_set,
        reduction_bin_cols,
        list(reduction_seed_chunk_cols),
        vaex_sort,
        reduction_agg,
        keys_config,
    )


def _iter_data(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    trim_start: bool,
    seed_index_restart: Union[int, float, pTimestamp, None],
    discard_last: bool,
    all_cols_in: List[str],
):
    """Return an iterator over seed data.

    Parameters
    ----------
    seed : Union[vDataFrame, Tuple[int, vDataFrame], ParquetFile]
        Seed data over which conducting streamed aggregations.
        If a tuple made of an `int` and a vaex dataframe, the `int` defines
        the size of chunks into which is split the dataframe.
        If purely a vaex dataframe, it is split into chunks of `6_345_000`
        rows, which for a dataframe with 6 columns of ``float64`` or ``int64``,
        results in a memory footprint (RAM) of about 290MB.
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order.
    trim_start : bool
        Flag to indicate if trimming seed head has to be trimmed
    seed_index_restart : Union[int, float, pandas timestamp]
        Index in 'ordered_on' column to trim seed head (excluded from trimmed
        part).
    discard_last : bool
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step.
    all_cols_in : List[str]
        Names of columns to be loaded from seed data to so as to perform the
        aggregation.

    Returns
    -------
    last_seed_index, Generator[pDataFrame]

        - ``last_seed_index`` being the last index value in seed data.
        - The generator yielding chunks of seed data.

    """
    # Reason to discard last seed row (or row group) is twofold.
    # - last row is temporary (yet to get some final values),
    # - last rows are part of a single row group not yet complete itself (new
    #   rows part of this row group to be expected).
    if isinstance(seed, ParquetFile):
        # Case seed is a parquet file.
        # 'ordered_on' being necessarily in ascending order, last index
        # value is its max value.
        last_seed_index = seed.statistics["max"][ordered_on][-1]
        filter_seed = []
        if trim_start:
            filter_seed.append((ordered_on, ">=", seed_index_restart))
        if discard_last:
            filter_seed.append((ordered_on, "<", last_seed_index))
        if filter_seed:
            iter_data = seed.iter_row_groups(
                filters=[filter_seed], row_filter=True, columns=all_cols_in
            )
        else:
            iter_data = seed.iter_row_groups(columns=all_cols_in)
    else:
        # Case seed is a vaex dataframe.
        if isinstance(seed, tuple):
            vdf_row_group_size = seed[0]
            seed = seed[1]
        else:
            vdf_row_group_size = VDATAFRAME_ROW_GROUP_SIZE
        len_seed = len(seed)
        last_seed_index = seed.evaluate(ordered_on, len_seed - 1, len_seed, array_type="numpy")[0]
        if trim_start:
            # 'seed_index_restart' is excluded if defined.
            if isinstance(seed_index_restart, pTimestamp):
                # Vaex does not accept pandas timestamp, only numpy or pyarrow
                # ones.
                seed_index_restart = np.datetime64(seed_index_restart)
            seed = seed[seed[ordered_on] >= seed_index_restart]
        if discard_last:
            seed = seed[seed[ordered_on] < last_seed_index]
        if trim_start or discard_last:
            seed = seed.extract()
        iter_data = (
            tup[2]
            for tup in seed.to_pandas_df(chunk_size=vdf_row_group_size, column_names=all_cols_in)
        )
    return last_seed_index, iter_data


def _post_n_bin(
    seed_chunk: pDataFrame,
    reduction: bool,
    dirpath: str,
    key: str,
    agg_res: Union[pDataFrame, None],
    agg_res_len: int,
    agg_chunks_buffer: List[pDataFrame],
    agg_n_rows: int,
    agg_mean_row_group_size: int,
    max_agg_row_group_size: int,
    write_config: dict,
    bin_out_col: Union[str, None],
    post: Union[Callable, None],
    isfbn: bool,
    post_buffer: dict,
    bins: Union[str, Series],
    by: Union[Grouper, Callable, str],
    cols_to_by: List[str],
    binning_buffer: dict,
    reduction_bin_col: str,
):
    """Conduct post-processing, and writing for iter. n-1, and binning for iter. n.

    Parameters
    ----------
    seed_chunk : pDataFrame
        Chunk of seed data.
    reduction : Union[bool, str]
        If the reduction step is to be performed.
        Can be "vaex".
    dirpath : str
        Path to which recording aggregation results.
    key : str
        Key for retrieving metadata corresponding to aggregation results.

    Other parameters
    ----------------
    config
        Settings related to 'key' for conducting post-processing, writing and
        binning.

    Returns
    -------
    key, updated_config

        - ``key``, key to which changed parameters are related.
        - ``updated_config``, dict with modified parameters.

    """
    if agg_res is not None:
        # If previous results, check if this is write time.
        # Spare last aggregation row as a dataframe for stitching with new
        # aggregation results from current iteration.
        if agg_res_len > 1:
            # Remove last row from 'agg_res' and add to
            # 'agg_chunks_buffer'.
            agg_chunks_buffer.append(agg_res.iloc[:-1])
            # Remove last row that is not recorded from total row number.
            agg_n_rows += agg_res_len - 1
        # Keep floor part.
        if agg_n_rows:
            # Length of 'agg_chunks_buffer' is number of times it has been
            # appended.
            agg_mean_row_group_size = agg_n_rows // len(agg_chunks_buffer)
            if agg_n_rows + agg_mean_row_group_size >= max_agg_row_group_size:
                # Write results from previous iteration.
                _post_n_write_agg_chunks(
                    chunks=agg_chunks_buffer,
                    dirpath=dirpath,
                    key=key,
                    write_config=write_config,
                    index_name=bin_out_col,
                    post=post,
                    isfbn=isfbn,
                    post_buffer=post_buffer,
                    other_metadata=None,
                )
                # Reset number of rows within chunk list and number of
                # iterations to fill 'agg_chunks_buffer'.
                agg_n_rows = 0
    if reduction:
        # In case reduction step is requested, 'reduction_bins' has to be an 1D
        # array-like.
        if callable(by):
            reduction_bins = by(data=seed_chunk.loc[:, cols_to_by], buffer=binning_buffer)
        elif isinstance(by, Grouper):
            reduction_bins = tcut(data=seed_chunk.loc[:, by.key], grouper=by).astype(
                "datetime64[ns]"
            )
        else:
            # Bin directly on existing column. Name is available with 'bins'.
            # Column will be used directly column from seed_chunk.
            reduction_bins = None
        if isinstance(reduction_bins, Series):
            # Set "generic" names to bin arrays so that when they are
            # concatenated, a same name is not used twice.
            if not bin_out_col and (out_name := reduction_bins.name):
                # Initialize 'bin_out_col' with name defined by 'by'.
                # Risk of several bin columns having the same name this way.
                # Only done in 'last resort' if 'bin_out_col' has not been
                # defined after setup.
                bin_out_col = out_name
            if reduction == VAEX:
                reduction_bins = from_arrays(**{reduction_bin_col: reduction_bins.to_numpy()})
            else:
                # Rename with "generic name".
                reduction_bins.name = reduction_bin_col
        elif reduction_bins is not None:
            # Then 'reduction_bins' is a collection of some sort.
            # Let's make it a pandas Series and provide it a name.
            reduction_bins = (
                from_arrays(**{reduction_bin_col: reduction_bins})
                if reduction == VAEX
                else Series(reduction_bins, name=reduction_bin_col)
            )
    else:
        # No reduction step.
        reduction_bins = None
        if callable(by):
            # Case callable. Bin 'ordered_on'.
            # If 'binning_buffer' is used, it has to be modified in-place, so
            # as to ship values from iteration N to iteration N+1.
            bins = by(data=seed_chunk.loc[:, cols_to_by], buffer=binning_buffer)
            if not bin_out_col and isinstance(bins, Series) and (out_name := bins.name):
                # Initialize 'bin_out_col' with name defined by 'by'.
                bin_out_col = out_name
    # Updating key settings.
    updated_conf = {
        "agg_chunks_buffer": agg_chunks_buffer,
        "agg_n_rows": agg_n_rows,
        "agg_mean_row_group_size": agg_mean_row_group_size,
        "post_buffer": post_buffer,
        "binning_buffer": binning_buffer,
        "reduction_bins": reduction_bins,
        "bins": bins,
        "bin_out_col": bin_out_col,
    }
    return key, updated_conf


def _group_n_stitch(
    seed_chunk: pDataFrame,
    key: str,
    bins: str,
    agg: dict,
    last_agg_row: pDataFrame,
    agg_chunks_buffer: List[pDataFrame],
    agg_n_rows: int,
    self_agg: dict,
    isfbn: bool,
):
    """Conduct groupby, and stitching of previous groupby with current one.

    Parameters
    ----------
    seed_chunk : pDataFrame
        Chunk of seed data.
    key : str
        Key for recording aggregation results.

    Other parameters
    ----------------
    config
        Settings related to 'key' for conducting groupby and stitching.

    Returns
    -------
    key, updated_config

        - ``key``, key to which changed parameters are related.
        - ``updated_config``, dict with modified parameters.

    """
    # Bin and aggregate. Do not sort to keep order of groups as they
    # appear. Group keys becomes the index.
    agg_res = seed_chunk.groupby(bins, sort=False).agg(**agg)
    agg_res_len = len(agg_res)
    # Stitch with last row from *prior* aggregation.
    if not last_agg_row.empty:
        isfbn = (first := agg_res.index[0]) != (last := last_agg_row.index[0])
        if isfbn:
            n_added_rows = 1
            # Bin of 'last_agg_row' does not match bin of first row in
            # 'agg_res'.
            if isinstance(bins, Grouper) and bins.freq:
                # If bins are defined with pandas time grouper ('freq'
                # attribute is not `None`), bins without values from seed
                # that could exist at start of chunk will be missing.
                # In an usual pandas aggregation, these bins would however
                # be present in aggregation results, with `NaN` values in
                # columns. These bins are thus added here to maintain
                # usual pandas behavior.
                missing = date_range(
                    start=last, end=first, freq=bins.freq, inclusive="neither", name=bins.key
                )
                if not missing.empty:
                    last_agg_row = pconcat(
                        [last_agg_row, pDataFrame(index=missing, columns=last_agg_row.columns)]
                    )
                    n_added_rows = len(last_agg_row)
            # Add last previous row (and possibly missing ones if pandas
            # time grouper) in 'agg_chunk_buffer' and do nothing with
            # 'agg_res' at this step.
            agg_chunks_buffer.append(last_agg_row)
            agg_n_rows += n_added_rows
        else:
            # If previous results existing, and if same bin labels shared
            # between last row of previous aggregation results (meaning same
            # bin), and first row of new aggregation results, then replay
            # aggregation between both.
            agg_res.iloc[:1] = (
                pconcat([last_agg_row, agg_res.iloc[:1]])
                .groupby(level=0, sort=False)
                .agg(**self_agg)
            )
    # Setting 'last_agg_row' from new 'agg_res'.
    last_agg_row = agg_res.iloc[-1:] if agg_res_len > 1 else agg_res
    # Updating key settings.
    updated_conf = {
        "agg_res": agg_res,
        "agg_res_len": agg_res_len,
        "isfbn": isfbn,
        "agg_chunks_buffer": agg_chunks_buffer,
        "agg_n_rows": agg_n_rows,
        "last_agg_row": last_agg_row,
    }
    return key, updated_conf


def agg_loop_wo_reduction(seed_chunk: pDataFrame, key: str, config: dict):
    """Aggregate without reduction mechanism."""
    _, updated_config = _post_n_bin(
        seed_chunk=seed_chunk,
        reduction=False,
        key=key,
        dirpath=config["dirpath"],
        agg_res=config["agg_res"],
        agg_res_len=config["agg_res_len"],
        agg_chunks_buffer=config["agg_chunks_buffer"],
        agg_n_rows=config["agg_n_rows"],
        agg_mean_row_group_size=config["agg_mean_row_group_size"],
        max_agg_row_group_size=config["max_agg_row_group_size"],
        write_config=config["write_config"],
        bin_out_col=config["bin_out_col"],
        post=config["post"],
        isfbn=config["isfbn"],
        post_buffer=config["post_buffer"],
        bins=config["bins"],
        by=config["by"],
        cols_to_by=config["cols_to_by"],
        binning_buffer=config["binning_buffer"],
        reduction_bin_col=config["reduction_bin_col"],
    )
    # Consolidate results.
    config.update(updated_config)
    _, updated_config = _group_n_stitch(
        seed_chunk=seed_chunk,
        key=key,
        bins=config["bins"],
        agg=config["agg"],
        last_agg_row=config["last_agg_row"],
        agg_chunks_buffer=config["agg_chunks_buffer"],
        agg_n_rows=config["agg_n_rows"],
        self_agg=config["self_agg"],
        isfbn=config["isfbn"],
    )
    # Consolidate results.
    config.update(updated_config)
    return key, config


def agg_loop_with_reduction(
    iter_data: Generator,
    ordered_on: str,
    keys_config: dict,
    p_job: Parallel,
    reduction_seed_chunk_cols: list,
    reduction_bin_cols: list,
    reduction_agg: dict,
    with_vaex: bool = False,
    vaex_sort: str = None,
):
    """Aggregate with reduction mechanism."""
    for seed_chunk in iter_data:
        bins_n_conf = p_job(
            delayed(_post_n_bin)(
                seed_chunk=seed_chunk,
                reduction=VAEX if with_vaex else True,
                key=key,
                dirpath=config["dirpath"],
                agg_res=config["agg_res"],
                agg_res_len=config["agg_res_len"],
                agg_chunks_buffer=config["agg_chunks_buffer"],
                agg_n_rows=config["agg_n_rows"],
                agg_mean_row_group_size=config["agg_mean_row_group_size"],
                max_agg_row_group_size=config["max_agg_row_group_size"],
                write_config=config["write_config"],
                bin_out_col=config["bin_out_col"],
                post=config["post"],
                isfbn=config["isfbn"],
                post_buffer=config["post_buffer"],
                bins=config["bins"],
                by=config["by"],
                cols_to_by=config["cols_to_by"],
                binning_buffer=config["binning_buffer"],
                reduction_bin_col=config["reduction_bin_col"],
            )
            for key, config in keys_config.items()
        )
        reduction_bins = []
        for key, config in bins_n_conf:
            if config["reduction_bins"] is not None:
                # Cases 'by' is pd.Grouper or Callable.
                # In this case, spare 'reduction_bins'.
                # It is popped so that it is not serialized/copy in next
                # parallel run (this data is a 1D-array).
                # Keys of 'reduction_bins' are names of
                # 'reduction_bin_col' for only new columns (not the ones
                # that have to be used directly for binning)
                reduction_bins.append(config.pop("reduction_bins"))
            # Consolidate results.
            keys_config[key].update(config)
        # Re-create a seed_chunk with new 'bins' columns.
        if with_vaex:
            # Reduction with vaex.
            seed_chunk = from_pandas(seed_chunk[reduction_seed_chunk_cols])
            if ordered_on not in reduction_seed_chunk_cols:
                # Keep track of initial order to sort aggregation results
                # after vaex reduction.
                seed_chunk[ordered_on] = vrange(0, len(seed_chunk))
            for red_bin in reduction_bins:
                seed_chunk.join(red_bin, inplace=True)
            seed_chunk = (
                seed_chunk.groupby(reduction_bin_cols, sort=False)
                .agg(reduction_agg)
                .sort(vaex_sort)
            )
            if VAEX_SORT in seed_chunk.get_column_names():
                seed_chunk = seed_chunk.drop(VAEX_SORT)
            seed_chunk = seed_chunk.to_pandas_df()
        else:
            # Reduction with pandas.
            seed_chunk = pconcat(
                [*reduction_bins, seed_chunk[reduction_seed_chunk_cols]], axis=1, copy=False
            )
            seed_chunk = seed_chunk.groupby(reduction_bin_cols, sort=False).agg(**reduction_agg)
            seed_chunk.reset_index(inplace=True)
        agg_res_n_conf = p_job(
            delayed(_group_n_stitch)(
                seed_chunk=seed_chunk,
                key=key,
                bins=config["bins"],
                agg=config["agg"],
                last_agg_row=config["last_agg_row"],
                agg_chunks_buffer=config["agg_chunks_buffer"],
                agg_n_rows=config["agg_n_rows"],
                self_agg=config["self_agg"],
                isfbn=config["isfbn"],
            )
            for key, config in keys_config.items()
        )
        for key, config in agg_res_n_conf:
            keys_config[key].update(config)


def chainagg(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    store: ParquetSet,
    keys: Union[dataclass, dict],
    agg: dict = None,
    by: Union[Grouper, Callable[[Series, dict], Series]] = None,
    bin_on: Union[str, Tuple[str, str]] = None,
    post: Callable = None,
    trim_start: bool = True,
    discard_last: bool = True,
    reduction: Union[bool, str] = False,
    parallel: bool = False,
    **kwargs,
):
    """Aggregate sequentially on successive chunks (stream) of ordered data.

    This function conducts 'streamed aggregation' iteratively (out-of core)
    with optional post-processing of aggregation results (by use of vectorized
    functions or for dataframe formatting).
    No (too much) attention is then required about how stitching new
    aggregation results (from new seed data) to prior aggregation results (from
    past seed data).

    Parameters
    ----------
    seed : Union[vDataFrame, Tuple[int, vDataFrame], ParquetFile]
        Seed data over which conducting streamed aggregations.
        If a tuple made of an `int` and a vaex dataframe, the `int` defines
        the size of chunks into which is split the dataframe.
        If purely a vaex dataframe, it is split into chunks of `6_345_000`
        rows, which for a dataframe with 6 columns of ``float64`` or ``int64``,
        results in a memory footprint (RAM) of about 290MB.
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order. While this parameter is compulsory (most notably to manage
        duplicates when writing new aggregated results to existing ones), seed
        data is not necessarily grouped by this column, in which case ``by``
        and/or ``bin_on`` parameters have to be set.
    store : ParquetSet
        Store to which recording aggregation results.
    keys : Union[Indexer, dict]
        Key for recording aggregation results.
        If a dict, several keys can be specified for operating multiple
        parallel aggregations on the same seed. In this case, the dict should
        be in the form
        ``{key: {'agg': agg, 'by': by, 'bin_on': bin_on, 'post': post, **kwargs}}``
        Any additional parameters, (``**kwargs``) are forwarded to
        ``oups.writer.write`` when writing aggregation results to store.
        Please, note:

          - If not specified, `by` and `bin_on` parameters in dict do not get
            default values.
          - If not specified `agg` and `post` parameters in dict get values
            from `agg` and `post` parameters defined when calling `chainagg`.
            If using 'post' when calling 'chainagg' and not willing to apply
            it for one key, set it to ``None`` in key specific config.

    agg : Union[dict, None], default None
        Dict in the form ``{"output_col":("input_col", "agg_function_name")}``
        where keys are names of output columns into which are recorded
        results of aggregations, and values describe the aggregations to
        operate. ``input_col`` has to exist in seed data.
        Examples of ``agg_function_name`` are `first`, `last`, `min`, `max` and
        `sum`.
        This parameter is compulsory, except if ``key`` parameter is a`dict`.
    by : Union[pd.Grouper, Callable[[pd.DataFrame, dict], array-like]], default
         None
        Parameter defining the binning logic.
        If a `Callable`, it is given following parameters.

          - A ``data`` parameter, a pandas dataframe made of column
            ``ordered_on``, and column ``bin_on`` if different than
            ``ordered_on``.
          - A ``buffer`` parameter, a dict that can be used as a buffer for
            storing temporary results from one chunk processing to
            the next.

        This `Callable` has then to return an array of the same length as the
        input dataframe, and that specifies bin labels, row per row.
        If data are required for re-starting calculation of bins on the next
        data chunk, the buffer has to be modified in place with temporary
        results for next-to-come binning iteration.
    bin_on : Union[str, Tuple[str, str]], default None
        ``bin_on`` may either be a string or a tuple of 2 string. When a
        string, it refers to an existing column in seed data. When a tuple,
        the 1st string refers to an existing column in seed data, the 2nd the
        name to use for the column which values will be the group keys in
        aggregation results.
        Moreover, setting of ``bin_on`` should be adapted depending how is
        defined ``by`` parameter. In all the cases mentioned below, ``bin_on``
        can either be a string or a tuple of 2 string.

          - if ``by`` is ``None`` then ``bin_on`` is expected to be set to an
            existing column name, which values are directly used for binning.
          - if ``by`` is a callable, then ``bin_on`` can have different values.

            - ``None``, the default.
            - the name of an existing column onto which applying the binning
              defined by ``by`` parameter. Its value is then carried over as
              name for the column containing the group keys.

        It is further used when writing results for defining ``duplicates_on``
        parameter (see ``oups.writer.write``).
    post : Callable, default None
        User-defined function accepting 3 parameters.

          - First, the pandas dataframe resulting from the aggregations defined
            by ``agg`` parameter, with first row already corrected with last
            row of previous streamed aggregation.
          - Second, a boolean which indicates if first row of aggregation
            result is a new bin, or is the 'same' that last bin of aggregation
            result. If 'same', all values may not be the same (updated), but
            the aggregation bin is the same.
          - Third, a dict to be used as data buffer, that can be necessary for
            some user-defined post-processing requiring data assessed in
            previous post-processing iteration.

        It has then to return a pandas dataframe that will be recorded.
        This optional post-processing is intended for use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), or dataframe formatting before results are finally recorded.
    trim_start : bool, default True
        If ``True``, and if aggregated results already existing, then retrieves
        the first index from seed data not processed yet (recorded in metadata
        of existing aggregated results), and trim all seed data before this
        index (index excluded from trim).
    discard_last : bool, default True
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step. See below
        notes.
    reduction : bool or "vaex", default False
        If `True`, perform a 1st multi-groupby and aggregation on seed data.
        This 1st step (common to all keys) reduces then the side of seed data
        before it is used for each individual groupby and aggregation of each
        key.
        The more the number of keys, the more the reduction step ought to bring
        performance improvement.
        If `"vaex"`, the reduction step is performed with vaex instead of
        pandas.
    parallel : bool, default False
        Conduct binning, post-processing and writing in parallel, with one
        process per `key`. If a single `key`, only one process is possible.
        This does not impact execution of reduction in step in parallel if
        vaex is used.

    Other parameters
    ----------------
    kwargs : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Can define for instance custom `max_row_group_size`
        parameter.

    Notes
    -----
    - Result is necessarily added to a dataset from an instantiated oups
      ``ParquetSet``. ``chainagg`` actually relies on the `advanced` update
      feature from oups.
    - If aggregation results already exist in the instantiated oups
      ``ParquetSet``, last index from previous aggregation is retrieved, and
      prior seed data is trimmed.
    - Aggregation is by default processed up to the last 'complete' index
      (included), and subsequent aggregation will start from the last index
      (included), assumed to be that of an incomplete row group.
      If `discard_last` is set `False, then aggregation is process up to the
      last data.
    - By default, with parameter `discard_last`` set ``True``, the last row
      group (composed from rows sharing the same value in `ordered_on` column),
      is discarded.

        - It may be for instance that this row group is not complete yet and
          should therefore not be accounted for. More precisely, new rows with
          same value in `ordered_on` may appear in seed data later on. Because
          seed data is trimmed to start from last processed value from
          `ordered_on` column (value included), these new rows would be
          excluded from the next aggregation, leading to an inaccurate
          aggregation result. Doing so is a way to identify easily when
          re-starting the aggregation in a case there can be duplicates in
          `ordered_on` column. A ``sum`` aggregation will then return the
          correct result for instance, as no data is accounted for twice.
        - Or if composed of a single row, this last row in seed data is
          temporary (and may get its final values only at a later time, when it
          becomes the one-but-last row, as a new row is added).

    - If ``kwargs`` defines a maximum row group size to write to disk, this
      value is also used to define a maximum size of aggregation results before
      actually triggering a write. If no maximum row group size is defined,
      then ``MAX_ROW_GROUP_SIZE`` defined in ``oups.writer.write`` is used.
    - With the post-processing step, user can also take care of removing
      columns produced by the aggregation step, but not needed afterwards.
      Other formatting operations on the dataframe can also be achieved
      (renaming columns or index, and so on...). To be noticed, group keys are
      available through a column having same name as initial column from seed
      data, or defined by 'bin_on' parameter if 'by' is a callable.
    - When recording, both 'ordered_on' and 'duplicates_on' parameters are set
      when calling ``oups.writer.write``. If additional parameters are defined
      by the user, some checks are made.

        - 'ordered_on' is forced to 'chainagg' ``ordered_on`` parameter.
        - If 'duplicates_on' is not set by the user or is `None`, then it is
          set to the name of the output column for group keys defined by
          `bin_on`. The rational is that this column identifies uniquely each
          bin, and so is a relevant column to identify duplicates. But then,
          there might be case for which 'ordered_on' column does this job
          already (if there are unique values in 'ordered_on') and the column
          containing group keys is then removed during user post-processing.
          To allow this case, if the user is setting ``duplicates_on`` as
          additional parameter to ``chainagg``, it is not
          modified. It means omission of the column name containing the group
          keys, as defined by 'bin_on' parameter when it is set, is a
          voluntary choice from the user.

    """
    # Parameter setup.
    if not isinstance(keys, dict):
        if not agg:
            raise ValueError("not possible to use a single key without specifying parameter 'agg'.")
        keys = {keys: {"agg": agg, "by": by, "bin_on": bin_on, "post": post}}
    # 'reduction_bin_cols' contain ALL columns to use for binning, including
    # column to be used "as they are" for binning. ('by' is `None`).
    (
        all_cols_in,
        trim_start,
        seed_index_restart_set,
        reduction_bin_cols,
        reduction_seed_chunk_cols,
        vaex_sort,
        reduction_agg,
        keys_config,
    ) = _setup(
        ordered_on=ordered_on,
        store=store,
        keys=keys,
        agg=agg,
        post=post,
        trim_start=trim_start,
        reduction=reduction,
        **kwargs,
    )
    if len(seed_index_restart_set) > 1:
        raise ValueError(
            "not possible to aggregate on multiple keys with existing"
            " aggregation results not aggregated up to the same seed index."
        )
    elif seed_index_restart_set:
        seed_index_restart = seed_index_restart_set.pop()
    else:
        seed_index_restart = None
    # Initialize 'iter_data' generator from seed data, with correct trimming.
    last_seed_index, iter_data = _iter_data(
        seed, ordered_on, trim_start, seed_index_restart, discard_last, all_cols_in
    )
    n_keys = len(keys)
    n_jobs = min(int(cpu_count() * 3 / 4), n_keys) if (parallel and n_keys > 1) else 1
    with Parallel(n_jobs=n_jobs, prefer="threads") as p_job:
        if reduction:
            # 'keys_config' is updated i-place.
            agg_loop_with_reduction(
                iter_data,
                ordered_on,
                keys_config,
                p_job,
                reduction_seed_chunk_cols,
                reduction_bin_cols,
                reduction_agg,
                reduction == VAEX,
                vaex_sort,
            )
        else:
            for seed_chunk in iter_data:
                agg_loop_res = p_job(
                    delayed(agg_loop_wo_reduction)(seed_chunk, key, config)
                    for key, config in keys_config.items()
                )
                for key, config in agg_loop_res:
                    keys_config[key].update(config)
        # Check if at least one iteration has been achieved or not.
        agg_res = next(iter(keys_config.values()))["agg_res"]
        if agg_res is None:
            # No iteration has been achieved, as no data.
            return
        # Post-process & write results from last iteration, this time keeping
        # last row, and recording metadata for a future 'chainagg' execution.
        # A deep copy is made for 'last_agg_row' to prevent a specific case where
        # 'agg_chuks_buffer' is a list of a single 'agg_res' dataframe of a single
        # row. In this very specific case, both 'agg_res' and 'last_agg_row' points
        # toward the same dataframe, but 'agg_res' gets modified in '_post_n_write'
        # while 'last_agg_row' should not be. The deep copy prevents this.
        p_job(
            delayed(_post_n_write_agg_chunks)(
                key=key,
                dirpath=config["dirpath"],
                chunks=[*config["agg_chunks_buffer"], config["agg_res"]],
                write_config=config["write_config"],
                index_name=config["bin_out_col"],
                post=config["post"],
                isfbn=config["isfbn"],
                post_buffer=config["post_buffer"],
                other_metadata=(
                    last_seed_index,
                    config["last_agg_row"].copy(),
                    config["binning_buffer"],
                ),
            )
            for key, config in keys_config.items()
        )
    # Add keys in store for those who where not in.
    for key in keys:
        if key not in store:
            store._keys.add(key)