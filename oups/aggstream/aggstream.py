#!/usr/bin/env python3
"""
Created on Wed Nov 15 21:30:00 2023.

@author: yoh

"""
from collections import ChainMap
from dataclasses import dataclass
from itertools import chain
from multiprocessing import cpu_count
from os import path as os_path
from typing import Callable, Iterable, Optional, Tuple, Union

from joblib import Parallel
from joblib import delayed
from numpy import ones
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat
from pandas.core.resample import TimeGrouper

from oups.aggstream.cumsegagg import cumsegagg
from oups.aggstream.cumsegagg import setup_cumsegagg
from oups.aggstream.jcumsegagg import FIRST
from oups.aggstream.jcumsegagg import LAST
from oups.aggstream.jcumsegagg import MAX
from oups.aggstream.jcumsegagg import MIN
from oups.aggstream.jcumsegagg import SUM
from oups.aggstream.segmentby import KEY_BIN_BY
from oups.aggstream.segmentby import KEY_BIN_ON
from oups.aggstream.segmentby import KEY_ORDERED_ON
from oups.aggstream.segmentby import KEY_SNAP_BY
from oups.aggstream.segmentby import setup_segmentby
from oups.aggstream.utils import dataframe_filter
from oups.store import ParquetSet
from oups.store import is_toplevel
from oups.store.router import ParquetHandle
from oups.store.writer import KEY_DUPLICATES_ON
from oups.store.writer import KEY_MAX_ROW_GROUP_SIZE
from oups.store.writer import MAX_ROW_GROUP_SIZE
from oups.store.writer import OUPS_METADATA
from oups.store.writer import OUPS_METADATA_KEY
from oups.store.writer import write


# Aggregation functions.
ACCEPTED_AGG_FUNC = {FIRST, LAST, MIN, MAX, SUM}
# List of keys.
KEY_AGGSTREAM = "aggstream"
KEY_SEGAGG_BUFFER = "segagg_buffer"
KEY_POST_BUFFER = "post_buffer"
KEY_BIN_ON_OUT = "bin_on_out"
KEY_AGG_RES_BUFFER = "agg_res_buffer"
KEY_BIN_RES_BUFFER = "bin_res_buffer"
KEY_CHECK = "check"
KEY_CHECK_BUFFER = "check_buffer"
KEY_FILTERS = "filters"
KEY_RESTART_INDEX = "restart_index"
KEY_AGG_RES = "agg_res"
KEY_BIN_RES = "bin_res"
# Filters
NO_FILTER_ID = "_"
# List of valid parameters for 'key_conf_in'
KEY_AGG = "agg"
KEY_POST = "post"
# 'bin_by' is a compulsory parameter, and a specific check is made for it.
# It is not added in 'KEY_CONF_IN_PARAMS'.
WRITE_PARAMS = set(write.__code__.co_varnames[: write.__code__.co_argcount])
KEY_CONF_IN_PARAMS = {KEY_BIN_ON, KEY_SNAP_BY, KEY_AGG, KEY_POST} | WRITE_PARAMS


def _is_aggstream_result(handle: ParquetHandle) -> bool:
    """
    Check if input handle is that of a dataset produced by streamaag.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file.
    filters :

    Returns
    -------
    bool
        `True` if parquet file contains metadata as produced by
        ``oups.aggstream``, which confirms this dataset has been produced with
        this latter function.

    """
    # As oups specific metadata is a string produced by json library, the last
    # 'in' condition is checking if the set of characters defined by
    # 'AGGSTREAM' is in a string.
    if OUPS_METADATA_KEY in handle.metadata:
        return KEY_AGGSTREAM in handle._oups_metadata


def _init_keys_config(
    seed_ordered_on: str,
    store: ParquetSet,
    keys_config: dict,
    keys_default: dict,
):
    """
    Consolidate keys' configuration into ``keys_config`` and ``agg_pd``.

    Parameters
    ----------
    seed_ordered_on : str
        Name of the column with respect to which seed is in ascending order.
        This parameter is used for seed segmentation. It is also used as
        default name of the column with respect to which aggregation results are
        in ascending order, if not provided in ``keys`` parameter.
    store : ParquetSet
        Store to which recording aggregation results.
    keys_config : dict
        Unconsolidated keys config.
    keys_default : dict
        Default values for missing parameters in ``keys_config``.

    Other Parameters
    ----------------
    kwargs : dict
        Other user parameters that will be set into ``keys_config``.

    Returns
    -------
    The following AggStream's parameters are initialized with this function.

      - ``keys_config``, dict of keys' config in the form:
        ``{key: {'dirpath': str, where to record agg res,
                 'bin_on_out' : str, name in aggregation results for column
                                with bin ids.
                 'seg_config' : dict specifying the segmentation config,
                 'post' : Callable or None,
                 'write_config' : {'ordered_on' : str,
                                   'duplicates_on' : str or list,
                                   ...
                                  },
               },
         }``
        To be noticed keys of dict ``keys_config`` are strings.
      - ``self.agg_pd``, dict, specifying per key the aggregation
        configuration.

    """
    consolidated_keys_config = {}
    agg_pd = {}
    for key, key_conf_in in keys_config.items():
        # Parameters in 'key_conf_in' take precedence over those in
        # 'keys_default'. Additionally, with this step, 'key_conf_in' is a
        # deep copy, and when parameters are popped, it does not affect
        # the initial 'key_conf_in'.
        try:
            bin_by = key_conf_in.pop(KEY_BIN_BY)
        except KeyError:
            raise ValueError(f"'{KEY_BIN_BY}' parameter is missing for key '{key}'.")
        key_conf_in = keys_default | key_conf_in
        key_as_str = str(key)
        # Check parameters in 'key_conf_in' are valid ones.
        for param in key_conf_in:
            if param not in KEY_CONF_IN_PARAMS:
                raise ValueError(
                    f"'{param}' not a valid parameters in" f" '{key}' aggregation config.",
                )
        bin_on = key_conf_in.pop(KEY_BIN_ON, None)
        if isinstance(bin_on, tuple):
            # 'bin_on_out' is name of column containing group keys in
            # 'agg_res'. Setting of 'bin_on_out' is an 'AggStream'
            # task, not a 'cumesegagg' one. This is because this
            # parameter clarifies then how to set 'duplicates_on'
            # parameter for 'oups.writer.write' which is also part of
            # 'AggStream' perimeter.
            bin_on, bin_on_out = bin_on
        else:
            bin_on_out = None
        # Setup 'seg_conf', 'bin_on_out' & 'agg_pd'.
        try:
            seg_config = setup_segmentby(
                bin_by=bin_by,
                bin_on=bin_on,
                ordered_on=seed_ordered_on,
                snap_by=key_conf_in.pop(KEY_SNAP_BY),
            )
        except Exception:
            raise ValueError(f"exception raised for key '{key}'")
        if bin_on := seg_config[KEY_BIN_ON]:
            if bin_on_out is None:
                # It may be that 'bin_on' value has been modified in
                # 'setup_segmentby'. If 'bin_on_out' has not been set
                # previously, then set it to this possibly new value of
                # 'bin_on'.
                bin_on_out = bin_on
        agg_pd[key_as_str] = key_conf_in.pop(KEY_AGG)
        # 'agg' is in the form:
        # {"output_col":("input_col", "agg_function_name")}
        if bin_on_out in agg_pd[key_as_str]:
            # Check that this name is not already that of an output
            # column from aggregation.
            raise ValueError(
                f"not possible to have {bin_on_out} as column name in"
                " aggregated results as it is also for column"
                " containing group keys.",
            )
        # Setup 'max_agg_row_group' and 'write_config'.
        # Initialize aggregation result max size before writing to disk.
        # If present, keep 'max_row_group_size' within 'key_conf_in' as
        # it is a parameter to be forwarded to the writer.
        if KEY_MAX_ROW_GROUP_SIZE not in key_conf_in:
            key_conf_in[KEY_MAX_ROW_GROUP_SIZE] = MAX_ROW_GROUP_SIZE
        # Initialize 'write_config', which are parameters remaining in
        # 'key_conf_in' and some adjustments.
        # Adding 'bin_on_out' to 'duplicates_on' except if
        # 'duplicates_on' is set already. In this case, if 'bin_on_out'
        # is not in 'duplicates_on', it is understood as a voluntary
        # user choice. For all other cases, 'duplicates_on' has been
        # set by user. Setting 'duplicates_on' is the true reason of
        # having 'bin_on_out'. It allows the user to inform 'AggStream'
        # that the binning column (with unique keys) is this one.
        if KEY_DUPLICATES_ON not in key_conf_in or key_conf_in[KEY_DUPLICATES_ON] is None:
            # Force 'bin_on_out', else reuse 'ordered_on' parameter
            # specific to keys (aggregation results).
            key_conf_in[KEY_DUPLICATES_ON] = (
                bin_on_out if bin_on_out else key_conf_in[KEY_ORDERED_ON]
            )
        consolidated_keys_config[key_as_str] = {
            "dirpath": os_path.join(store._basepath, key.to_path),
            "seg_config": seg_config,
            KEY_BIN_ON_OUT: bin_on_out,
            KEY_POST: key_conf_in.pop(KEY_POST),
            "write_config": key_conf_in,
        }
    return consolidated_keys_config, agg_pd


def _init_agg_buffers(
    store: ParquetSet,
    keys: dict,
):
    """
    Initialize aggregation buffers into ``agg_buffers``.

    Also set ``seed_index_restart``.

    Parameters
    ----------
    store : ParquetSet
        Store to which aggregation results may already exist, and from which
        retrieving previous buffer data.
    keys : Union[dataclass, dict]
        Single level dict as defined in ``_init__`` function.

    Returns
    -------
    The following AggStream's parameters are initialized in this function.
      - ``seed_index_restart``, int, float or pTimestamp, the index
        from which (included) should be restarted the nex aggregation
        iteration.
      - ``agg_buffers``, dict of aggregation buffer variables specific for each
        key, in the form:
        ``{key: {'agg_n_rows' : 0,
                 'agg_res' : None,
                 'bin_res' : None,
                 'agg_res_buffer' : list, agg_res_buffer,
                 'bin_res_buffer' : list, bin_res_buffer,
                 'segagg_buffer' : dict, possibly empty,
                 'post_buffer' : dict, possibly empty,
               },
         }``
        To be noticed keys of dict ``agg_buffers`` are strings.

    """
    agg_buffers = {}
    seed_index_restart_set = set()
    for key in keys:
        key_as_str = str(key)
        # Default values for aggregation counters and buffers.
        # 'agg_n_rows' : number of rows in aggregation result.
        # 'agg_res_buffer' and 'bin_res_buffer' are buffers to keep
        # aggregation chunks before a concatenation to record. Because
        # they are appended in-place for each key, they are created
        # separately for each key.
        # Because 'segagg_buffer' and 'post_buffer' are modified
        # in-place for each key, they are created separately for
        # each key.
        agg_buffers[key_as_str] = _reset_agg_buffers()
        # Process metadata if already existing aggregation results.
        if key in store:
            # Prior AggStream results already in store.
            # Retrieve corresponding metadata to re-start aggregations.
            prev_agg_res = store[key]
            if not _is_aggstream_result(prev_agg_res):
                raise ValueError(
                    f"provided '{key}' data is not an " "AggStream result.",
                )
            aggstream_md = prev_agg_res._oups_metadata[KEY_AGGSTREAM]
            # - 'last_seed_index' to trim accordingly head of seed data.
            # - metadata related to binning process from past binnings
            # on prior data. It is used in case 'bin_by' is a callable.
            # If not used, it is an empty dict.
            # - metadata related to post-processing of prior
            # aggregation results, to be used by 'post'. If not used,
            # it is an empty dict.
            seed_index_restart_set.add(aggstream_md[KEY_RESTART_INDEX])
            agg_buffers[key_as_str][KEY_SEGAGG_BUFFER] = (
                aggstream_md[KEY_SEGAGG_BUFFER] if aggstream_md[KEY_SEGAGG_BUFFER] else {}
            )
            agg_buffers[key_as_str][KEY_POST_BUFFER] = (
                aggstream_md[KEY_POST_BUFFER] if aggstream_md[KEY_POST_BUFFER] else {}
            )
        else:
            agg_buffers[key_as_str][KEY_SEGAGG_BUFFER] = {}
            agg_buffers[key_as_str][KEY_POST_BUFFER] = {}

    if len(seed_index_restart_set) > 1:
        raise ValueError(
            "not possible to aggregate on multiple keys with existing"
            " aggregation results not aggregated up to the same seed index.",
        )
    return (
        None if not seed_index_restart_set else seed_index_restart_set.pop(),
        agg_buffers,
    )


def _reset_agg_buffers(agg_buffers: Optional[dict] = None) -> Optional[dict]:
    """
    Reset aggregation buffers and counters.

    Either modify in-place, or return a new dict.

    Parameters
    ----------
    agg_buffers : Optional[dict], default None
        Buffer to keep track of aggregation sequence intermediate results.

        - n_rows : int, number of rows in main aggregation results (snapshots
          is snapshots are quested, or bins otherwise). It is reset here after
          writing.
        - agg_res : pDataFrame, last aggregation results, to reset to None
          after writing.
        - bin_res : pDataFrame, last aggregation results, to reset to None
          after writing.
        - agg_res_buffer : List[pDataFrame], list of chunks resulting from
          aggregation (pandas DataFrame), either from bins if only bins
          requested, or from snapshots if bins and snapshots requested.
        - bin_res_buffer : List[pandas.DataFrame], list of bins resulting from
          aggregation (pandas dataframes), when bins and snapshots are
          requested.
        - post_buffer : dict, buffer to keep track of data that can be
          processed during previous iterations. This pointer should not be
          re-initialized in 'post' or data from previous iterations will be
          lost. This dict has to contain data that can be serialized, as data
          is then kept in parquet file metadata.
        - segagg_buffer : dict, parameters from segmentation and aggregation
          process, that are required when restarting the aggregation with new
          seed data. (for recording in metadata of aggregation results)

    Returns
    -------
    dict
        A dict with initialized values for ``agg_buffers``.

    """
    init_values = {
        "agg_n_rows": 0,
        KEY_AGG_RES: None,
        KEY_BIN_RES: None,
        KEY_AGG_RES_BUFFER: [],
        KEY_BIN_RES_BUFFER: [],
    }
    if agg_buffers is None:
        return init_values
    else:
        agg_buffers |= init_values


class SeedCheckException(Exception):
    """
    Exception related to user-defined checks on seed chunk.
    """

    def __init__(self):
        """
        Exception message.
        """
        self.message = "failing used-defined checks."


def _iter_data(
    seed: Iterable[pDataFrame],
    ordered_on: str,
    restart_index: Union[int, float, pTimestamp, None],
    check: Union[Callable, None],
    check_buffer: dict,
    filters: Union[dict, None],
    trim_start: bool,
    discard_last: bool,
) -> Tuple[str, pDataFrame]:
    """
    Iterate provided seed, applying sequentially (optionally) filters.

    Parameters
    ----------
    seed : Iterable[pDataFrame]
        Iterable of pandas Dataframe.
    ordered_on : str
        Name of column with respect to which seed data is in ascending
        order.
    restart_index : int, float, pTimestamp or None
        Index (excluded) in `ordered_on` column before which rows in seed
        will be trimmed.
    check : Callable or None
        Used-defined Callable to proceed checks over each item of the seed
        Iterable, accepting 2 parameters:

          - An ``on`` parameter, a pandas dataframe, the current seed item
            (before any filter is applied).
          - A ``buffer`` parameter, a dict that can be used as a buffer
            for storing temporary results from one chunk processing to
            the next. Its initial value is that provided by `check.buffer`.

    check_buffer : dict or None
        Buffer to keep track of intermediate data that can be required for
        proceeding with check of individual seed item.
    filters : dict or None
        Dict in the form
        ``{"filter_id":[[("col", op, val), ...], ...]}``
        To filter out data from seed.
        Filter syntax: [[(column, op, val), ...],...]
        where op is [==, =, >, >=, <, <=, !=, in, not in]
        The innermost tuples are transposed into a set of filters applied
        through an `AND` operation.
        The outer list combines these sets of filters through an `OR`
        operation.
        A single list of tuples can also be used, meaning that no `OR`
        operation between set of filters is to be conducted.
    trim_start : bool
        Flag to indicate if seed head has to be trimmed.
    discard_last : bool
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step.

    Returns
    -------
    last_seed_index, filder_id, filtered_chunk
        - 'last_seed_index', Union[int, float, pTimestamp], the last seed
          index value (likely of an incomplete group), of the current seed
          chunk, before filters are applied.
        - 'filter_id', str, indicating which set of filters has been
          applied for the seed chunk provided.
        - 'filtered_chunk', pDataFrame, from the seed Iterable, with
          optionally filters applied.

    Notes
    -----
    Checks are applied after having trimming seed head (if ``trim_start``
    is True) and discard last row group (if ``discard_last`` is True).

    Reasons to discard last seed row (or row group) may be twofold:
      - last row is temporary (yet to get some final values, for instance
        if seed data is some kind of aggregation stream itself),
      - last rows are part of a single row group not yet complete itself
        (new rows part of this row group to be expected).

    """
    if restart_index is None:
        # No aggregation result existing yet. Whatever 'trim_start' value, no
        # trimming is possible.
        trim_start = False
    seed_remainder = None
    for seed_chunk in seed:
        # Check seed chunk is ordered on 'ordered_on', else order.
        # if not seed_chunk[ordered_on].is_monotonic_increasing:
        # This re-ordering is made because for 'trim_start' and
        # 'discard_last', this ordereding is required.
        # seed_chunk.sort_values(by=ordered_on, inplace=True)
        # Step 1 / Seed check by user.
        if check:
            # Apply user checks.
            try:
                check(seed_chunk, check_buffer)
            except Exception:
                # Stop iteration in case of failing check.
                # Aggregation has been run up to the last valid chunk.
                raise SeedCheckException()
        # Step 2 / If a previous remainder, concatenate it to give current
        # DataFrame its 'final' length.
        if not (seed_remainder is None or seed_remainder.empty):
            seed_chunk = pconcat([seed_remainder, seed_chunk], ignore_index=True)
        # Step 3 / Prepare filter to trim seed head and tail if requested.
        if trim_start:
            if seed_chunk.loc[:, ordered_on].iloc[-1] < restart_index:
                # Do not process this chunk.
                continue
            else:
                filter_array = seed_chunk[ordered_on] >= restart_index
                # Once it has been applied once, no need to check for it
                # again on subsequent chunks.
                trim_start = False
        else:
            filter_array = ones(len(seed_chunk), dtype=bool)
        # 'ordered_on' being necessarily in ascending order, last index
        # value is its max value.
        last_seed_index = seed_chunk.loc[:, ordered_on].iloc[-1]
        if discard_last:
            filter_main_chunk = seed_chunk.loc[:, ordered_on] < last_seed_index
            seed_remainder = seed_chunk.loc[~filter_main_chunk]
            filter_array &= filter_main_chunk
        if not filter_array.any():
            # DataFrame will be empty after filtering.
            # Proceed with next iteration.
            continue
        # Step 4 / Filter seed and yield.
        for filt_id, filters_ in filters.items():
            # Filter.
            if filt_id != NO_FILTER_ID:
                filter_array &= dataframe_filter(seed_chunk, filters_)
                if not filter_array.any():
                    # DataFrame will be empty after filtering.
                    # Proceed with next iteration.
                    continue
            if filter_array.all():
                yield last_seed_index, filt_id, seed_chunk
            else:
                filtered_chunk = seed_chunk.loc[filter_array]
                filtered_chunk.reset_index(drop=True, inplace=True)
                yield last_seed_index, filt_id, filtered_chunk


def _post_n_write_agg_chunks(
    agg_buffers: dict,
    append_last_res: bool,
    dirpath: str,
    key: str,
    write_config: dict,
    index_name: Optional[str] = None,
    post: Optional[Callable] = None,
    last_seed_index: Optional[Union[int, float, pTimestamp]] = None,
):
    """
    Write list of aggregation row groups with optional post.

    Buffer variables 'agg_res_buffer', 'bin_res_buffer' are then reset.

    Parameters
    ----------
    agg_buffers : dict
        Buffer to keep track of aggregation sequence intermediate results.

        - n_rows : int, number of rows in main aggregation results (snapshots
          is snapshots are quested, or bins otherwise). It is reset here after
          writing.
        - agg_res : pDataFrame, last aggregation results, to reset to None
          after writing.
        - bin_res : pDataFrame, last aggregation results, to reset to None
          after writing.
        - agg_res_buffer : List[pDataFrame], list of chunks resulting from
          aggregation (pandas DataFrame), either from bins if only bins
          requested, or from snapshots if bins and snapshots requested.
          It is flushed here after writing
        - bin_res_buffer : List[pandas.DataFrame], list of bins resulting from
          aggregation (pandas dataframes), when bins and snapshots are
          requested.
          It is flushed here after writing
        - post_buffer : dict, buffer to keep track of data that can be
          processed during previous iterations. This pointer should not be
          re-initialized in 'post' or data from previous iterations will be
          lost. This dict has to contain data that can be serialized, as data
          is then kept in parquet file metadata.
          It is NOT reset after writing. It is however required to be
          written in metadata.
        - segagg_buffer : dict, parameters from segmentation and aggregation
          process, that are required when restarting the aggregation with new
          seed data. (for recording in metadata of aggregation results)
          It is NOT reset after writing. It is however required to be
          written in metadata.

    append_last_res : bool
        If 'agg_res' should be appended to 'agg_res_buffer' and if 'bin_res'
        should be appended to 'bin_res_buffers'.
    dirpath : str
        Path to which recording aggregation results.
    key : str
        Key for retrieving corresponding metadata.
    write_config : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Compulsory parameter defining at least `ordered_on`
        and `duplicates_on` columns.
    index_name : str, default None
        If a string, index name of dataframe resulting from aggregation with
        this value, which will be enforced in written results.
    post : Callable, default None
        User-defined function accepting 3 parameters.

          - ``buffer``, a dict to be used as data buffer, that can be necessary
            for some user-defined post-processing requiring data assessed in
            previous post-processing iteration.
          - ``bin_res``, a pandas dataframe resulting from the aggregations
            defined by ``agg`` parameter, with first row already corrected
            with last row of previous streamed aggregation.
            These are aggregation results for bins.
          - ``snap_res`` (optional), a pandas dataframe resulting from the
            aggregations defined by ``agg`` parameter that contains snapshots.

        It has then to return a pandas dataframe that will be recorded.
        This optional post-processing is intended for use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), or dataframe formatting before results are finally recorded.

    last_seed_index : Union[int, float, pTimestamp], default None
        Last index in seed data. Can be numeric type, timestamp... (for
        recording in metadata of aggregation results)
        Writing metadata is triggered ONLY if ``last_seed_index`` is provided.

    """
    # Concat list of aggregation results.
    # TODO: factorize these 2 x 5 rows of code below, for bin res and snap res ...
    # 'agg_res_buffer'
    agg_res_buffer = (
        [*agg_buffers[KEY_AGG_RES_BUFFER], agg_buffers[KEY_AGG_RES]]
        if append_last_res
        else agg_buffers[KEY_AGG_RES_BUFFER]
    )
    # Make a copy when a single item to not propagate to original 'agg_res'
    # the 'reset_index'.
    agg_res = (
        pconcat(agg_res_buffer) if len(agg_res_buffer) > 1 else agg_res_buffer[0].copy(deep=False)
    )
    if index_name:
        # In case 'by' is a callable, index may have no name, but user may have
        # defined one with 'bin_on' parameter.
        agg_res.index.name = index_name
    # Keep group keys as a column before post-processing.
    agg_res.reset_index(inplace=True)
    # 'bin_res_buffer'
    bin_res = agg_buffers[KEY_BIN_RES]
    if bin_res is not None:
        bin_res_buffer = (
            [*agg_buffers[KEY_BIN_RES_BUFFER], bin_res]
            if append_last_res
            else agg_buffers[KEY_BIN_RES_BUFFER]
        )
        bin_res = (
            pconcat(bin_res_buffer)
            if len(bin_res_buffer) > 1
            else bin_res_buffer[0].copy(deep=False)
        )
        if index_name:
            # In case 'by' is a callable, index may have no name, but user may
            # have defined one with 'bin_on' parameter.
            bin_res.index.name = index_name
        bin_res.reset_index(inplace=True)
    post_buffer = agg_buffers[KEY_POST_BUFFER]
    if post:
        # Post processing if any.
        # 'post_buffer' has to be modified in-place.
        agg_res = (
            post(buffer=post_buffer, bin_res=agg_res)
            if bin_res is None
            else post(buffer=post_buffer, bin_res=bin_res, snap_res=agg_res)
        )
    if last_seed_index:
        # If 'last_seed_index', set oups metadata.
        OUPS_METADATA[key] = {
            KEY_AGGSTREAM: {
                KEY_RESTART_INDEX: last_seed_index,
                KEY_SEGAGG_BUFFER: agg_buffers[KEY_SEGAGG_BUFFER],
                KEY_POST_BUFFER: post_buffer,
            },
        }
    # Record data.
    write(dirpath=dirpath, data=agg_res, md_key=key, **write_config)
    # Reset aggregation buffers and counters.
    _reset_agg_buffers(agg_buffers)


def agg_iter(
    seed_chunk: pDataFrame,
    key: str,
    keys_config: dict,
    agg_config: dict,
    agg_buffers: dict,
):
    """
    Post-process and write iter. n-1, segment and aggregate iter. n.

    Parameters
    ----------
    seed_chunk : pDataFrame
        Chunk of seed data.
    key : str
        Key for recording aggregation results.
    keys_config
        Settings related to 'key' for conducting post-processing, writing and
        segmentation.
    agg_config : dict
        Settings related to 'key' for conducting aggregation.
    agg_buffers : dict
        Buffer to keep track of aggregation sequence intermediate results.

    Returns
    -------
    key, updated_agg_buffers
        - ``key``, key to which changed parameters are related.
        - ``updated_agg_buffers``, dict with modified parameters.

    """
    # Post process and write.
    if (agg_res := agg_buffers[KEY_AGG_RES]) is not None:
        # If previous results, check if this is write time.
        # Retrieve length of aggregation results.
        agg_res_len = len(agg_res)
        agg_res_buffer = agg_buffers[KEY_AGG_RES_BUFFER]
        if agg_res_len > 1:
            # Remove last row from 'agg_res' and add to
            # 'agg_res_buffer'.
            agg_res_buffer.append(agg_res.iloc[:-1])
            # Remove last row that is not recorded from total number of rows.
            agg_buffers["agg_n_rows"] += agg_res_len - 1
            agg_n_rows = agg_buffers["agg_n_rows"]
            if (bin_res := agg_buffers[KEY_BIN_RES]) is not None:
                # If we have bins & snapshots, do same with bins.
                agg_buffers[KEY_BIN_RES_BUFFER].append(bin_res.iloc[:-1])
            # Keep floor part.
            if agg_n_rows:
                # Length of 'agg_res_buffer' is number of times it has been
                # appended.
                agg_mean_row_group_size = agg_n_rows // len(agg_res_buffer)
                if (
                    agg_n_rows + agg_mean_row_group_size
                    > keys_config["write_config"][KEY_MAX_ROW_GROUP_SIZE]
                ):
                    # Write results from previous iteration.
                    _post_n_write_agg_chunks(
                        agg_buffers=agg_buffers,
                        append_last_res=False,
                        dirpath=keys_config["dirpath"],
                        key=key,
                        write_config=keys_config["write_config"],
                        index_name=keys_config[KEY_BIN_ON_OUT],
                        post=keys_config[KEY_POST],
                    )
    # Segment and aggregate. Group keys becomes the index.
    agg_res = cumsegagg(
        data=seed_chunk,
        agg=agg_config,
        bin_by=keys_config["seg_config"],
        buffer=agg_buffers[KEY_SEGAGG_BUFFER],
    )
    # 'agg_res' is 'main' aggregation results onto which are assessed
    # 'everything'. If only 'bins' are requested, it gathers bins.
    # If 'bins' and 'snapshots' are requested, it gathers snapshots.
    agg_buffers[KEY_BIN_RES], agg_buffers[KEY_AGG_RES] = (
        agg_res if isinstance(agg_res, tuple) else None,
        agg_res,
    )
    return key, agg_buffers


class AggStream:
    """
    Persist configuration data to run aggregation in sequence.

    Attributes
    ----------
      - ``self.seed_config`, a dict keeping track of seed-related parameters.
        ``{'ordered_on' : string, specifying column name in seed data in
                          ascending order.
           'restart_index' : int, float or pTimestamp, the index from which
                             (included) should be restarted the next
                             aggregation iteration.
           'check' : Callable, to apply user-defined check on each seed item.
           'check_buffer' : dict, to keep track of intermediate values for
                           `check` function.
           'filters' : dict, as per `filters` parameter.
          }``
      - ``self.store``, oups store, as per `store` parameter.
      - ``self.agg_pd``, dict, as per `agg` parameter, in pandas format.
      - ``self.agg_cs``, an attribute initialized once an aggregation
        iteration has been run, and defining aggregation in `cumsegagg`
        standard. It is initialized in ``self.agg`` function, the 1st time
        an aggregation is run (seed data dtypes is required).
      - ``self.filter_ids_to_keys``, dict, mapping filter ids to list of
        keys.
      - ``self.min_number_of_keys_per_filter``, int, minimal number of keys
        for a given filter set, among all filter sets.
      - ``self.keys_config``, dict of keys config in the form:
        ``{key: {'dirpath': str, where to record agg res,
                 'bin_on_out' : str, name in aggregation results for column
                                with bin ids.
                 'seg_config' : dict specifying the segmentation config,
                 'post' : Callable or None,
                 'write_config' : {'ordered_on' : str,
                                   'duplicates_on' : str or list,
                                   ...
                                  },
               },
         }``
        To be noticed, keys of dict ``keys_config`` are strings.
      - ``self.agg_buffers``, dict to keep track of aggregation iteration
                              intermediate results.
        ``{key: {'agg_n_rows' : 0,
                 'agg_res' : None,
                 'bin_res' : None,
                 'agg_res_buffer' : list, agg_res_buffer,
                 'bin_res_buffer' : list, bin_res_buffer,
                 'segagg_buffer' : dict, possibly empty,
                 'post_buffer' : dict, possibly empty,
               },
         }``
        To be noticed, keys of dict ``agg_buffers`` are strings.
      - ``self.all_keys``, list of keys, kept as Indexer. This attribute is to
        be removed if key can be kept as an Indexer in ``self.keys_config``.

    """

    def __init__(
        self,
        ordered_on: str,
        store: ParquetSet,
        keys: Union[dataclass, dict],
        check: Optional[Callable] = None,
        check_buffer: Optional[dict] = None,
        filters: Optional[dict] = None,
        agg: Optional[dict] = None,
        bin_by: Optional[Union[TimeGrouper, Callable[[Series, dict], tuple]]] = None,
        bin_on: Optional[Union[str, Tuple[str, str]]] = None,
        snap_by: Optional[Union[TimeGrouper, Series, DatetimeIndex]] = None,
        post: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Initialize aggregation stream on ordered data.

        This object enables 'streamed aggregation', iteratively
        (out-of core) with optional filtering of seed data, and optional
        post-processing of aggregation results (by use of vectorized functions
        or for dataframe formatting).
        Aggregation results are recoreded into a 'oups store'.

        Parameters
        ----------
        ordered_on : str
            Name of the column with respect to which seed dataset is in
            ascending order. While this parameter is compulsory for correct
            restart on seed data, seed data is not necessarily grouped by this
            column. ``bin_by`` and/or ``bin_on`` parameters can be used to
            define such a different parameter.
            This value is also used as default 'ordered_on' parameter for
            aggregation results, if not provided separately for each key.
        store : ParquetSet
            Store to which recording aggregation results.
        keys : Union[Indexer, dict]
            Key(s) for recording aggregation results.
            If a dict, several keys can be specified for operating multiple
            parallel aggregations on the same seed. In this case, the dict can
            be of two forms.

              - In case seed data is not to be filtered, it should be in the
                form 1, defined as:
                ``{key: {'agg': agg,
                         'bin_by': bin_by,
                         'bin_on': bin_on,
                         'snap_by': snap_by,
                         'post': post,
                         **kwargs}
                   }``
                Any additional parameters, (``**kwargs``) are forwarded to
                ``oups.writer.write`` when writing aggregation results to
                store, such as custom `max_row_group_size`, 'duplicates_on' or
                'ordered_on' parameters (see not below for 'duplicates_on').
                Please, note:

                  - `bin_by` is a compulsory parameter.
                  - If not specified, `bin_on` parameter in dict does not get
                    default values.
                  - If not specified in dict, `agg`, `snap_by`, `post` and
                    other parameters related to writing of aggregation
                    results... get values from `agg`, `snap_by`, `post`,
                    `ordered_on` and `**kwargs` parameters defined when
                    initializing `AggStream`.
                    If using `snap_by` or `post` when initializing `AggStream`
                    and not willing to apply it for one key, set it to ``None``
                    in key specific config.

              - In case seed is to be filtered, dict written in form 1 are
                themselves values within an upper dict. Keys for this upper
                dict are string used as filter id. Each of these filter ids
                have then to be listed in ``filters`` parameter.
                For keys deriving from unfiltered data, use the `NO_FILTER_ID`
                ``"_"``.

        check : Callable, default None
            Used-defined Callable to proceed checks over each item of the seed
            Iterable, accepting 2 parameters:

              - An ``on`` parameter, a pandas dataframe, the current seed item
                (before any filter is applied).
              - A ``buffer`` parameter, a dict that can be used as a buffer
                for storing temporary results from one chunk processing to
                the next. Its initial value is that provided by `check.buffer`.

            If running ``check`` raises an exception (whichever type it is), a
            ``SeedCheckException`` will subsequently be raised.

        check_buffer : dict, default None
            Buffer to keep track of intermediate data that can be required for
            proceeding with check of individual seed item.
            Once aggregation stream is over, its value is not recorded.
            User has to take care of this if needed. Its value can be
            retrieved with ``self.check_buffer`` object attribute.
        filters : Union[dict, None], default None
            Dict in the form
            ``{"filter_id":[[("col", op, val), ...], ...]}``
            To filter out data from seed.
            Filter syntax: [[(column, op, val), ...],...]
            where op is [==, =, >, >=, <, <=, !=, in, not in]
            The innermost tuples are transposed into a set of filters applied
            through an `AND` operation.
            The outer list combines these sets of filters through an `OR`
            operation.
            A single list of tuples can also be used, meaning that no `OR`
            operation between set of filters is to be conducted.
        agg : Union[dict, None], default None
            Dict in the form
            ``{"output_col":("input_col", "agg_function_name")}``
            where keys are names of output columns into which are recorded
            results of aggregations, and values describe the aggregations to
            operate. ``input_col`` has to exist in seed data.
            Examples of ``agg_function_name`` are `first`, `last`, `min`, `max`
            and `sum`.
            This parameter is compulsory, except if ``key`` parameter is a
            `dict`.
        bin_by : Union[TimeGrouper, Callable[[pd.DataFrame, dict]]], default None
            Parameter defining the binning logic.
            If a `Callable`, it is given following parameters.

              - An ``on`` parameter, a pandas dataframe made of column
                ``ordered_on``, and column ``bin_on`` if different than
                ``ordered_on``.
              - A ``buffer`` parameter, a dict that can be used as a buffer for
                storing temporary results from one chunk processing to
                the next.

            TThis parameter is the ``bin_by`` parameter of
            ``oups.aggstream.segmentby.segmentby`` function. For more
            information, please, read its docstring.
        bin_on : Union[str, Tuple[str, str]], default None
            ``bin_on`` may either be a string or a tuple of 2 string. When a
            string, it refers to an existing column in seed data onto which
            applying the binning defined by ``bin_by`` parameter. Its value is
            then carried over as name for the column containing the group keys.
            It is further used when writing results for defining
            ``duplicates_on`` parameter (see ``oups.writer.write``).
            When a tuple, the 1st string refers to an existing column in seed
            data, the 2nd the name to use for the column which values will be
            the group keys in aggregation results.
            Setting of ``bin_on`` should be adapted depending how is defined
            ``bin_by`` parameter. When ``bin_by`` is a Callable, then
            ``bin_on`` can have different values.

              - ``None``, the default.
              - the name of an existing column onto which applying the binning.
                Its value is then carried over as name for the column
                containing the group keys.

        snap_by : Optional[Union[TimeGrouper, Series, DatetimeIndex]], default None
            Values positioning points of observation, either derived from a
            pandas TimeGrouper, or contained in a pandas Series.
            In case 'snap_by' is a Series, values serve as locations for points
            of observation.
            Additionally, ``closed`` value defined by 'bin_on' specifies if
            points of observations are included or excluded.

              - `left`, then values at points of observation are excluded.
              - `right`, then values at points of observation are included.

        post : Callable, default None
            User-defined function accepting up to 3 parameters.

              - ``buffer``, a dict to be used as data buffer, that can be
                necessary for some user-defined post-processing requiring data
                assessed in previous post-processing iteration.
              - ``bin_res``, a pandas dataframe resulting from the aggregations
                defined by ``agg`` parameter, with first row already corrected
                with last row of previous streamed aggregation.
                These are aggregation results for bins.
              - ``snap_res`` (optional), a pandas dataframe resulting from the
                aggregations defined by ``agg`` parameter that contains
                snapshots.

            It has then to return a pandas dataframe that will be recorded.
            This optional post-processing is intended for use of vectorized
            functions (not mixing rows together, but operating on one or
            several columns), or dataframe formatting before results are
            finally recorded.

        Other Parameters
        ----------------
        kwargs : dict
            Settings forwarded to ``oups.writer.write`` when writing
            aggregation results to store. Can define for instance custom
            `max_row_group_size` or `duplicates_on` parameters (see notes below
            for `duplicates_on`).

        Notes
        -----
        - Result is necessarily added to a dataset from an instantiated oups
          ``ParquetSet``. ``AggStream`` actually relies on the update feature
          from oups.
        - If ``kwargs`` defines a maximum row group size to write to disk, this
          value is also used to define a maximum size of aggregation results
          before actually triggering a write. If no maximum row group size is
          defined, then ``MAX_ROW_GROUP_SIZE`` defined in ``oups.writer.write``
          is used.
        - With the post-processing step, user can also take care of removing
          columns produced by the aggregation step, but not needed afterwards.
          Other formatting operations on the dataframe can also be achieved
          (renaming columns or index, and so on...). To be noticed, group keys
          are available through a column having same name as initial column
          from seed data, or defined by 'bin_on' parameter if 'bin_by' is a
          Callable.
        - When recording, both 'ordered_on' and 'duplicates_on' parameters are
          set when calling ``oups.writer.write``. If additional parameters are
          defined by the user, some checks are made.

            - 'ordered_on' is forced to 'AggStream' ``ordered_on`` parameter.
            - If 'duplicates_on' is not set by the user or is `None`, then it
              is

                - either set to the name of the output column for group keys
                  defined by `bin_on` if `bin_on` is set. The rational is that
                  this column identifies uniquely each bin, and so is a
                  relevant column to identify duplicates.
                - if `bin_on` is not set, then it defaults to `ordered_on`
                  column.

              There might case when this logic is unsuited. For instance,
              perhaps values in 'ordered_on' column does provide a unique valid
              identifier for bins already (if there are unique values in
              'ordered_on'). It may then be that the column containing group
              keys is removed during user post-processing.
              To allow such specific use case, the user can set
              ``duplicates_on`` as additional parameter to ``AggStream``. If
              the user omit a column name, it means that this is a voluntary
              choice from the user.

        - If an exception is raised by ``check`` function on seed data, then,
          last good results are still written to disk with correct metadata. If
          an exception is raised at some other point of the aggregation
          process, results are not written.

        """
        # Check 'kwargs' parameters are those expected for 'write' function.
        for param in kwargs:
            if param not in WRITE_PARAMS:
                raise ValueError(
                    f"'{param}' is neither a valid parameter for `AggStream`"
                    " initialization, nor for `oups.write` function.",
                )
        # Seed-related attributes.
        if filters is not None:
            # Check if only an "AND" part has been provided. If yes, enclose it
            # in an outer list.
            filters = {
                filt_id: [filters_] if isinstance(filters_[0], tuple) else filters_
                for filt_id, filters_ in filters.items()
            }
        # Store and keys related attributes.
        self.store = store
        # Set default values for keys' config.
        keys_default = {
            KEY_SNAP_BY: snap_by,
            KEY_AGG: agg,
            KEY_POST: post,
            KEY_ORDERED_ON: ordered_on,
        } | kwargs
        if not isinstance(keys, dict):
            keys = {keys: keys_default | {KEY_BIN_BY: bin_by, KEY_BIN_ON: bin_on}}
        if is_toplevel(next(iter(keys))):
            # Case no filter is used.
            keys = {NO_FILTER_ID: keys}
            filters = {NO_FILTER_ID: None}
        else:
            # Check 'filters' parameter is used.
            if filters is None:
                raise ValueError(
                    "not possible to use filter syntax for `keys` parameter "
                    "without providing `filters` parameter as well.",
                )
            else:
                # Check same filters id are both in 'keys' and 'filters'
                # parameters.
                if NO_FILTER_ID in filters and filters[NO_FILTER_ID] is not None:
                    raise ValueError(
                        f"not possible to use '{NO_FILTER_ID}' as key in "
                        "`filters` parameter with a value different than "
                        "`None`.",
                    )
                filt_filt_ids = set(filters)
                filt_filt_ids.discard(NO_FILTER_ID)
                if filt_filt_ids != (keys_filt_ids := set(keys)):
                    raise ValueError(
                        "not possible to have different lists of filter ids"
                        " between `keys` and `filters` parameters.\n"
                        f" List of filter ids in `keys` parameter is {keys_filt_ids}.\n"
                        f" List of filter ids in `filters` parameter is {filt_filt_ids}.",
                    )
        _filter_ids_to_keys = {}
        _min_number_of_keys_per_filter = []
        _all_keys = []
        for filt_id in keys:
            # Keep keys as str.
            _filter_ids_to_keys[filt_id] = list(map(str, keys[filt_id]))
            _min_number_of_keys_per_filter.append(len(keys[filt_id]))
            # TODO: remove '_all_keys' if key can be serialize.
            _all_keys.extend(keys[filt_id])
        self.all_keys = _all_keys
        self.filter_ids_to_keys = _filter_ids_to_keys
        # Once filters have been managed, simplify 'keys' as a single level
        # dict.
        keys = ChainMap(*keys.values())
        self.min_number_of_keys_per_filter = min(_min_number_of_keys_per_filter)
        (
            self.keys_config,
            self.agg_pd,
        ) = _init_keys_config(ordered_on, store, keys, keys_default)
        (
            restart_index,
            self.agg_buffers,
        ) = _init_agg_buffers(store, keys)
        self.seed_config = {
            KEY_ORDERED_ON: ordered_on,
            KEY_CHECK: check,
            KEY_CHECK_BUFFER: {} if check_buffer is None else check_buffer,
            KEY_FILTERS: filters,
            KEY_RESTART_INDEX: restart_index,
        }
        # Cumsegagg-like agg definition.
        # Cannot be set yet, because seed dtype is required.
        # Is a dict, specifying for each key, its expected aggregation.
        self.agg_cs = {}

    def _init_agg_cs(self, seed: Iterable[pDataFrame]):
        """
        Initialize ``self.agg_cs``.

        Because dtypes of seed DataFrame is required, the first seed chunk is
        generated from the Iterable. Seed Iterable is then repacked with first
        item already in memory.

        Parameters
        ----------
        seed : Iterable[pDataFrame]
            Seed data, from which getting pandas DataFrame dtypes.

        Returns
        -------
        seed
            Seed that had to be repacked.

        """
        remainder = iter(seed)
        first = next(remainder)
        # Recompose seed with 1st item materialized.
        seed = chain([first], remainder)
        seed_dtypes = first.dtypes.to_dict()
        for key in self.keys_config:
            try:
                self.agg_cs[key] = setup_cumsegagg(self.agg_pd[key], seed_dtypes)
            except Exception:
                raise ValueError(f"exception raised for key '{key}'")
        return seed

    def agg(
        self,
        seed: Union[pDataFrame, Iterable[pDataFrame]],
        trim_start: Optional[bool] = True,
        discard_last: Optional[bool] = True,
        final_write: Optional[bool] = True,
        parallel: Optional[bool] = False,
    ):
        """
        Aggregate sequentially on successive chunks (stream) of ordered data.

        This function conducts 'streamed aggregation', iteratively (out-of
        core) with optional post-processing of aggregation results (by use of
        vectorized functions or for dataframe formatting).

        Parameters
        ----------
        seed : Union[pDataFrame, Iterable[pDataFrame]]
            Seed data over which conducting streamed aggregations.
        trim_start : bool, default True
            If ``True``, and if aggregated results already exist, then
            retrieves the first index from seed data not processed yet
            (recorded in metadata of existing aggregated results), and trim all
            seed data before this index (index excluded from trim).
        discard_last : bool, default True
            If ``True``, last row group in seed data (sharing the same value in
            `ordered_on` column) is removed from the aggregation step. See
            below notes.
        final_write : bool, default True
            If ``True``, after last iteration of aggregation, aggregation
            results are written to disk.
            If ``False``, last aggregation results remain set in memory,
            available for a next iteration.
        parallel : bool, default False
            Conduct processing of keys in parallel, with one process per `key`.
            If a single `key`, only one process is possible.

        Notes
        -----
        - If aggregation results already exist in oups ``ParquetSet`` instance,
          and `trim_start` is `True`, last index from previous aggregation is
          retrieved, and prior seed data is trimmed.
        - Aggregation is by default processed up to the last index excluded,
          and subsequent aggregation will start from this last index included,
          assumed to be that of an incomplete row group.
          If `discard_last` is set `False`, then aggregation is process up to the
          last data.
        - By default, with parameter `discard_last`` set ``True``, the last row
          group (composed from rows sharing the same value in `ordered_on` column),
          is discarded.

            - It may be for instance that this row group is not complete yet
              and should therefore not be accounted for. More precisely, new
              rows with same value in `ordered_on` may appear in seed data
              later on. Because seed data is trimmed to start from last
              processed value from `ordered_on` column (value included), these
              new rows would be excluded from the next aggregation, leading to
              an inaccurate aggregation result. Doing so is a way to identify
              easily when re-starting the aggregation in a case there can be
              duplicates in `ordered_on` column. A ``sum`` aggregation will
              then return the correct result for instance, as no data is
              accounted for twice.
            - Or if composed of a single row, this last row in seed data is
              temporary (and may get its final values only at a later time,
              when it becomes the one-but-last row, as a new row is added).

        """
        if isinstance(seed, pDataFrame):
            # Make the seed an iterable.
            seed = [seed]
        if not self.agg_cs:
            # If first time an aggregation is made with this object,
            # initialize 'agg_cs'.
            seed = self._init_agg_cs(seed)
        # Per filter set, take min number of keys.
        n_keys = self.min_number_of_keys_per_filter
        n_jobs = min(int(cpu_count() * 3 / 4), n_keys) if (parallel and n_keys > 1) else 1
        seed_check_exception = False
        with Parallel(n_jobs=n_jobs, prefer="threads") as p_job:
            try:
                for _last_seed_index, filter_id, filtered_chunk in _iter_data(
                    seed=seed,
                    **self.seed_config,
                    trim_start=trim_start,
                    discard_last=discard_last,
                ):
                    agg_loop_res = p_job(
                        delayed(agg_iter)(
                            seed_chunk=filtered_chunk,
                            key=key,
                            keys_config=self.keys_config[key],
                            agg_config=self.agg_cs[key],
                            agg_buffers=self.agg_buffers[key],
                        )
                        for key in self.filter_ids_to_keys[filter_id]
                    )
                    for key, agg_res in agg_loop_res:
                        self.agg_buffers[key].update(agg_res)
                    # Set 'seed_index_restart' to the 'last_seed_index' with
                    # which restarting the next aggregation iteration.
                    self.seed_config[KEY_RESTART_INDEX] = _last_seed_index
            except SeedCheckException:
                seed_check_exception = True
            # Check if at least one iteration has been achieved or not.
            agg_res = next(iter(self.agg_buffers.values()))[KEY_AGG_RES]
            if agg_res is None:
                # No iteration has been achieved, as no data.
                return
            if final_write:
                # Post-process & write results from last iteration, this time
                # keeping last aggregation row, and recording metadata for a
                # future 'AggStream.agg' execution.
                p_job(
                    delayed(_post_n_write_agg_chunks)(
                        key=key,
                        dirpath=self.keys_config[key]["dirpath"],
                        agg_buffers=agg_res,
                        append_last_res=True,
                        write_config=self.keys_config[key]["write_config"],
                        index_name=self.keys_config[key][KEY_BIN_ON_OUT],
                        post=self.keys_config[key][KEY_POST],
                        last_seed_index=self.seed_config[KEY_RESTART_INDEX],
                    )
                    for key, agg_res in self.agg_buffers.items()
                )
        # Add keys in store for those who where not in.
        # This is needed because cloudpickle is unable to serialize Indexer.
        # But dill can. Try to switch to dill?
        # https://joblib.readthedocs.io/en/latest/parallel.html#serialization-processes
        for key in self.all_keys:
            if key not in self.store:
                self.store._keys.add(key)
        if seed_check_exception:
            raise SeedCheckException()


# TODO:
# - Use dill to serialize 'keys' in joblib / if working:
#    - replace setting of dir_path in keys_config by store directly.
#    - remove 'self.all_keys'
# Should we persist / store 'p_job' between 'AggStream.agg' execution?

# Tests:
# - do a test using only "no filter id" "_" with another filter and check no filter
#   is used indeed.
# - in test case with snapshot: when snapshot is a TimeGrouper, make sure that stitching
# works same as for bin: that empty snapshots are generated between 2 row groups.
# - test case, test parameter value not in 'streamagg' nor in 'write' signature.
# - un test snap + bin avec deux ou trois chunks dans lesquels il n'y a ni snap, ni bin qui se finissent.
# - bien faire un cas test snapshot ou le 2nd seed chunk démarre sur une nouvelle bin:
#    straight away / ça pose des problèmes quand c'est simplement bin,
#    alors bin+snapshot, il y a des chances que ça ne marche pas non plus.
# - Do a test case to check that if in 'post' an exception is raised for instance,
#    then values in class are properly set: segagg_buffer, post_buffer, seed_index_restart,
#    and agg_res_buffer and bin_res_buffer are empty lists.
# - Test new parameters: 'final_write' and seed check exception
#    for seed check exception, check the last '_last_seed_index' has been correctly recorded
#    and aggregation results integrate results from last seed chunk.
# - A test that with a given filter release an empty dataframe.
# - A test with a check function that raises an exception: check that last seed index has been
#    that of the last iteration, prior to the exception.
# - A test for filter_dataframe
#    check when some chunk are empty for a given filter id
# - A test with first seed generator like Parquet File, then 1 dataframe each time.
# - Test with different 'ordered_on' values for a key vs seed
