#!/usr/bin/env python3
"""
Created on Wed Nov 15 21:30:00 2023.

@author: yoh

"""
from dataclasses import dataclass
from multiprocessing import cpu_count
from os import path as os_path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from fastparquet import ParquetFile
from joblib import Parallel
from joblib import delayed
from pandas import DataFrame as pDataFrame
from pandas import DatetimeIndex
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import concat as pconcat
from pandas.core.resample import TimeGrouper
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet
from oups.router import ParquetHandle
from oups.streamagg.cumsegagg import cumsegagg
from oups.streamagg.cumsegagg import setup_cumsegagg
from oups.streamagg.jcumsegagg import FIRST
from oups.streamagg.jcumsegagg import LAST
from oups.streamagg.jcumsegagg import MAX
from oups.streamagg.jcumsegagg import MIN
from oups.streamagg.jcumsegagg import SUM
from oups.streamagg.segmentby import KEY_BIN_BY
from oups.streamagg.segmentby import KEY_BIN_ON
from oups.streamagg.segmentby import KEY_ORDERED_ON
from oups.streamagg.segmentby import KEY_SNAP_BY
from oups.streamagg.segmentby import setup_segmentby
from oups.writer import DUPLICATES_ON
from oups.writer import KEY_MAX_ROW_GROUP_SIZE
from oups.writer import MAX_ROW_GROUP_SIZE
from oups.writer import OUPS_METADATA
from oups.writer import OUPS_METADATA_KEY
from oups.writer import write


VDATAFRAME_ROW_GROUP_SIZE = 6_345_000
# Aggregation functions.
ACCEPTED_AGG_FUNC = {FIRST, LAST, MIN, MAX, SUM}
# List of keys to metadata of aggregation results.
KEY_STREAMAGG = "streamagg"
KEY_LAST_SEED_INDEX = "last_seed_index"
KEY_SEGAGG_BUFFER = "segagg_buffer"
KEY_POST_BUFFER = "post_buffer"
KEY_BIN_ON_OUT = "bin_on_out"
# List of valid parameters for 'key_conf_in'
AGG = "agg"
POST = "post"
# 'bin_by' is a compulsory parameter, and a specific check is made for it.
# It is not added in 'KEY_CONF_IN_PARAMS'.
WRITE_PARAMS = set(write.__code__.co_varnames[: write.__code__.co_argcount])
KEY_CONF_IN_PARAMS = {KEY_BIN_ON, KEY_SNAP_BY, AGG, POST} | WRITE_PARAMS


def _is_streamagg_result(handle: ParquetHandle) -> bool:
    """
    Check if input handle is that of a dataset produced by streamaag.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file.

    Returns
    -------
    bool
        `True` if parquet file contains metadata as produced by
        ``oups.streamagg``, which confirms this dataset has been produced with
        this latter function.

    """
    # As oups specific metadata is a string produced by json library, the last
    # 'in' condition is checking if the set of characters defined by
    # 'STREAMAGG' is in a string.
    if OUPS_METADATA_KEY in handle.metadata:
        return KEY_STREAMAGG in handle._oups_metadata


def _setup(
    store: ParquetSet,
    keys: Dict[dataclass, dict],
    ordered_on: str,
    seed_dtypes: dict,
    trim_start: bool,
    **kwargs,
):
    """
    Consolidate settings for streamed (possibly parallelized) aggregations.

    Parameters
    ----------
    store : ParquetSet
        Store to which recording aggregation results.
    keys : Dict[dataclass, dict]
        Dict of keys for recording aggregation results in the form
        ``{key: {'agg': agg, 'bin_by': bin_by, 'bin_on': bin_on, 'post': post, **kwargs}}``
        `bin_on` is a compulsory parameter.
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order.
    seed_dtypes : dict
        Data types for each column of seed data.
    trim_start : bool
        Flag possibly modified to indicate if trimming seed head is possible
        or not

    Other Parameters
    ----------------
    kwargs : dict
        Settings considered as default ones if not specified within ``keys``.
        Default values for parameters related to aggregation can be set this
        way. (``agg``, ``snap_by`` & ``post``)
        Parameters related to writing data are added to ``write_config``, that
        is then forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Can define for instance custom `max_row_group_size`
        parameter.

    Returns
    -------
    tuple
        Settings for streamed aggregation.

          - ``all_cols_in``, list specifying all columns to be loaded from seed
            data, required to perform the aggregation.
          - ``trim_start``, bool indicating if seed head is to be trimmed or
            not.
          - ``seed_index_restart_set``, set of ``seed_index_restart`` from each
            keys, if aggregation results are already available.
          - ``keys_config``, dict of keys config. A config is also a dict in
            the form:
            ``{key: {'dirpath': str, where to record agg res,
                     'agg_n_rows' : 0,
                     'agg_mean_row_group_size' : 0,
                     'agg_res' : None,
                     'bin_res' : None,
                     'seg_config' : dict specifying the segmentation config,
                     'agg_config' : dict specifying the aggregation config,
                     'post' : Callable or None,
                     'agg_res_buffer' : agg_res_buffer,
                     'bin_res_buffer' : bin_res_buffer,
                     'segagg_buffer' : dict, possibly empty,
                     'post_buffer' : dict, possibly empty,
                     'write_config' : {'ordered_on' : str,
                                       'duplicates_on' : str or list,
                                       ...
                                       },
                     },
               }``
            To be noticed:

              - key of dict ``keys_config`` is a string.

    """
    keys_config = {}
    seed_index_restart_set = set()
    all_cols_in = {ordered_on}
    # Some default values for keys.
    # 'agg_n_rows' : number of rows in aggregation result.
    key_default = {
        "agg_n_rows": 0,
        "agg_mean_row_group_size": 0,
        "agg_res": None,
        "bin_res": None,
    }
    # Make a deep copy because of modifying the dictionary content in the loop.
    for key, key_conf_in in keys.items():
        # Parameters in 'key_conf_in' take precedence over those in 'kwargs'.
        # Additionally, with this step, 'key_conf_in' is a deep copy, and when
        # parameters are popped, it does not affect the initial 'key_conf_in'.
        key_conf_in = kwargs | key_conf_in
        try:
            bin_by = key_conf_in.pop(KEY_BIN_BY)
        except KeyError:
            raise ValueError(f"'{KEY_BIN_BY}' parameter is missing for key '{key}'.")
        # Check parameters in 'key_conf_in' are valid ones.
        for param in key_conf_in.keys():
            if param not in KEY_CONF_IN_PARAMS:
                raise ValueError(f"'{param}' not a valid parameters in '{key}' aggregation config.")
        # Step 1 / Process parameters.
        bin_on = key_conf_in.pop(KEY_BIN_ON, None)
        if isinstance(bin_on, tuple):
            # 'bin_on_out' is name of column containing group keys in 'agg_res'.
            # Setting of 'bin_on_out' is a 'streamagg' task, not a 'cumesegagg'
            # one. This is because this parameter clarifies then how to set
            # 'duplicates_on' parameter for 'oups.writer.write' which is also
            # part of 'streamagg' perimeter.
            bin_on, bin_on_out = bin_on
        else:
            bin_on_out = None
        # Step 1.1 / 'seg_conf': segmentation step.
        try:
            seg_config = setup_segmentby(
                bin_by=bin_by,
                bin_on=bin_on,
                ordered_on=ordered_on,
                snap_by=key_conf_in.pop(KEY_SNAP_BY, None),
            )
        except Exception:
            raise ValueError(f"error raised for key '{key}'")
        # Step 1.2 / 'all_cols_in'.
        # If 'bin_on' is defined, it is to be loaded as well.
        if bin_on := seg_config[KEY_BIN_ON]:
            all_cols_in.add(bin_on)
            if bin_on_out is None:
                # It may be that 'bin_on' value has been modified in
                # 'setup_segmentby'. If 'bin_on_out' has not been set
                # previously, then set it to this possibly new value of
                # 'bin_on'.
                bin_on_out = bin_on
        agg = key_conf_in.pop(AGG)
        # 'agg' is in the form:
        # {"output_col":("input_col", "agg_function_name")}
        if bin_on_out in agg:
            # Check that this name is not already that of an output column
            # from aggregation.
            raise ValueError(
                f"not possible to have {bin_on_out} as column name in"
                " aggregated results as it is also for column containing group"
                " keys.",
            )
        # Update 'all_cols_in', list of columns from seed to be loaded.
        all_cols_in.update({col_in for col_in, _ in agg.values()})
        # Step 1.3 / 'max_agg_row_group' and 'write_config'.
        # Initialize aggregation result max size before writing to disk.
        # If present, keep 'max_row_group_size' within 'key_conf_in' as it
        # is a parameter to be forwarded to the writer.
        if KEY_MAX_ROW_GROUP_SIZE not in key_conf_in.keys():
            key_conf_in[KEY_MAX_ROW_GROUP_SIZE] = MAX_ROW_GROUP_SIZE
        # Initialize 'write_config', which are parameters remaining in
        # 'key_conf_in' and some adjustments.
        # Forcing 'ordered_on' for write.
        key_conf_in[KEY_ORDERED_ON] = ordered_on
        # Adding 'bin_on_out' to 'duplicates_on' except if 'duplicates_on' is
        # set already. In this case, if 'bin_on_out' is not in
        # 'duplicates_on', it is understood as a voluntary user choice.
        # For all other cases, 'duplicates_on' has been set by user.
        # Setting 'duplicates_on' is the true reason of having 'bin_on_out'.
        # If allows the user to inform 'streamagg' that the binning colume
        # (with unique keys) is this one.
        if DUPLICATES_ON not in key_conf_in or key_conf_in[DUPLICATES_ON] is None:
            # Force 'bin_on_out'.
            key_conf_in[DUPLICATES_ON] = bin_on_out if bin_on_out else ordered_on
        # Step 2 / Process metadata if already existing aggregation results.
        # Initialize variables.
        if key in store:
            # Prior streamagg results already in store.
            # Retrieve corresponding metadata to re-start aggregations.
            prev_agg_res = store[key]
            if not _is_streamagg_result(prev_agg_res):
                raise ValueError(f"provided '{key}' data is not a 'streamagg' result.")
            streamagg_md = prev_agg_res._oups_metadata[KEY_STREAMAGG]
            # - 'last_seed_index' to trim accordingly head of seed data.
            # - metadata related to binning process from past binnings on prior data.
            # It is used in case 'bin_by' is a callable. If not used, it is an empty dict.
            # - metadata related to post-processing of prior aggregation results, to be
            # used by 'post'. If not used, it is an empty dict.
            seed_index_restart_set.add(streamagg_md[KEY_LAST_SEED_INDEX])
            segagg_buffer = (
                streamagg_md[KEY_SEGAGG_BUFFER] if streamagg_md[KEY_SEGAGG_BUFFER] else {}
            )
            post_buffer = streamagg_md[KEY_POST_BUFFER] if streamagg_md[KEY_POST_BUFFER] else {}
        else:
            # Because 'segagg_buffer' and 'post_buffer' are modified in-place
            # for each key, they are created separately for each key.
            segagg_buffer = {}
            post_buffer = {}
        # 'agg_res_buffer' and 'bin_res_buffer' are buffers to keep aggregation
        # chunks before a concatenation to record. Because they are appended
        # in-place for each key, they are created separately for each key.
        try:
            agg_config = setup_cumsegagg(agg, seed_dtypes)
        except Exception:
            raise ValueError(f"error raised for key '{key}'")
        keys_config[str(key)] = key_default | {
            "dirpath": os_path.join(store._basepath, key.to_path),
            "seg_config": seg_config,
            "agg_config": agg_config,
            KEY_BIN_ON_OUT: bin_on_out,
            POST: key_conf_in.pop(POST),
            "agg_res_buffer": [],
            "bin_res_buffer": [],
            KEY_SEGAGG_BUFFER: segagg_buffer,
            KEY_POST_BUFFER: post_buffer,
            "write_config": key_conf_in,
        }
    if not seed_index_restart_set:
        # No aggregation result existing yet. Whatever 'trim_start' value, no
        # trimming is possible.
        trim_start = False
    return (
        list(all_cols_in),
        trim_start,
        seed_index_restart_set,
        keys_config,
    )


def _iter_data(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    trim_start: bool,
    seed_index_restart: Union[int, float, pTimestamp, None],
    discard_last: bool,
    all_cols_in: List[str],
) -> Tuple[Union[int, float, pTimestamp], Generator]:
    """
    Return an iterator over seed data.

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
                filters=[filter_seed],
                row_filter=True,
                columns=all_cols_in,
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


def _post_n_write_agg_chunks(
    agg_res_buffer: List[pDataFrame],
    dirpath: str,
    key: str,
    write_config: dict,
    index_name: Union[str, None] = None,
    bin_res_buffer: Union[List[pDataFrame], None] = None,
    post: Callable = None,
    post_buffer: dict = None,
    last_seed_index=None,
    segagg_buffer: dict = None,
):
    """
    Write list of aggregation row groups with optional post, then reset list.

    Parameters
    ----------
    agg_res_buffer : List[pandas.DataFrame]
        List of chunks resulting from aggregation (pandas dataframes),
        either from bins  if only bins requested, or from snapshots if bins and
        snapshots requested.
    dirpath : str
        Path to which recording aggregation results.
    key : str
        Key for retrieving corresponding metadata.
    write_config : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Compulsory parameter defining at least `ordered_on`
        and `duplicates_on` columns.
    index_name : str, default None
        If a string, name index of dataframe resulting from aggregation with
        this value.
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
    bin_res_buffer : List[pandas.DataFrame], default None
        List of bins resulting from aggregation (pandas dataframes), when bins
        and snapshots are requested. None otherwise.
    post_buffer : dict, default None
        Buffer to keep track of data that can be processed during previous
        iterations. This pointer should not be re-initialized in 'post' or
        data from previous iterations will be lost.
        This dict has to contain data that can be serialized, as data is then
        kept in parquet file metadata.
    last_seed_index : default None
        Last index in seed data. Can be numeric type, timestamp... (for
        recording in metadata of aggregation results)
    segagg_buffer : dict
        Parameters from segmentation and aggregation process, that are required
        when restarting the aggregation with new seed data. (for recording in
        metadata of aggregation results)

    """
    # Concat list of aggregation results.
    agg_res = pconcat(agg_res_buffer) if len(agg_res_buffer) > 1 else agg_res_buffer[0]
    if index_name:
        # In case 'by' is a callable, index may have no name, but user may have
        # defined one with 'bin_on' parameter.
        agg_res.index.name = index_name
    # Keep group keys as a column before post-processing.
    agg_res.reset_index(inplace=True)
    # Reset (in place) buffer.
    agg_res_buffer.clear()
    if bin_res_buffer:
        bin_res = pconcat(bin_res_buffer) if len(bin_res_buffer) > 1 else bin_res_buffer[0]
        if index_name:
            # In case 'by' is a callable, index may have no name, but user may
            # have defined one with 'bin_on' parameter.
            bin_res.index.name = index_name
        bin_res.reset_index(inplace=True)
        bin_res_buffer.clear()
    else:
        bin_res = None
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
            KEY_STREAMAGG: {
                KEY_LAST_SEED_INDEX: last_seed_index,
                KEY_SEGAGG_BUFFER: segagg_buffer,
                KEY_POST_BUFFER: post_buffer,
            },
        }
    # Record data.
    write(dirpath=dirpath, data=agg_res, md_key=key, **write_config)


def agg_iter(
    seed_chunk: pDataFrame,
    key: str,
    config: dict,
):
    """
    Post-process and write iter. n-1, segment and aggregate iter. n.

    Parameters
    ----------
    seed_chunk : pDataFrame
        Chunk of seed data.
    key : str
        Key for recording aggregation results.

    Other Parameters
    ----------------
    config
        Settings related to 'key' for conducting post-processing, writing,
        segmentation and aggregation.

    Returns
    -------
    key, updated_config

        - ``key``, key to which changed parameters are related.
        - ``config``, dict with modified parameters.

    """
    # Post process and write.
    if (agg_res := config["agg_res"]) is not None:
        # If previous results, check if this is write time.
        # Retrieve length of aggregation results.
        agg_res_len = len(agg_res)
        agg_res_buffer = config["agg_res_buffer"]
        agg_n_rows = config["agg_n_rows"]
        if agg_res_len > 1:
            # Remove last row from 'agg_res' and add to
            # 'agg_res_buffer'.
            agg_res_buffer.append(agg_res.iloc[:-1])
            # Remove last row that is not recorded from total number of rows.
            agg_n_rows += agg_res_len - 1
            if (bin_res := config["bin_res"]) is not None:
                # If we have bins & snapshots, do same with bins.
                config["bin_res_buffer"].append(bin_res.iloc[:-1])
            # Keep floor part.
            if agg_n_rows:
                # Length of 'agg_res_buffer' is number of times it has been
                # appended.
                agg_mean_row_group_size = agg_n_rows // len(agg_res_buffer)
                if (
                    agg_n_rows + agg_mean_row_group_size
                    >= config["write_config"][KEY_MAX_ROW_GROUP_SIZE]
                ):
                    # Write results from previous iteration.
                    _post_n_write_agg_chunks(
                        agg_res_buffer=agg_res_buffer,
                        dirpath=config["dirpath"],
                        key=key,
                        write_config=config["write_config"],
                        index_name=config[KEY_BIN_ON_OUT],
                        bin_res_buffer=config["bin_res_buffer"],
                        post=config[POST],
                        post_buffer=config[KEY_POST_BUFFER],
                    )
                    # Reset number of rows within chunk list and number of
                    # iterations to fill 'agg_res_buffer'.
                    agg_n_rows = 0
            config["agg_n_rows"] = agg_n_rows
    # Segment and aggregate. Group keys becomes the index.
    agg_res = cumsegagg(
        data=seed_chunk,
        agg=config["agg_config"],
        bin_by=config["seg_config"],
        buffer=config[KEY_SEGAGG_BUFFER],
    )
    # 'agg_res' is 'main' aggregation results onto which are assessed
    # 'everything'. If only 'bins' are requested, it gathers bins.
    # If 'bins' and 'snapshots' are requested, it gathers snapshots.
    config["bin_res"], config["agg_res"] = agg_res if isinstance(agg_res, tuple) else None, agg_res
    return key, config


def streamagg(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    store: ParquetSet,
    keys: Union[dataclass, dict],
    agg: Optional[dict] = None,
    bin_by: Optional[Union[TimeGrouper, Callable[[Series, dict], tuple]]] = None,
    bin_on: Optional[Union[str, Tuple[str, str]]] = None,
    snap_by: Optional[Union[TimeGrouper, Series, DatetimeIndex]] = None,
    post: Optional[Callable] = None,
    trim_start: Optional[bool] = True,
    discard_last: Optional[bool] = True,
    parallel: Optional[bool] = False,
    **kwargs,
):
    """
    Aggregate sequentially on successive chunks (stream) of ordered data.

    This function conducts 'streamed aggregation', iteratively (out-of core)
    with optional post-processing of aggregation results (by use of vectorized
    functions or for dataframe formatting).

    Parameters
    ----------
    seed : Union[vDataFrame, Tuple[int, vDataFrame], ParquetFile]
        Seed data over which conducting streamed aggregations.
        If a tuple made of an `int` and a vaex dataframe, the `int` defines
        the size of chunks into which is split the dataframe.
        If purely a vaex dataframe, it is split by default into chunks of
        `6_345_000` rows, which for a dataframe with 6 columns of ``float64``
        or ``int64``, results in a memory footprint (RAM) of about 290MB.
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order. While this parameter is compulsory (most notably to manage
        duplicates when writing new aggregated results to existing ones), seed
        data is not necessarily grouped by this column, in which case ``bin_by``
        and/or ``bin_on`` parameters have to be set.
    store : ParquetSet
        Store to which recording aggregation results.
    keys : Union[Indexer, dict]
        Key(s) for recording aggregation results.
        If a dict, several keys can be specified for operating multiple
        parallel aggregations on the same seed. In this case, the dict should
        be in the form
        ``{key: {'agg': agg, 'bin_by': bin_by, 'bin_on': bin_on, 'snap_by': snap_by, 'post': post, **kwargs}}``
        Any additional parameters, (``**kwargs``) are forwarded to
        ``oups.writer.write`` when writing aggregation results to store, such
        as custom `max_row_group_size` or 'duplicates_on' parameters (see not
        below for 'duplicates_on').
        Please, note:

          - `bin_by` is a compulsory parameter.
          - If not specified, `bin_on` parameter in dict does not get default
            values.
          - If not specified, `agg`, `snap_by` and `post` parameters in dict
            get values from `agg`, `snap_by` and `post` parameters defined when
            calling `streamagg`.
            If using `snap_by` or `post` when calling `streamagg` and not
            willing to apply it for one key, set it to ``None`` in key specific
            config.

    agg : Union[dict, None], default None
        Dict in the form ``{"output_col":("input_col", "agg_function_name")}``
        where keys are names of output columns into which are recorded
        results of aggregations, and values describe the aggregations to
        operate. ``input_col`` has to exist in seed data.
        Examples of ``agg_function_name`` are `first`, `last`, `min`, `max` and
        `sum`.
        This parameter is compulsory, except if ``key`` parameter is a`dict`.
    bin_by : Union[TimeGrouper, Callable[[pd.DataFrame, dict]]], default None
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
        string, it refers to an existing column in seed data onto which
        applying the binning defined by ``bin_by`` parameter. Its value is then
        carried over as name for the column containing the group keys.
        It is further used when writing results for defining ``duplicates_on``
        parameter (see ``oups.writer.write``).
        When a tuple, the 1st string refers to an existing column in seed data,
        the 2nd the name to use for the column which values will be the group
        keys in aggregation results.
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
    snap_by : Optional[Union[TimeGrouper, Series, DatetimeIndex]], default None
        Values positioning the points of observation, either derived from a
        pandas TimeGrouper, or contained in a pandas Series.
        In case 'snap_by' is a Series, values  serve as locations for points of
        observation.
        Additionally, ``closed`` value defined by 'bin_on' specifies if points
        of observations are included or excluded. As "should be logical", if
          - `left`, then values at points of observation are excluded.
          - `right`, then values at points of observation are included.
    post : Callable, default None
        User-defined function accepting up to 3 parameters.

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
    trim_start : bool, default True
        If ``True``, and if aggregated results already existing, then retrieves
        the first index from seed data not processed yet (recorded in metadata
        of existing aggregated results), and trim all seed data before this
        index (index excluded from trim).
    discard_last : bool, default True
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step. See below
        notes.
    parallel : bool, default False
        Conduct processingof keys in parallel, with one process per `key`.
        If a single `key`, only one process is possible.

    Other Parameters
    ----------------
    kwargs : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Can define for instance custom `max_row_group_size`
        or 'duplicates_on' parameters (see not below for 'duplicates_on').

    Notes
    -----
    - Result is necessarily added to a dataset from an instantiated oups
      ``ParquetSet``. ``streamagg`` actually relies on the `advanced` update
      feature from oups.
    - If aggregation results already exist in oups ``ParquetSet`` instance,
      last index from previous aggregation is retrieved, and prior seed data is
      trimmed.
    - Aggregation is by default processed up to the last 'complete' index
      (included), and subsequent aggregation will start from the last index
      (included), assumed to be that of an incomplete row group.
      If `discard_last` is set `False`, then aggregation is process up to the
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
      data, or defined by 'bin_on' parameter if 'bin_by' is a callable.
    - When recording, both 'ordered_on' and 'duplicates_on' parameters are set
      when calling ``oups.writer.write``. If additional parameters are defined
      by the user, some checks are made.

        - 'ordered_on' is forced to 'streamagg' ``ordered_on`` parameter.
        - If 'duplicates_on' is not set by the user or is `None`, then it is
            - either set to the name of the output column for group keys
              defined by `bin_on` if `bin_on` is set. The rational is that this
              column identifies uniquely each bin, and so is a relevant column
              to identify duplicates.
            - if `bin_on` is not set, then it defaults to `ordered_on` column.

          There might case when this logic is unsuited. For instance, perhaps
          values in 'ordered_on' column does provide a unique valid identifier
          for bins already (if there are unique values in 'ordered_on'). It may
          then be that the column containing group keys is removed during user
          post-processing.
          To allow such specific use case, the user can set ``duplicates_on``
          as additional parameter to ``streamagg``. If the user omit a column
          name, it means that this is a voluntary choice from the user.

    """
    # Parameter setup.
    if not isinstance(keys, dict):
        keys = {
            keys: {
                AGG: agg,
                KEY_BIN_BY: bin_by,
                KEY_BIN_ON: bin_on,
                KEY_SNAP_BY: snap_by,
                POST: post,
            },
        }
    # Check 'kwargs' parameters are those expected for 'write' function.
    for param in kwargs.keys():
        if param not in WRITE_PARAMS:
            raise ValueError(
                f"{param} is neither a valid parameter for `streamagg` function, "
                "nor for `oups.write` function.",
            )
    (
        all_cols_in,
        trim_start,
        seed_index_restart_set,
        keys_config,
    ) = _setup(
        ordered_on=ordered_on,
        # TODO: if solved, simplify way of getting dtypes from vaex:
        # https://github.com/vaexio/vaex/issues/2403
        seed_dtypes=seed.dtypes if isinstance(seed, ParquetFile)
        # Case vaex dataframe.
        else seed[:1].to_pandas_df().dtypes.to_dict() if isinstance(seed, vDataFrame)
        # Case vaex dataframe in a tuple.
        else seed[1][:1].to_pandas_df().dtypes.to_dict(),
        store=store,
        keys=keys,
        agg=agg,
        snap_by=snap_by,
        post=post,
        trim_start=trim_start,
        **kwargs,
    )
    if len(seed_index_restart_set) > 1:
        raise ValueError(
            "not possible to aggregate on multiple keys with existing"
            " aggregation results not aggregated up to the same seed index.",
        )
    elif seed_index_restart_set:
        seed_index_restart = seed_index_restart_set.pop()
    else:
        seed_index_restart = None
    # Initialize 'iter_data' generator from seed data, with correct trimming.
    last_seed_index, iter_data = _iter_data(
        seed,
        ordered_on,
        trim_start,
        seed_index_restart,
        discard_last,
        all_cols_in,
    )
    n_keys = len(keys)
    n_jobs = min(int(cpu_count() * 3 / 4), n_keys) if (parallel and n_keys > 1) else 1
    with Parallel(n_jobs=n_jobs, prefer="threads") as p_job:
        for seed_chunk in iter_data:
            agg_loop_res = p_job(
                delayed(agg_iter)(seed_chunk, key, config) for key, config in keys_config.items()
            )
            for key, config in agg_loop_res:
                keys_config[key].update(config)
        # Check if at least one iteration has been achieved or not.
        agg_res = next(iter(keys_config.values()))["agg_res"]
        if agg_res is None:
            # No iteration has been achieved, as no data.
            return
        # Post-process & write results from last iteration, this time keeping
        # last aggregation row, and recording metadata for a future 'streamagg'
        # execution.
        p_job(
            delayed(_post_n_write_agg_chunks)(
                key=key,
                dirpath=config["dirpath"],
                agg_res_buffer=[*config["agg_res_buffer"], config["agg_res"]],
                write_config=config["write_config"],
                index_name=config[KEY_BIN_ON_OUT],
                bin_res_buffer=[*config["bin_res_buffer"], config["bin_res"]]
                if config["bin_res"]
                else [],
                post=config[POST],
                post_buffer=config[KEY_POST_BUFFER],
                last_seed_index=last_seed_index,
                segagg_buffer=config[KEY_SEGAGG_BUFFER],
            )
            for key, config in keys_config.items()
        )
    # Add keys in store for those who where not in.
    for key in keys:
        if key not in store:
            store._keys.add(key)
