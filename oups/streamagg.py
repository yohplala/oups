#!/usr/bin/env python3
"""
Created on Wed Mar  9 21:30:00 2022.

@author: yoh
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import numpy as np
from fastparquet import ParquetFile
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Series
from pandas import Timestamp as pTimestamp
from pandas import concat
from pandas import date_range
from pandas import read_json
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet
from oups.router import ParquetHandle
from oups.writer import MAX_ROW_GROUP_SIZE
from oups.writer import OUPS_METADATA
from oups.writer import OUPS_METADATA_KEY


VDATAFRAME_ROW_GROUP_SIZE = 6_345_000
ACCEPTED_AGG_FUNC = {"first", "last", "min", "max", "sum"}
# List of keys to metadata of aggregation results.
MD_KEY_STREAMAGG = "streamagg"
MD_KEY_LAST_SEED_INDEX = "last_seed_index"
MD_KEY_BINNING_BUFFER = "binning_buffer"
MD_KEY_LAST_AGGREGATION_ROW = "last_aggregation_row"
MD_KEY_POST_BUFFER = "post_buffer"
# Config. for pandas dataframe serialization / de-serialization.
PANDAS_SERIALIZE = {"orient": "table", "date_unit": "ns", "double_precision": 15}
PANDAS_DESERIALIZE = {"orient": "table", "date_unit": "ns", "precise_float": True}


def _is_stremagg_result(handle: ParquetHandle) -> bool:
    """Check if input handle is that of a dataset produced by streamaag.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file to check.

    Returns
    -------
    bool
        `True` if parquet file contains metadata as produced by
        ``oups.streamagg``, which confirms this dataset has been produced with
        this latter function.
    """
    # As oups specific metadata are a string produced by json library, the last
    # 'in' condition is checking if the set of characters defined by
    # 'MD_KEY_STREAMAGG' is in a string.
    pf = handle.pf
    return (
        OUPS_METADATA_KEY in pf.key_value_metadata
        and MD_KEY_STREAMAGG in pf.key_value_metadata[OUPS_METADATA_KEY]
    )


def _get_streamagg_md(handle: ParquetHandle) -> tuple:
    """Retrieve and deserialize stremagg metadata from previous aggregation.

    Parameters
    ----------
    handle : ParquetHandle
        Handle to parquet file from which extracting metadata.

    Returns
    -------
    tuple
        Data recorded from previous aggregation to allow pursuing it with
        new seed data. 3 variables are returned, in order.

          - ``last_seed_index``, pandas dataframe with unique value being last
            index from seed data.
          - ``binning_buffer``, a dict to be forwarded to ``by`` if a callable.
          - ``last_agg_row``, last row grom previously aggregated results.
          - ``post_buffer``, a dict to be forwarded to ``post`` callable.

    """
    # Retrieve corresponding metadata to re-start aggregations.
    # Get seed index value to start new aggregation.
    # It is a value to be excluded when filtering seed data.
    # Trim accordingly head of seed data in this case.
    streamagg_md = handle._oups_metadata[MD_KEY_STREAMAGG]
    if streamagg_md[MD_KEY_LAST_SEED_INDEX]:
        last_seed_index = streamagg_md[MD_KEY_LAST_SEED_INDEX]
        # De-serialize 'last_seed_index'.
        last_seed_index = read_json(last_seed_index, **PANDAS_DESERIALIZE).iloc[0, 0]
    else:
        last_seed_index = None
    print("_get_streamagg_md")
    print("seed_index_restart:")
    print(last_seed_index)
    # Metadata related to binning process from past binnings on prior data.
    # It is used in cased 'by' is a callable.
    #    binning_buffer = streamagg_md[MD_KEY_BINNING_BUFFER]
    if streamagg_md[MD_KEY_BINNING_BUFFER]:
        binning_buffer = (
            read_json(streamagg_md[MD_KEY_BINNING_BUFFER], **PANDAS_DESERIALIZE).iloc[0].to_dict()
        )
    else:
        binning_buffer = {}
    print("binning_buffer:")
    print(binning_buffer)
    # 'last_agg_row' for stitching with new aggregation results.
    last_agg_row = read_json(streamagg_md[MD_KEY_LAST_AGGREGATION_ROW], **PANDAS_DESERIALIZE)
    print("last_agg_row:")
    print(last_agg_row)
    print("")
    # Metadata related to post-processing on prior aggregation results, to be
    # used by 'post'.
    if streamagg_md[MD_KEY_POST_BUFFER]:
        post_buffer = (
            read_json(streamagg_md[MD_KEY_POST_BUFFER], **PANDAS_DESERIALIZE).iloc[0].to_dict()
        )
    else:
        post_buffer = {}
    print("post_buffer:")
    print(post_buffer)
    return last_seed_index, binning_buffer, last_agg_row, post_buffer


def _set_streamagg_md(
    last_seed_index=None,
    binning_buffer: dict = None,
    last_agg_row: pDataFrame = None,
    post_buffer: dict = None,
):
    """Serialize and record stremagg metadata from last aggregation and post.

    Parameters
    ----------
    last_seed_index : default None
        Last index in seed data. Can be numeric type, timestamp...
    binning_buffer : dict
        Last values from binning process, that can be required when restarting
        the binning process with new seed data.
    last_agg_row : pDataFrame
        Last row from last aggregation results, required for stitching with
        aggregation results from new seed data.
    post_buffer : dict
        Last values from post-processing, that can be required when restarting
        post-processing of new aggregation results.
    """
    # Setup metadata for a future 'streamagg' execution.
    # Store a json serialized pandas series, to keep track of 'whatever
    # the object' the index is.
    print("within set_metadata")
    print("last_seed_index")
    print(last_seed_index)
    if last_seed_index:
        last_seed_index = pDataFrame({MD_KEY_LAST_SEED_INDEX: [last_seed_index]}).to_json(
            **PANDAS_SERIALIZE
        )
    print("last_agg_row")
    print(last_agg_row)
    last_agg_row = last_agg_row.to_json(**PANDAS_SERIALIZE)
    if binning_buffer:
        binning_buffer = pDataFrame(binning_buffer, index=[0]).to_json(**PANDAS_SERIALIZE)
    if post_buffer:
        post_buffer = pDataFrame(post_buffer, index=[0]).to_json(**PANDAS_SERIALIZE)
    print("binning_buffer")
    print(binning_buffer)
    # Set oups metadata.
    metadata = {
        MD_KEY_STREAMAGG: {
            MD_KEY_LAST_SEED_INDEX: last_seed_index,
            MD_KEY_BINNING_BUFFER: binning_buffer,
            MD_KEY_LAST_AGGREGATION_ROW: last_agg_row,
            MD_KEY_POST_BUFFER: post_buffer,
        }
    }
    OUPS_METADATA.update(metadata)


def _post_n_write_agg_chunks(
    chunks: List[pDataFrame],
    store: ParquetSet,
    key: dataclass,
    write_config: dict,
    index_name: str = None,
    post: Callable = None,
    isfrn: bool = None,
    post_buffer: dict = None,
    metadata: tuple = None,
):
    """Write list of aggregation row groups with optional post, then reset it.

    Parameters
    ----------
    chunks : List[pandas.DataFrame]
        List of chunks resulting from aggregation (pandas dataframes).
    store : ParquetSet
        Store to which recording aggregation results.
    key : Indexer
        Key for recording aggregation results.
    index_name : str, default None
        If set, name index of dataframe resulting from aggregation with this
        value.
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
    isfrn : boolean, default None
        Boolean indicating if first row of aggregation result is a new row, or
        is the 'same' that last row of aggregation result. If 'same', all
        values may not be the same, but the aggregation bin is the same.
    post_buffer : dict, default None
        Buffer to keep track of data that can be processed during previous
        iterations. This pointer should not be re-initialized in 'post' or
        data from previous iterations will be lost.
        This dict has to contain data that can be serialized, as data is then
        kept in parquet file metadata.
    metadata : tuple, default None
        Metadata to be recorded in parquet file. Data has to be serializable.
        If `None`, no metadata is recorded.
    """
    print("")
    print("_post_n_write_agg_chunks")
    # Keep last row as there might be not further iteration.
    if len(chunks) > 1:
        agg_res = concat(chunks)
    else:
        agg_res = chunks[0]
    if index_name:
        # In case 'by' is a callable, index may have no name.
        agg_res.index.name = index_name
    #    agg_res.index.name = index_name
    print("agg_res with updated index name:")
    print(agg_res)
    # Keep group keys as a column before post-processing.
    agg_res.reset_index(inplace=True)
    # Reset (in place) buffer.
    chunks.clear()
    if post:
        # Post processing if any.
        print("going to post processing")
        print("")
        agg_res = post(agg_res, isfrn, post_buffer)
    print("agg_res after post processing, right before writing")
    print(agg_res)
    print("")
    if metadata:
        print("last_agg_row again within post_n_write")
        print(metadata[2])
        # Set oups metadata.
        _set_streamagg_md(*metadata, post_buffer)
    # Record data.
    store[key] = write_config, agg_res


def streamagg(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    agg: dict,
    store: ParquetSet,
    key: dataclass,
    by: Union[Grouper, Callable[[Series, dict], Union[Series, Tuple[Series, dict]]]] = None,
    bin_on: Union[str, Tuple[str, str]] = None,
    post: Callable = None,
    trim_start: bool = True,
    discard_last: bool = True,
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
    agg : dict
        Dict in the form ``{"output_col":("input_col", "agg_function_name")}``
        where keys are names of output columns into which are recorded
        results of aggregations, and values describe the aggregations to
        operate. ``"input_col"`` has to exist in seed data.
    store : ParquetSet
        Store to which recording aggregation results.
    key : Indexer
        Key for recording aggregation results.
    by : Union[pd.Grouper, Callable[[pd.DataFrame, dict], array-like]], default
         None
        Parameter defining the binning logic.
        If a `Callable`, it is given following parameters.

          - A ``data`` parameter, corresponding to a dataframe made of
            column ``ordered_on``, and column ``bin_on`` if different than
            ``ordered_on``.
          - A ``buffer`` parameter, corresponding to a dict that can be used as
            a buffer for storing temporary results from one chunk processing to
            the next.

        This `Callable` has then to return an array of the same length as the
        input dataframe, and that specifyes bin labels, row per row.
        If data are required for re-starting calculation of bins on the next
        data chunk, the buffer has to be modified in place with temporary
        results to record for next-to-come binning iteration.
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
    trim_start : bool, default True
        If ``True``, and if aggregated results already existing, then retrieves
        the first index from seed data not processed yet (recorded in metadata
        of existing aggregated results), and trim all seed data before this
        index (index excluded from trim).
    discard_last : bool, default True
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step. See below
        notes.

    Other parameters
    ----------------
    kwargs : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store. Can define for instance custom `max_row_group_size`
        parameter.

    Notes
    -----
    - Result is necessarily added to a dataset from an instantiated oups
      ``ParquetSet``. ``streamagg`` actually relies on the `advanced` update
      feature from oups.
    - If aggregation results already exist in the instantiated oups
      ``ParquetSet``, last 'complete' index from previous aggregation is
      retrieved, and seed data is trimmed starting from this index.
    - Aggregation is by default processed up to this last 'complete' index
      (included), and subsequent aggregation will start from this index
      (excluded).
      If `discard_last` is set `False, then aggregation is process up to the
      last data.
    - This index is either

        - The one-but-last value from `ordered_on` column in seed data (default
          use case).
        - The value actually recorded as last 'complete' index in metadata of
          seed data, if this metadata exists, and `discard_last`` is ``False``;

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

    - In the case binning is operated on a column different than `ordered_on`,
      then it may be that some bin edges fall in the middle of same values in
      `ordered_on`. In this case, to prevent omitting new values that would
      have same index value as was processed last, `discard_last` should remain
      set `True`. If not, user has to consider the following options.

        - Either to ensure that the binning logic defined in ``by`` ends bins
          in-between different values in `ordered_on` column.
        - Or have unique values in `ordered_on` column coming from seed data.

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
      when calling ``oups.writer.write``. If additinal parameters are defined
      by the user, some checks are made.

        - 'ordered_on' is forced to 'streamagg' ``ordered_on`` parameter.
        - If 'duplicates_on' is not set by the user or is `None`, then it is
          set to the name of the output column for group keys defined by
          `bin_on`. The rational is that this column identifies uniquely each
          bin, and so is a relevant column to identify duplicates. But then,
          there might be case for which 'ordered_on' column does this job
          already (if there are unique values in 'ordered_on') and the column
          containing group keys is then removed during user post-processing.
          To allow this case, if the user is setting ``duplicates_on`` as
          additional parameter to ``streamagg``, it is not
          modified. It means omission of the column name containing the group
          keys, as defined by 'bin_on' parameter when it is set, is a
          voluntary choice from the user.

    """
    print("")
    print("setup streamagg")
    # Initialize 'self_agg', and check if aggregation functions are allowed.
    self_agg = {}
    for col_out, (_, agg_func) in agg.items():
        if agg_func not in ACCEPTED_AGG_FUNC:
            raise ValueError(
                f"{agg_func} has not been tested so far."
                " Consider testing it to proceed to its implementation."
            )
        self_agg[col_out] = (col_out, agg_func)
    # Initialize 'iter_dataframe' from seed data, with correct trimming.
    seed_index_restart = None
    #    seed_index_end = None
    binning_buffer = {}
    post_buffer = {}
    # Initializing 'last_agg_row' to a pandas dataframe, to allow using 'empty'
    # attribute in subsequent loop.
    last_agg_row = pDataFrame()
    if key in store:
        prev_agg_res = store[key]
        if _is_stremagg_result(prev_agg_res):
            print("key is in store")
            # Prior streamagg results already in store.
            # Retrieve corresponding metadata to re-start aggregations.
            seed_index_restart, binning_buffer_, last_agg_row, post_buffer_ = _get_streamagg_md(
                prev_agg_res
            )
            #        seed_index_restart = seed_index_restart.iloc[0, 0]
            if binning_buffer_:
                binning_buffer.update(binning_buffer_)
            if post_buffer_:
                post_buffer.update(post_buffer_)
        else:
            raise ValueError(
                f"provided key {key} is not that of aggregated results as" " issued by 'streamagg'."
            )
    else:
        # Results not existing yet. Whatever 'trim_start' value, no trimming
        # is possible yet.
        trim_start = False

    # Define aggregation result max size before writing to disk.
    max_agg_row_group_size = (
        kwargs["max_row_group_size"] if "max_row_group_size" in kwargs else MAX_ROW_GROUP_SIZE
    )
    # Ensure 'by' and 'bin_on' are set.
    if bin_on:
        if isinstance(bin_on, tuple):
            # 'bin_out_col': name of column containing group keys in agg res.
            bin_on, bin_out_col = bin_on
        else:
            bin_out_col = bin_on
        all
    else:
        bin_out_col = None
    if by:
        if callable(by):
            #            if bin_on is None:
            #                raise ValueError(
            #                    "not possible to have `bin_on` set to `None` while `by` is a callable."
            #                )
            #            elif bin_on in agg:
            if bin_out_col in agg:
                # Check that this name is not already that of an output column
                # from aggregation.
                raise ValueError(
                    f"not possible to have {bin_on} as column name for group"
                    " keys in aggregated result as it is also the name of one "
                    " of the output column names from aggregation."
                )
            elif bin_on == ordered_on or not bin_on:
                # Define columns forwarded to 'by'.
                cols_to_by = [ordered_on]
            else:
                cols_to_by = [ordered_on, bin_on]
        elif isinstance(by, Grouper):
            # Case pandas grouper.
            # https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
            bins = by
            by_key = by.key
            if bin_on and by_key and bin_on != by_key:
                raise ValueError(
                    "two different columns are defined for "
                    "achieving binning, both by `bin_on` and `by`"
                    f" parameters, pointing to '{bin_on}' and "
                    f"'{by_key}' columns respectively."
                )
            elif by_key and not bin_on:
                bin_on = by_key
    elif bin_on:
        # Case of using values of an existing column directly for binning.
        bins = bin_on
    else:
        raise ValueError("one or several among `by` and `bin_on` are required.")
    print(f"bin_on: {bin_on}")
    print(f"seed_index_restart: {seed_index_restart}")
    # Retrieve lists of input and output columns from 'agg'.
    all_cols_in = {val[0] for val in agg.values()}
    if bin_on and bin_on != ordered_on:
        all_cols_in = all_cols_in.union({ordered_on, bin_on})
    else:
        all_cols_in.add(ordered_on)
    all_cols_in = list(all_cols_in)
    # Seed index value to end new aggregation. Depending 'discard_last', it is
    # excluded or not.
    # Reason to discard last seed row (or row group) is twofold.
    # - last row is temporary (itself result of an on-going aggregation, not
    #   yet completed),
    # - last rows are part of a single row group not yet complete itself (new
    #   rows part of this row group to be expected).
    #    last_complete_seed_index = None
    if isinstance(seed, ParquetFile):
        # Case seed is a parquet file.
        # 'ordered_on' being necessarily in ascending order, last index
        # value is its max value.
        last_seed_index = seed.statistics["max"][ordered_on][-1]
        #        else:
        #            if _is_stremagg_result(seed):
        #                # If 'seed' is itself a 'streamagg' result, carry over the
        #                # 'last_complete_seed_index' to upcoming results from this
        #                # 'streamagg' execution.
        #                # 'last_complete_seed_index' is not de-serialized as it will be
        #                # ported as it is in metadata of upcoming results from this
        #                # 'streamagg' execution.
        #                last_complete_seed_index = json.loads(seed.key_value_metadata[OUPS_METADATA_KEY])[
        #                    MD_KEY_STREAMAGG
        #                ][MD_KEY_LAST_COMPLETE_SEED_INDEX]
        #                print(f"from seed data, last_complete_seed_index: {last_complete_seed_index}")
        #            else:
        #                # Error if 'discard_last' is False and seed data is not a streamagg
        #                # result. This use case is unknown and if relevant, the expected
        #                # processing logic is yet to be defined in regard of this use case.
        #                raise ValueError(
        #                    "`discard_last` cannot be `False` if seed data is not itself "
        #                    "a 'streamagg' result."
        #                )
        filter_seed = []
        if trim_start:
            filter_seed.append((ordered_on, ">=", seed_index_restart))
        if discard_last:
            filter_seed.append((ordered_on, "<", last_seed_index))
        print("filters:")
        print(filter_seed)
        print("row filters:")
        print(bool(filter_seed))
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
            #            seed_index_end = seed[-1:].to_numpy()[0]
        len_seed = len(seed)
        last_seed_index = seed.evaluate(ordered_on, len_seed - 1, len_seed, array_type="numpy")[0]
        if discard_last:
            seed = seed[seed[ordered_on] < last_seed_index]
        if trim_start:
            # 'seed_index_restart' is excluded if defined.
            if isinstance(seed_index_restart, pTimestamp):
                # Vaex does not accept pandas timestamp, only numpy or pyarrow
                # ones.
                seed_index_restart = np.datetime64(seed_index_restart)
            seed = seed[seed[ordered_on] >= seed_index_restart]
        #        if seed_index_end:
        #            seed = seed[seed[ordered_on] < seed_index_end]
        if trim_start or discard_last:
            seed = seed.extract()
        print("seed index restart")
        print(seed_index_restart)
        print(type(seed_index_restart))
        print("seed index end")
        print(last_seed_index)
        print(type(last_seed_index))
        iter_data = (
            tup[2]
            for tup in seed.to_pandas_df(chunk_size=vdf_row_group_size, column_names=all_cols_in)
        )
    # Number of rows in aggregation result.
    agg_n_rows = 0
    agg_mean_row_group_size = 0
    #    # Define 'index_name' of 'agg_res' if needed (to be used in 'post').
    #    index_name = bin_out_col
    # Buffer to keep aggregation chunks before a concatenation to record.
    agg_chunks_buffer = []
    # Setting 'write_config'.
    write_config = kwargs
    # Forcing 'ordered_on' for write.
    write_config["ordered_on"] = ordered_on
    # Adding 'bin_out_col' to 'duplicates_on' except if 'duplicates_on' is set
    # already. In this case, if 'bin_out_col' is not in 'duplicates_on', it is
    # understood as a voluntary user choice to not have 'bin_on' in
    # 'duplicates_on'.
    if "duplicates_on" not in write_config or write_config["duplicates_on"] is None:
        if bin_out_col:
            # Force 'bin_out_col'.
            write_config["duplicates_on"] = bin_out_col
        else:
            write_config["duplicates_on"] = ordered_on
        # For all other cases, 'duplicates_on' has been set by user.
        # If 'bin_out_col' is not in 'duplicates_on', it is understood as a
        # voluntary choice by the user.
    agg_res = None
    len_agg_res = None
    # Initialise 'isfrn': is first row (from aggregation result) a new row?
    # For 1st iteration it is necessarily a new one.
    isfrn = True
    #    print("")
    #   /!\ Think to remove i once everything done.
    i = 0
    for seed_chunk in iter_data:
        print("")
        print(f"iteration to fill agg_chunk_buffer: {i}")
        print("")
        print("seed_chunk:")
        print(seed_chunk)
        if agg_res is not None:
            # If previous results, check if this is write time.
            # Spare last aggregation row as a dataframe for stitching with new
            # aggregation results from current iteration.
            #            print("processing agg_res")
            #            print(f"agg_res with length: {len_agg_res}")
            if len_agg_res > 1:
                # Remove last row from 'agg_res' and add to
                # 'agg_chunks_buffer'.
                agg_chunks_buffer.append(agg_res.iloc[:-1])
                # Remove last row that is not recorded from total row number.
                agg_n_rows += len_agg_res - 1
                # Number of iterations to increment 'agg_chunk_buffer'.
                # Check if 'i' can be removed when all prints are removed.
                i += 1
            #            print("")
            # Keep floor part.
            if agg_n_rows:
                agg_mean_row_group_size = agg_n_rows // i
                #                print(f"agg_n_rows: {agg_n_rows}")
                #                print(f"agg_n_rows is supposed to be: {sum([len(df) for df in agg_chunks_buffer])}")
                #                print(f"agg_mean_row_group_size: {agg_mean_row_group_size}")
                #                print("targeted next number of rows in agg_chunks_buffer at next iteration:")
                #                print(agg_n_rows + agg_mean_row_group_size)
                #                print(f"limit to equal or exceed to trigger write: {max_agg_row_group_size}")
                if agg_n_rows + agg_mean_row_group_size >= max_agg_row_group_size:
                    # Write results from previous iteration.
                    #                    print("writing chunk")
                    #                    print("")
                    print("OUPS_METADATA_KEY")
                    print(OUPS_METADATA_KEY)
                    print("OUPS_METADATA")
                    print(OUPS_METADATA)
                    print("write_config during loop")
                    print(write_config)
                    _post_n_write_agg_chunks(
                        chunks=agg_chunks_buffer,
                        store=store,
                        key=key,
                        write_config=write_config,
                        #                        index_name=index_name,
                        index_name=bin_out_col,
                        post=post,
                        isfrn=isfrn,
                        post_buffer=post_buffer,
                        metadata=None,
                    )
                    # Reset number of rows within chunk list and number of
                    # iterations to fill 'agg_chunks_buffer'.
                    agg_n_rows = 0
                    i = 0
        #                    print("agg_chunks_buffer is supposed to be empty:")
        #            print("agg_chunk_buffer")
        #            print(agg_chunks_buffer)
        #            print("")
        #            print("")
        #            print("last agg row after last row setting is:")
        #            print(last_agg_row)
        #            print("")
        #        print("")
        #        print("1st row in seed chunk")
        #        print(seed_chunk.iloc[:1])
        #        print("last row in seed chunk")
        #        print(seed_chunk.iloc[-1:])
        #        print("")
        if callable(by):
            # Case callable. Bin 'ordered_on'.
            # If 'binning_buffer' is used, it has to be modified in-place, so
            # as to ship values from iteration N to iteration N+1.
            bins = by(data=seed_chunk.loc[:, cols_to_by], buffer=binning_buffer)
        # Bin and aggregate. Do not sort to keep order of groups as they
        # appear. Group keys becomes the index.
        agg_res = seed_chunk.groupby(bins, sort=False).agg(**agg)
        len_agg_res = len(agg_res)
        print("agg_res after aggregation:")
        print(agg_res)
        print("")
        # Stitch with last row from *prior* aggregation.
        print("last row before stitching:")
        print(last_agg_row)
        print("")
        if not last_agg_row.empty:
            #            print("last_agg_row is not empty")
            isfrn = (first := agg_res.index[0]) != (last := last_agg_row.index[0])
            if isfrn:
                n_added_rows = 1
                # Bin of 'last_agg_row' does not match bin of first row in
                # 'agg_res'.
                if isinstance(by, Grouper) and by.freq:
                    # If bins are defined with pandas time grouper ('freq'
                    # attribute is not `None`), bins without values from seed
                    # that could exist at start of chunk will be missing.
                    # In a classic pandas aggregation, these bins would however
                    # be present in aggregation results, with `NaN` values in
                    # columns. These bins are thus added here to maintain
                    # classic pandas behavior.
                    missing = date_range(
                        start=last, end=first, freq=by.freq, inclusive="neither", name=by.key
                    )
                    if not missing.empty:
                        last_agg_row = concat(
                            [last_agg_row, pDataFrame(index=missing, columns=last_agg_row.columns)]
                        )
                        n_added_rows = len(last_agg_row)
                # Add last previous row (and possibly missing ones if pandas
                # time grouper) in 'agg_chunk_buffer' and do nothing with
                # 'agg_res' at this step.
                print("last_agg_row after possible extension with NaN")
                print(last_agg_row)
                agg_chunks_buffer.append(last_agg_row)
                agg_n_rows += n_added_rows
                print(f"Number of rows in agg_chunks_buffer: {agg_n_rows}")
                # Number of iterations to increment 'agg_chunk_buffer'.
                i += 1
            else:
                # If previous results existing, and if same bin labels shared
                # between last row of previous aggregation results (meaning same
                # bin), and first row of new aggregation results, then replay
                # aggregation between both.
                #                print("before aggregation of last row to first row")
                #                print(agg_res.iloc[:1])
                #                print("concat/agg res:")
                #                print(
                #                    concat([last_agg_row.iloc[:1], agg_res.iloc[:1]])
                #                    .groupby(level=0, sort=False)
                #                    .agg(**self_agg)
                #                )
                agg_res.iloc[:1] = (
                    concat([last_agg_row, agg_res.iloc[:1]])
                    .groupby(level=0, sort=False)
                    .agg(**self_agg)
                )
            #                print("agg_res after aggregation of last row to first row")
            #                print(agg_res)
        #                print("last_agg_row has been simply added to list of chunks.")
        # Setting 'last_agg_row' from new 'agg_res'.
        last_agg_row = agg_res.iloc[-1:] if len_agg_res > 1 else agg_res
    # Post-process & write results from last iteration, this time keeping
    # last row, and recording metadata for a future 'streamagg' execution.
    agg_chunks_buffer.append(agg_res)
    #    print("writing last chunk")
    #    print("agg_chunks_buffer")
    #    print(agg_chunks_buffer)
    # Set metadata for a future 'streamagg' execution.
    # Define index of last complete seed 'row'.
    # - Either it is the last index recorded in metadata of seed data if existing
    # and 'discard_last' is False. In this case 'last_complete_seed_index' has
    # already been set, otherwise, it is `None`.
    # - Or it is last index value processed from seed data, which is already
    # set as:
    #   - one-but-last value of initial seed data if 'discard_last' is True,
    #   - or last value of initial seed data if 'discard_last' is False.
    #    if not discard_last:
    #    if last_complete_seed_index is None:
    #        last_complete_seed_index = seed_chunk[ordered_on].iloc[-1:].reset_index(drop=True)
    #        seed_index_end = None
    print("last_agg_row before sending to post_n_write")
    print(last_agg_row)
    # A deep copy is made for 'last_agg_row' to prevent a specific case where
    # 'agg_chuks_buffer' is a list of a single 'agg_res' dataframe of a single
    # row. In this very specific case, both 'agg_res' and 'last_agg_row' points
    # toward the same dataframe, but 'agg_res' gets modified in '_post_n_write'
    # while 'last_agg_row' should not be. The deep copy prevents this.
    print("write_config before last write")
    print(write_config)
    _post_n_write_agg_chunks(
        chunks=agg_chunks_buffer,
        store=store,
        key=key,
        write_config=write_config,
        #        index_name=index_name,
        index_name=bin_out_col,
        post=post,
        isfrn=isfrn,
        post_buffer=post_buffer,
        metadata=(last_seed_index, binning_buffer, last_agg_row.copy()),
    )
    #        print("from new streamagg result, last_complete_seed_index:")
    #        print(last_complete_seed_index)
