#!/usr/bin/env python3
"""
Created on Wed Mar  9 21:30:00 2022.

@author: yoh
"""
import json
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

from fastparquet import ParquetFile
from pandas import DataFrame as pDataFrame
from pandas import Grouper
from pandas import Series
from pandas import concat
from pandas import read_json
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet
from oups.writer import MAX_ROW_GROUP_SIZE
from oups.writer import OUPS_METADATA
from oups.writer import OUPS_METADATA_KEY


VDATAFRAME_ROW_GROUP_SIZE = 6_345_000
ACCEPTED_AGG_FUNC = {min.__name_, max.__name__, sum.__name__}
# List of metadata keys to information to keep in metadata of aggregation
# results.
MD_KEY_STREAMAGG = "streamagg"
MD_KEY_LAST_COMPLETE_SEED_INDEX = "last_complete_seed_index"
MD_KEY_BINNING_BUFFER = "binning_buffer"
MD_KEY_LAST_AGGREGATION_ROW = "last_aggregation_row"


def _is_stremagg_result(seed: ParquetFile) -> bool:
    """Check if input parquet file is that of a dataset produced by streamaag.

    Parameters
    ----------
    seed : ParquetFile
        Parquet file to check.

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
    return (
        OUPS_METADATA_KEY in seed.key_value_metadata
        and MD_KEY_STREAMAGG in seed.key_value_metadata[OUPS_METADATA_KEY]
    )


def _post_n_write_agg_chunks(
    chunks: List[pDataFrame],
    store: ParquetSet,
    key: dataclass,
    write_config: dict,
    index_name: str = None,
    post: Callable = None,
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
        results to store.
    post : Callable, default None
        User-defined function accepting as a single parameter the pandas
        dataframe resulting from the aggregations defines by ``agg`` parameter,
        with first row already corrected with last row of previous streamed
        aggregation.
        This optional post-processing is intended for use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), or dataframe formatting before results are finally recorded.
    """
    # Keep last row as there might be not further iteration.
    if len(chunks) > 1:
        agg_res = concat(chunks)
    else:
        agg_res = chunks[0]
    if index_name:
        # In case 'by' is a callable, index has no name.
        agg_res.index.name = index_name
        print("agg_res with updated index name:")
        print(agg_res)
    agg_res = agg_res.reset_index(inplace=True)
    # Reset buffer.
    chunks = []
    # Post processing if any.
    if post:
        agg_res = post(agg_res)
    # Record data.
    store[key] = write_config, agg_res


def streamagg(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    agg: dict,
    store: ParquetSet,
    key: dataclass,
    by: Union[Grouper, Callable[[Series, dict], Union[Series, Tuple[Series, dict]]]] = None,
    bin_on: str = None,
    post: Callable = None,
    discard_last: bool = True,
    **kwargs,
):
    """Aggregate on continuous chunks (stream) of ordered data.

    This function conducts streamed aggregation iteratively (out-of core) with
    optional post-processing of aggregation results (by use of vectorized
    functions or dataframe formatting).
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
        rows, which for a dataframe with 6 columns of ``float64``/``int64``
        results in a memory footprint (RAM) of about 290MB.
    ordered_on : str
        Name of the column with respect to which seed dataset is in ascending
        order. Seed data is not necessarily grouped by this column, in which
        case ``by`` and/or ``bin_on`` parameters have to be set.
    by : Union[pd.Grouper, Callable[[Series, dict], array-like]], default None
        Parameter defining the binning logic.
        If a `Callable`, it is given

          - a ``data`` parameter, corresponding to a dataframe made of 2
            columns: ``ordered_on`` and ``bin_on``;
          - a ``buffer`` parameter, corresponding to a dict that can be used as
            a buffer for storing temporary results from one chunk processing to
            the next.

        This `Callable` has then to return an array of the same length as the
        input dataframe, and that specifyes bin labels, row per row.
        If data are required for re-starting calculation of bins on the next
        data chunk, the buffer has to be modified in place with temporary
        results to record for next-to-come binning iteration.
    agg : dict
        Dict in the form ``{"output_column":("input_col", agg_function)}``
        where keys are names of output columns into which are recorded
        results of aggregations, and values describe the aggregations to
        operate. ``"input_col"`` has to exist in seed data.
    store : ParquetSet
        Store to which recording aggregation results.
    key : Indexer
        Key for recording aggregation results.
    bin_on : str, default None
        Name of the column onto which applying the binning defined by ``by``
        parameter if ``by`` is not ``None``.
        Its value is then carried over as  name for the column containing the
        group keys. It is further used when writing results for defining
        ``duplicates_on`` parameter (see ``oups.writer.write``).
        If ``by`` is ``None`` (or a callable), then ``bin_on`` is expected to
        be set to an existing column name.
    post : Callable, default None
        User-defined function accepting as a single parameter the pandas
        dataframe resulting from the aggregations defines by ``agg`` parameter,
        with first row already corrected with last row of previous streamed
        aggregation.
        This optional post-processing is intended for use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), or dataframe formatting before results are finally recorded.
    discard_last : bool, default True
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step. See below
        notes.

    Other parameters
    ----------------
    kwargs : dict
        Settings forwarded to ``oups.writer.write`` when writing aggregation
        results to store.

    Notes
    -----
    - Result is necessarily added to a dataset from an instantiated oups
      ``ParquetSet``. ``streamagg`` actually relies on the `advanced` update
      feature from oups.
    - If aggregation results already exist in the instantiated oups
      ``ParquetSet``, index of last aggregation is retrieved, and new data is
      trimmed starting from this index. With this approach, a ``sum``
      aggregation will return the correct result, as same data is not accounted
      for twice.
    - This last index is recorded in the metadata.

        - It is either the value actually recorded as last index in metadata of
          seed data, if this metadata exists, and `discard_last`` is ``False``;
        - Or it is the last value from `ordered_on` column in seed data.

    - In the case binning is operated on a column different than `ordered_on`,
      then it may be that some bin edges fall in the middle of same values in
      `ordered_on`. In this case, because aggregation index used to trim head
      of new seed data stems from `ordered_on`, some values may be omitted from
      aggregation. To prevent this, user has to consider the following.

        - Either to ensure that the binning logic defined in ``by`` ends bins
          in-between different values in `ordered_on` column.
        - Or have unique values in `ordered_on` column coming from seed data.

    - By default, last row group (composed from rows sharing the same value in
      `ordered_on` column), is discarded (parameter ``discard_last`` set
      ``True``).

        - It may be for instance that this row group is not complete yet and
          should therefore not be accounted for. More precisely, new rows with
          same value in `ordered_on` may appear in seed data later on. Because
          seed data is trimmed to start from last processed value from
          `ordered_on` column (value excluded), these new rows would be
          excluded from the next aggregation, leading to an inaccurate
          aggregation result.
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
      available through the index, and if not reset as a column, it will not be
      stored (as index is discarded in oups parquet collection).
    """
    # Initialize 'self_agg', and check accepted aggregation functions.
    self_agg = {}
    for col_out, (_, agg_func) in agg.items():
        if agg_func.__name__ not in ACCEPTED_AGG_FUNC:
            raise ValueError(
                f"{agg_func.__name__} has not been tested so far."
                " Consider testing it before actually using it."
            )
        self_agg[col_out] = (col_out, agg_func)
    # Retrieve lists of input and output columns from 'agg'.
    all_cols_in = list({val[0] for val in agg.values()}) + [ordered_on]
    # Initialize 'iter_dataframe' from seed data, with correct trimming.
    seed_index_start = None
    seed_index_end = None
    binning_buffer = {}
    last_agg_row = None
    if key in store:
        # Prior streamagg results already in store.
        # Retrieve corresponding metadata to re-start aggregations.
        # Get seed index value to start new aggregation.
        # It is a value to be excluded when filtering seed data.
        # Trim accordingly head of seed data in this case.
        streamagg_md = store[key]._oups_metadata[MD_KEY_STREAMAGG]
        seed_index_start = streamagg_md[MD_KEY_LAST_COMPLETE_SEED_INDEX]
        # De-serialize 'seed_index_start'.
        seed_index_start = read_json(
            seed_index_start, orient="table", date_unit="ns", precise_float=True
        ).iloc[0, 0]
        print(f"seed_index_start: {seed_index_start}")
        if callable(by) and MD_KEY_BINNING_BUFFER in streamagg_md:
            # If 'by' is a callable, retrieve metadata related to binning
            # process from past binnings on prior data.
            binning_buffer.update(streamagg_md[MD_KEY_BINNING_BUFFER])
        last_agg_row = read_json(
            streamagg_md[MD_KEY_LAST_AGGREGATION_ROW],
            orient="table",
            date_unit="ns",
            precise_float=True,
        )
    # Seed index value to end new aggregation. Depending 'discard_last', it is
    # excluded or not.
    # Reason to discard last seed row (or row group) is twofold.
    # - last row is temporary (itself result of an on-going aggregation, not
    #   yet completed),
    # - last rows are part of a single row group not yet complete itself (new
    #   rows part of this row group to be expected).
    last_complete_seed_index = None
    if isinstance(seed, ParquetFile):
        # Case seed is a parquet file.
        if discard_last:
            # 'ordered_on' being necessarily in ascending order, last index
            # value is its max value.
            seed_index_end = seed.statistics["max"][ordered_on][-1]
        else:
            if _is_stremagg_result(seed):
                # If 'seed' is itself a 'streamagg' result, carry over the
                # 'last_complete_seed_index' to upcoming results from this
                # 'streamagg' execution.
                # 'last_complete_seed_index' is not de-serialized as it will be
                # ported as it is in metadata of upcoming results from this
                # 'streamagg' execution.
                last_complete_seed_index = json.loads(seed.key_value_metadata[OUPS_METADATA_KEY])[
                    MD_KEY_STREAMAGG
                ][MD_KEY_LAST_COMPLETE_SEED_INDEX]
            else:
                # Error if 'discard_last' is False and seed data is not a streamagg
                # result. This use case is unknown and if relevant, the expected
                # processing logic is yet to be defined in regard of this use case.
                raise ValueError(
                    "`discard_last` cannot be `False` if seed data is not itself "
                    "a 'streamagg' result."
                )
        filter_seed = []
        if seed_index_start:
            # 'seed_index_start' is excluded if defined.
            filter_seed.append((ordered_on, ">", seed_index_start))
        if seed_index_end:
            # 'seed_index_end' is excluded if defined.
            filter_seed.append((ordered_on, "<", seed_index_end))
        iter_data = seed.iter_row_groups(
            filters=filter_seed, row_filter=bool(filter_seed), columns=all_cols_in
        )
    else:
        # Case seed is a vaex dataframe.
        if isinstance(seed, tuple):
            vdf_row_group_size = seed[0]
            seed = seed[1]
        else:
            vdf_row_group_size = VDATAFRAME_ROW_GROUP_SIZE
        if discard_last:
            seed_index_end = seed[-1:].to_numpy()[0]
        if seed_index_start:
            seed = seed[seed[ordered_on] > seed_index_start]
        if seed_index_end:
            seed = seed[seed[ordered_on] < seed_index_start]
        if seed_index_start or seed_index_end:
            seed = seed.extract()
        iter_data = seed.to_pandas_df(chunk_size=vdf_row_group_size, column_names=all_cols_in)
    # Define aggregation result max size before writing to disk.
    max_agg_row_group_size = (
        kwargs["max_row_group_size"] if "max_row_group_size" in kwargs else MAX_ROW_GROUP_SIZE
    )
    print(f"from seed data, last_complete_seed_index: {last_complete_seed_index}")
    # Ensure 'by' and/or 'bin_on' are set.
    if by:
        if callable(by):
            if bin_on is None:
                raise ValueError(
                    "not possible to have `bin_on` set to `None " "while `by` is a callable."
                )
            elif bin_on in agg:
                # Check that this name is not already that of an output column
                # from aggregation.
                raise ValueError(
                    "not possible to have `bin_on` with value "
                    f"{bin_on} as it one of the output column "
                    "names from aggregation."
                )
            elif bin_on == ordered_on:
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
                    f" parameters, pointing to {bin_on} and "
                    f"{by_key} parameters."
                )
            elif by_key and bin_on is None:
                bin_on = by_key
    elif bin_on:
        # Case name of an existing column.
        bins = bin_on
    else:
        raise ValueError("one or several among `by` and `bin_on` are " "required.")
    print(f"seed_index_start: {seed_index_start}")
    print(f"seed_index_end: {seed_index_end}")
    # Number of rows in aggregation result.
    agg_n_rows = 0
    agg_mean_row_group_size = 0
    # Define 'index_name' of 'agg_res' if needed (to be used in 'post').
    index_name = bin_on if callable(by) else None
    # Buffer to keep aggregation chunks before a concatenation to record.
    agg_chunks_buffer = []
    # Setting minimal parameters in 'kwargs' if needed.
    write_config = {"ordered_on": ordered_on, "duplicates_on": bin_on}
    write_config = write_config | kwargs
    print(f"bin_on: {bin_on}")
    agg_res = None
    for i, seed_chunk in enumerate(iter_data):
        print(f"iteration: {i}")
        if agg_res:
            # If previous results, check if this is write time.
            # Spare last aggregation row as a dataframe for stitching with new
            # aggregation results from current iteration.
            len_agg_res = len(agg_res)
            print(f"agg_res with length: {len_agg_res}")
            if len_agg_res > 1:
                last_agg_row = agg_res.iloc[-1:]
                # Remove last row that is not recorded from total row number.
                agg_n_rows += len_agg_res - 1
                # Keep floor part.
                agg_mean_row_group_size = agg_n_rows // i
                # Remove last row from 'agg_res' and add to
                # 'agg_chunks_buffer'.
                agg_chunks_buffer.append(agg_res.iloc[:-1])
                if agg_n_rows + agg_mean_row_group_size >= max_agg_row_group_size:
                    # Write results from previous iteration.
                    _post_n_write_agg_chunks(
                        chunks=agg_chunks_buffer,
                        store=store,
                        key=key,
                        write_config=write_config,
                        index_name=index_name,
                        post=post,
                    )
                    # Reset number of rows within chunk list.
                    agg_n_rows = 0
            else:
                # Case 'agg_res' is a single row.
                print("agg_res is a single row")
                last_agg_row = agg_res
            print(f"agg_chunks_buffer is supposed to be empty: {agg_chunks_buffer}")
            print(f"last row after last row setting is: {last_agg_row}")
        print("1st row in seed")
        print(seed_chunk[:1])
        print("last row in seed")
        print(seed_chunk[-1])
        if callable(by):
            # Case callable. Bin 'ordered_on'.
            # If 'binning_buffer' is used, it has to be modified in-place, so
            # as to ship values from iteration N to iteration N+1.
            bins = by(data=seed_chunk.loc[:, cols_to_by], buffer=binning_buffer)
        # Bin and aggregate. Do not sort to keep order of groups as they
        # appear. Group keys becomes the index.
        agg_res = seed_chunk.groupby(bins, sort=False).agg(**agg)
        print(f"agg_res after aggregation: {agg_res}")
        # Stitch with last row from prior aggregation.
        if last_agg_row:
            if last_agg_row.index[0] == agg_res.index[0]:
                # If previous results existing, and if same bin labels shared
                # between last row of previous aggregation results (meaning same
                # bin), and first row of new aggregation results, then replay
                # aggregation between both.
                print("before aggregation of last row to first row")
                print(agg_res.iloc[0, :])
                agg_res.iloc[0, :] = (
                    concat([last_agg_row.iloc[:0], agg_res.iloc[:0]])
                    .groupby(level=0, sort=False)
                    .agg(**self_agg)
                )
                print("after aggregation of last row to first row")
                print(agg_res.iloc[0, :])
            else:
                # Simply add the last previous row in 'agg_chunk_buffer'
                # and do nothing with 'agg_res'.
                agg_chunks_buffer.append(last_agg_row)
                agg_n_rows += 1
                print("last_agg_row has been simply added to list of chunks.")
    # Setup metadata for a future 'streamagg' execution.
    last_agg_row = agg_res.iloc[-1:] if len_agg_res > 1 else agg_res
    last_agg_row = last_agg_row.to_json(orient="table", date_unit="ns", double_precision=15)
    # Define index of last complete seed 'row'.
    # - Either it is the last index recorded in metadata of seed data if existing
    # and 'discard_last' is False. In this case 'last_complete_seed_index' has
    # already been set, otherwise, it is `None`.
    # - Or it is last index value processed from seed data, which is already
    # set as:
    #   - one-but-last value of initial seed data if 'discard_last' is True,
    #   - or last value of initial seed data if 'discard_last' is False.
    if last_complete_seed_index is None:
        # Store a json serialized pandas series, to keep track of 'whatever
        # the object' the index is.
        last_complete_seed_index = (
            seed_chunk[ordered_on]
            .iloc[-1:]
            .to_json(orient="table", date_unit="ns", double_precision=15)
        )
    print(f"from new streamagg result, last_complete_seed_index: {last_complete_seed_index}")
    # Set oups metadata.
    metadata = {
        MD_KEY_STREAMAGG: {
            MD_KEY_LAST_COMPLETE_SEED_INDEX: last_complete_seed_index,
            MD_KEY_BINNING_BUFFER: binning_buffer,
            MD_KEY_LAST_AGGREGATION_ROW: last_agg_row,
        }
    }
    OUPS_METADATA.update(metadata)
    # Post-process & write results from last iteration, this time keeping
    # last row, and recording metadata for a future 'streamagg' execution.
    agg_chunks_buffer.append(agg_res)
    _post_n_write_agg_chunks(
        chunks=agg_chunks_buffer,
        store=store,
        key=key,
        write_config=write_config,
        index_name=index_name,
        post=post,
    )
