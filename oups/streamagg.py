#!/usr/bin/env python3
"""
Created on Wed Mar  9 21:30:00 2022.

@author: yoh
"""
from dataclasses import dataclass
from typing import Callable, Tuple, Union

from fastparquet import ParquetFile
from pandas import Grouper
from pandas import Series
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet


VDATAFRAME_ROW_GROUP_SIZE = 6_345_000
METADATA_KEY = "streamagg"


def streamagg(
    seed: Union[vDataFrame, ParquetFile],
    ordered_on: str,
    by: Union[Grouper, Callable[[Series, dict], Union[Series, Tuple[Series, dict]]]],
    agg: dict,
    store: ParquetSet,
    key: dataclass,
    write_conf: dict = None,
    post: Callable = None,
    discard_last: bool = True,
):
    """Aggregate on continuous chunks (stream) of ordered data.

    This function conducts streamed aggregation iteratively (out-of core) with
    optional post-processing on aggregation results (using vectorized
    functions).
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
        order. Seed data is necessarily grouped by this column.
    by : Union[pd.Grouper, Callable[[Series, dict], Union[Series, Tuple[Series, dict]]]
        Parameter defining the binning logic.
        If a `Callable`, it is given the column specified by ``ordered_on``
        parameter and a dict that can be used as a buffer for storing
        temporary results from one chunk processing to the next.
        This `Callable` has then to return an array-like of the same length
        specifying bin labels, row per row. If data are required for
        re-starting calculation of bins on the next data chunk, a tuple should
        be returned, with first item the array-like of bin labels, and second
        item a dict containing the temporary results to record for next-to-come
        binning iteration.
    agg : dict
        Dict which keys are names of columns into which are recorded results of
        aggregations, and values describe the aggregations to operate, in the
        form ``{"res_column":{"input_col", agg_function}}``.
        ``"input_col"`` has to exist in seed data.
    store : ParquetSet
        Store to which recording aggregation results.
    key : Indexer
        Key for recording aggregation results.
    write_conf : dict, default None
        Settings forwarded to ``oups.writer.pwrite`` when writing aggregation
        results to store.
    post : Callable, default None
        User-defined function accepting as 1st parameter the pandas dataframe
        resulting from the aggregations defines by ``agg`` parameter, with
        first row already corrected with last row of previous streamed
        aggregation.
        This optional post-processing can only be managed by use of vectorized
        functions (not mixing rows together, but operating on one or several
        columns), before results are finally recorded.
        During this step, columns containing aggregation results can be removed
        if not needed afterwards.
    discard_last : bool, default True
        If ``True``, last row group in seed data (sharing the same value in
        `ordered_on` column) is removed from the aggregation step. See below
        notes.

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

    - By default, last row group (composed from rows sharing the same value in
      `ordered_on` column), is discarded (parameter ``discard_last`` set
      ``True``).

        - It may be for instance that this row group is not complete yet and
          should therefore not be accounted for. More precisely, new rows with
          same value in `ordered_on` may appear in seed data later on. Because
          seed data is trimmed to start from last processed value from
          `ordered_on` column, these new rows would be excluded from the next
          aggregation, leading to an inaccurate aggregation result.
        - Or if composed of a single row, this last row in seed data is
          temporary (and may get its final values only at a later time, when it
          becomes the one-but-last row, as a new row is added).

    """
    # Initialize 'iter_dataframe' from seed data, with correct trimming.
    # Trim start if streamagg data already exists.
    b_oups = b"oups_streamagg"
    #    b_slsi = b'streamagg_last_seed_index'
    if key in store:
        # Prior streamagg results already in store. Retrieve corresponding
        # metadata.
        # /!\ Shortcut to decoded metadata in router.
        streamagg_md = store[key].pf.key_value_metadata[b_oups].decode()
        streamagg_md.copy()  # dummy code
        # Decode
        # streamagg_md = pickle.loads(codecs.decode(streamagg_md.encode("ascii"), "base64"))

        # if b_oups in kvm and b_slsi in kvm[b_oups]:
        #    seed_index_start = kvm[b_oups][b_slsi]

    seed_index_end = None
    if isinstance(seed, ParquetFile):
        # Case seed is 'ParquetFile'.
        if discard_last:
            # Last index value is necessarily
            seed_index_end = seed.statistics["max"][ordered_on][-1]
        elif not (
            b"oups" in seed.key_value_metadata
            and b"streamagg_last_seed_index" in seed.key_value_metadata[b"oups"]
        ):
            # Error if 'discard_last' is False and no 'last_seed_index' exists
            # in seed metadata. This use case is unknown and the relevant logic
            # is to be defined in regard of this use case.
            raise ValueError(
                "`discard_last` cannot be `False` if seed data is"
                " not itself a 'streamagg' result."
            )
    else:
        # Case seed is vaex dataframe.
        if isinstance(seed, tuple):
            vdf_row_group_size = seed[0]
            seed = seed[1]
        else:
            vdf_row_group_size = VDATAFRAME_ROW_GROUP_SIZE
            vdf_row_group_size.copy()  # dummy code
        if discard_last:
            seed_index_end = seed[-1:].to_numpy()[0]
            seed_index_end.copy()  # dummy code

            # /!\ WiP

            # last_index =  None
            # if key in store:
            # Get "start" from last_index and make sure
            # last_index = store[key].pf. ... last value
            # Trim data if data available (made in next step)

            # Retrieve and filter seed data:
            #  - if existing aggregated data, trim to keep only newer data.
            #    - retrieve last_index from metadata of **aggregated** data
            #  - if 'discard_last', discard last row groups (sharing same index).
            #  - if ParquetFile, load only columns in 'seed' present in 'agg' and 'ordered_on',
            # if isinstance(seed, vDataFrame):
            # Trim and iterate
            # iter_data = seed[ts > start].to_pandas_df(chunk=)
            # elif isinstance(seed, ParquetFile):
            # Trim and iterate
            # iter_data = seed.iter_row_groups(filter=....)

            # Iterate.

            # Apply 'by' if a callable, recording temporary binning data in metadata.
            # https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html

            # Aggregate.

            # Stitch.
            # Assess need for stitching by comparing 1st aggregation label from new
            # results with last from existing results. Stitch results if same bin label.
            # Retrieve result column names from 'agg' for stitching step.

            # Spare (buffer) last aggregation row for next iteration or last recording step,
            # Trim last row from aggregation result.

            # If not empty aggregation results.
            # - vectorized post-processing.
            # - record aggregated chunk.

            # End of iteration.

            # Record in metadata:
            #  - last aggregation row
            #  - last binning buffer
            #  - last index
            #     - either being the last_index recoreded in metadata of seed data
            #       if existing and 'discard_last' is False
            #     - else last index value processed from seed data
            #          (one-but-last value if 'discard_last' is True,
            #           or last value if discard_last is False)


#            encoded = codecs.encode(
#                pickle.dumps(cols, protocol=pickle.HIGHEST_PROTOCOL), "base64"
#            ).decode("ascii")
# Record last row.
# Record last row of aggregated data as json:
# https://pandas.pydata.org/docs/user_guide/io.html#table-schema
