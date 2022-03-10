#!/usr/bin/env python3
"""
Created on Wed Mar  9 21:30:00 2022.

@author: yoh
"""
from typing import Callable, Union

from fastparquet import ParquetFile
from pandas import Grouper
from vaex.dataframe import DataFrame as vDataFrame

from oups.collection import ParquetSet
from oups.indexer import Indexer


def streamagg(
    seed: Union[vDataFrame, ParquetFile],
    store: ParquetSet,
    key: Indexer,
    ordered_on: str,
    by: Union[Grouper, Callable],
    agg: dict,
    post: Callable,
    discard_last: bool = True,
):
    """Aggregate on continuous chunks (stream) of ordered data.

        This function allows the user to post process aggregation results (using
        vectorized functions) without paying (too much) attention about how
        stitching results from new data to results from past data.
        It conducts group-by aggregations over continuous data, in an iterative
        process.

        Parameters
        ----------
        seed : Union[vDataFrame, ParquetFile]
            Seed data over which conducting streamed aggregations.
        store : ParquetSet
            Store to which recording aggregation results.
        key : Indexer
            Key for recording aggregation results.
        ordered_on : str
            Name of the column with respect to which dataset is in ascending order.
        by : Union[pd.Grouper, Callable]
            Parameter defining the binning logic.
            If a `Callable`, it is given as parameter the column specified by
            ``ordered_on`` parameter, and it has to return an array of the same
            length specifying the bins.
        agg : dict
            Dict which keys are names of columns into which are recorded results of
            aggregations, and values are corresponding aggregation functions, in
            form ``{"res_column":{"input_col", agg_function}}``. ``"input_col"``
            has to be found
        post : Callable
            User-defined function accepting 2 parameters.

                - The pandas dataframe resulting from the aggregations defines by
                  ``agg`` parameter.
                - The last of row of results that could already exist from prior
                  ``streamagg`` run with same parameters.

            ``post`` can ensure 2 functions.

                - The first is compulsory. The same aggregations as defined by
                  ``agg_function`` from ``agg`` parameter, has to be conducted
                  between the last row of prior results, and the 1st row of new results.
                - The 2nd one is depending use case. Additional post-processing
                  can be conducted at this step, before results are finally
                  recorded.
        discard_last : bool, default True

        df_generator (either vdf.to_pandas(chunk...), either pf.iter_row_groups) // or vdf or pdf // or simple pf
        store: ParquetSet

        key: indexer to record new data

        ordered_on: column name
        last_on: column name where is/will be stored last index
                 (means it is to be recorded)
        discard_last: bool - means last rows in seed is to be excluded before data is used for aggregation
                             means we need to know it is the last rows:
                                 - get their index (in ordered_on)
                                 - exclude this index by filtering the data
        cum_agg: dict {col_name: agg_function}
        func:
               user defined aggregation function accepting a pandas dataframe,
               defining its own binning strategy, and aggregations
               returning data either as vdf or pdf

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

    https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
    """
