``ParquetSet``
==============

Purpose and creation
--------------------

An instance of ``ParquetSet`` class gathers a collection of datasets.
``ParquetSet`` instantiation requires the definition of a *collection path* and a dataset *indexing logic*.

* A *collection path* is directory path (existing or not) where will be (are) gathered directories for each dataset.
* An *indexing logic* is formalized by use of a ``@toplevel``-decorated class as presented in :doc:`indexing`.

.. code-block:: python

    from os import path as os_path
    from oups import ParquetSet, toplevel

    # Define an indexing logic to generate each individual dataset folder name.
    @toplevel
    class DatasetIndex:
        country: str
        city: str

    # Define a collection path.
    dirpath = os_path.expanduser('~/Documents/code/data/weather_knowledge_base')

    # Initialize a parquet dataset collection.
    ps = ParquetSet(dirpath, DatasetIndex)

Usage notes
-----------

Dataframe format
~~~~~~~~~~~~~~~~

* *oups* accepts `pandas <https://github.com/pandas-dev/pandas>`_ or `vaex <https://github.com/vaexio/vaex>`_ dataframes.
* Row index is dropped when recording. If the index of your dataframe is meaningful, make sure to reset it as a column. This only applies for *pandas* dataframes, as *vaex* ones have no row index.

.. code-block:: python

    pandas_df = pandas_df.reset_index()

* Column multi-index can be recorded. Here again *vaex* has no support for column multi-index. But if your *vaex* dataframe comes from a *pandas* one initially with column multi-index, you can expand it again at recording.

.. code-block:: python

    # With 'vaex_df' created from a pandas dataframe with column multi-index.
    ps[idx] = {'cmidx_expand'=True}, vaex_df

Writing
~~~~~~~

* When recording data to disk, ``ParquetSet`` instance accepts a ``tuple`` which first item is then a dict defining recording setting. Parameters accepted are those of ``oups.writer.write`` function and complementary to ``dirpath`` and ``data`` (see :doc:`api` for a review).

.. code-block:: python

    ps[idx] = {'row_group_size'=5_000_000, 'compression'='BROTLI'}, df

* New datasets can be added to the same collection, as long as the index used is an instance from the same ``@toplevel``-decorated class as the one used at ``ParquetSet`` instantiation.

Reading
~~~~~~~

* A ``ParquetSet`` instance returns a ``ParquetHandle`` which gives access to data either through 'handles' (*vaex* dataframe or *fastparquet* parquet file) or directly as a *pandas* dataframe.

  * *fastparquet* parquet file ``ps[idx].pf``,
  * or *pandas* dataframe ``ps[idx].pdf``,
  * or *vaex* dataframe ``ps[idx].vdf``.

Updating
~~~~~~~~

If using an *index* already present in ``Parquet`` instance, existing data is updated with new one. Different keywords control data updating logic. These keywords can also be reviewed in :doc:`api`, looking at ``write`` function signature.

* ``ordered_on``, default ``None``

This keyword specifies the name of a column according which dataset is ordered (ascending order).

  * When specifying it, position of the new data with respect to existing data is checked. It allows data insertion.
  * It also enforces *sharp* row group boundaries, meaning that a row group will necessarily starts with a new value in column specified by ``ordered_on`` at the expense of ensuring a constant row group size. If used continuously each time data is written, no row group start in the middle of duplicates values. This has two advantages. First, insertion of a new row group among existing ones is unambiguous. Second is related to drop of duplicates, discussed below.

* ``duplicates_on``, default ``None``

This keyword specifies the names of columns to identify duplicates. If it is an empty list ``[]``, all columns are used.

Motivation for dropping duplicates is that new values (from new data) can replace old values (in existing data). Typical use case is that of updating *OHLC* financial datasets, for which the *High*, *Low* and *Close* values of the last candle (in-progress) can change until the candle is completed. When appending newer data, values of this last candle need then to be updated.

The implementation of this logic in a way that it only needs to be carried out row group per row group and not over the full dataset, has most notably 2 implications. Make sure to understand them and check if it applies correctly to your own use case. If not, a solution for you is to prepare the data the way you intend it to be before recording it anew.

  * Duplicates in existing data that is not rewritten are not dropped.
  * ``ordered_on`` column is also a value of the row that contributes to identifying duplicates. ``ordered_on`` column is thus added to the list of columns specified by ``duplicates_on``.

* ``irgs_max``, default ``None``

This keyword specifies the maximum number allowed of `incomplete` row groups. An `incomplete` row group is one that does not quite reach ``max_row_group_size`` yet (some approximations of this target are managed within the code).
By using this parameter, you allow a `buffer` of trailing `incomplete` row groups. Hence, new data is not systematically merged to existing one, but only appended as new row groups.
The interest is that an `appending` operation is faster than `merging` with existing row groups, and for adding only few more rows, `merging` seems like a heavy, unjustified operation.
Setting ``irgs_max`` triggers assessment of 2 conditions to initiate a `merge` (`coalescing` all incomplete trailing row groups to try making `complete` ones) Either one or the other has to be met to validate a `merge`.

  * ``irgs_max`` is reached;
  * The total number of rows within the `incomplete` row groups summed with the number of rows in the new data equals or exceeds `max_row_group_size`.

.. code-block:: python

    # Initiating a new dataset
    ps[idx1] = df1
    # Appending the same data.
    ps[idx1] = {'irgs_max': 4}, df1
    # Reading.
    ps[idx1].pdf
    Out[2]:
       timestamp  temperature
    0 2021-01-01           10
    1 2021-01-02           11
    2 2021-01-03           12
    3 2021-01-04           13
    4 2021-01-05           14
    5 2021-01-01           10    # new appended data
    6 2021-01-02           11
    7 2021-01-03           12
    8 2021-01-04           13
    9 2021-01-05           14

Other "goodies"
~~~~~~~~~~~~~~~

.. code-block:: python

    # Review store content.
    ps
    Out[3]:
    germany-berlin
    japan-tokyo

    # Get number of datasets.
    len(ps)
    Out[4]: 2

    # Delete a dataset (delete data from disk).
    del ps[idx1]
    ps
    Out[5]: japan-tokyo

    # 'Discover' an existing dataset collection.
    # (initial schema definition is needed)
    ps2 = ParquetSet(store_path, DatasetIndex)
    ps2
    Out[6]: japan-tokyo
