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

* Currently only appending of new data to an existing one is possible, with no additional processing (in particular, no dropping of duplicates, nor re-ordering of data when 'old' data is being added).

.. code-block:: python

    # Initiating a new dataset
    ps[idx1] = df1
    # Appending the same data.
    ps[idx1] = df1
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
