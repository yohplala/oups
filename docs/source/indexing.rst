Collection indexing
===================

Motivation
----------

Datasets are gathered within a parent directory as a collection. Each of them materialize as parquet files located in a child directory whose naming is derived from a user-defined index.

By formalizing this index through a *likewise dataclass*, index management (user scope) is dissociated from path management (*oups* scope).

Implementation
--------------

*oups* provides 2 class decorators for defining an indexing logic.

* ``@toplevel`` is compulsory, and defines naming logic of the first directory level,
* ``@sublevel`` is optional, and can be used as many times as number of sub-directories are required

By splitting indexes into different directory levels, related datasets can be gathered in common directories.
A first level could for instance specify physical quantities in different places, and a second one could specify the sampling frequency of the measures.

Each of these levels is thus specified by a *likewise dataclass*. Those corresponding to a parent directory necessarily embed as last attribute the sublevel-related class (see example).

Fields separator
----------------

When decorating with ``@toplevel``, ``fields_sep`` parameter can be modified to a different character than default one ``-``. This separator applies to all *levels*.

Example
-------

.. code-block:: python

    from oups import sublevel, toplevel

    @sublevel
    class Sampling:
        frequency: str
    @toplevel
    class Measure:
        quantity: str
        city: str
        sampling: Sampling
    # Define different indexes for temperature in Berlin.
    berlin_1D = Measure('temperature', 'berlin', Sampling('1D'))
    berlin_1W = Measure('temperature', 'berlin', Sampling('1W'))


    # Store data in a new collection
    from os import path as os_path
    import pandas
    from oups import ParquetSet

    dirpath = os_path.expanduser('~/Documents/code/data/weather_kbase')
    ps = ParquetSet(dirpath, Measure)
    dummy_data_1D = pd.DataFrame(
                       {'timestamp':pd.date_range('2021/01/01', '2021/01/05', freq='1D'),
    	                'temperature':range(10,15)})
    dummy_data_1W = pd.DataFrame(
                       {'timestamp':pd.date_range('2021/01/01', '2021/01/14', freq='1W'),
    	                'temperature':range(10,12)})
    ps[berlin_1D] = dummy_data_1D
    ps[berlin_1W] = dummy_data_1W

Created folders and files ought to be organized then as illustrated below.

.. code-block::

    data
    |- weather_kbase
       |- temperature-berlin
          |- 1D
          |  |- _common_metadata
          |  |- _metadata
          |  |- part.0.parquet
          |
          |- 1W
             |- _common_metadata
             |- _metadata
             |- part.0.parquet
