Collection indexing
===================

Motivation
----------

Datasets are gathered in a parent directory as a collection. Each of them materialize as parquet files located in a child directory whose naming is derived from a user-defined index.

By formalizing this index through a *likewise dataclass*, index management (user scope) is dissociated from path management (*oups* scope).

Proposal
--------

*oups* provides 2 class decorators for defining an indexing logic.

* ``@toplevel`` is compulsory, and defines naming logic of the first directory level,
* ``@sublevel`` is optional, and can be used as many times as number of sub-directories are required

By splitting indexes into different directory levels, related datasets can be gathered in common directories.
A first level could for instance specify physical quantities in different places, and a second one could specify the sampling frequency of the measures.

Each of these levels is thus specified by a class. Those corresponding to a parent directory necessarily embed as last attribute the sub-level-related class.

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

Created folders and files are then organized as illustrated below.

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

``@toplevel``
-------------

``@toplevel`` accepts an optional ``fields_sep`` parameter to define the character separating fields (by default ``-``). This separator applies to all *levels*.

Decorated class can have any number of attributes (also named *fields*), but only of types ``int`` or ``str``.

If an attribute is a ``@sublevel``-decorated class, it is necessarily positioned last.


``@toplevel`` decorator provides attributes and functions which are used by a ``ParquetSet`` instance to

* generate *paths* from attributes values (``__str__`` and ``to_path`` methods),
* generate class instance (``from_path`` classmethod)

It modifies the ``__init__`` method of decorated class so that attributes values are checked at instantiation, and use of any forbidden character or combination raises related exception.

Lastly, it calls ``@dataclass`` class decorator, with ``order`` and ``frozen`` parameters set as ``True``. This setting enables equality between class instances with same attributes values.

``@sublevel``
-------------

Likewise,

* decorated class can have any number of attributes, but only of types ``int`` or ``str``.
* if yet another deeper *sub-level* is defined (using a ``@sublevel``-decorated class), it necessarily has to be positioned as last attribute.

``@sublevel`` is here only an alias for ``@dataclass``, with ``order`` and ``frozen`` parameters set as ``True``.
