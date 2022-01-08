Quickstart
==========

ParquetSet and indexing
-----------------------

An instance of ``ParquetSet`` class gathers a collection of datasets.
``ParquetSet`` instantiation requires the definition of a *collection path* and a dataset *indexing logic*.

**Collection path**

It is directory path (existing or not) where will be (are) gathered directories for each dataset.

**Indexing logic**

A logic is formalized by use of a decorated class. Index themselves are then materialized by instantiating this class, and more specifically by the instance attributes values.

The class itself is declared just as a `dataclass <https://docs.python.org/3/library/dataclasses.html>`_.
``@toplevel`` is then used as a class decorator (and not ``@dataclass``).

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

Writing new data
----------------

.. code-block:: python

    import pandas as pd

    # Index of a first dataset, for some temperature records related to Berlin.
    idx1 = DatasetIndex('germany','berlin')
    # Data to be recorded.
    df1 = pd.DataFrame({'timestamp':pd.date_range('2021/01/01', '2021/01/05', freq='1D'),
    	                'temperature':range(10,15)})
    # Populate parquet collection with a first dataset.
    ps[idx1] = df1

``weather_knowledge_base`` folder has now been created with new data.

.. code-block::

    data
    |- weather_knowledge_base
       |- germany-berlin
          |- _common_metadata
          |- _metadata
          |- part.0.parquet

Reading existing data
---------------------

.. code-block:: python

    # Read data as a pandas dataframe.
    df = ps[idx1].pdf
