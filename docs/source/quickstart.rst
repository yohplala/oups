Quickstart
==========

ParquetSet and indexing
-----------------------

Gathering a collection of datasets into a *oups* ``ParquetSet`` requires to define a *collection path* and a *dataset indexing logic*.

**Collection path**

It is directory path (existing or not) where will be (are) gathered directories for each dataset.

**Indexing logic**

It is a class whose attributes values define an index for a given dataset, and is declared just as a `dataclass <https://docs.python.org/3/library/dataclasses.html>`_.
``@toplevel`` is then used as a class decorator (and not ``@dataclass``) so that naming of each dataset directory can be derived appropriately.

.. code-block:: python

    from os import path as os_path
    from oups import ParquetSet, toplevel

    # Define the indexing logic to generate each individual dataset folder name.
    @toplevel
    class DatasetIndex:
        country: str
        city: str

    # Initialize a parquet dataset collection,
    # specifying collection path and indexing logic.
    dirpath = os_path.expanduser('~/Documents/code/data/weather_knowledge_base')
    ps = ParquetSet(dirpath, DatasetIndex)

Writing new data
----------------

.. code-block:: python

    import pandas as pd

    # Index of a first dataset, for some temperature records related to Berlin.
    idx1 = DatasetIndex('germany','berlin')
    df1 = pd.DataFrame({'timestamp':pd.date_range('2021/01/01', '2021/01/05', freq='1D'),
    	                'temperature':range(10,15)})
    # Initiate the new parquet dataset.
    ps[idx1] = df1

``weather_knowledge_base`` folder has now been created, and populated with new data.

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
