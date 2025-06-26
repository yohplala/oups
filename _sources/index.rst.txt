.. oups documentation master file, created by
   sphinx-quickstart on Sat Jan  1 09:50:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

oups
====

*oups* stands for **Ordered Updatable Parquet Store**.

*oups* is a Python library that provides powerful tools for managing collections of ordered parquet datasets. It enables efficient storage, indexing, and querying of time-series data with validated ordering and good performance.

Key Features
------------

- **Ordered Storage**: Validates data ordering within datasets
- **Schema-based Indexing**: Hierarchical organization using dataclass schemas
- **Incremental Updates**: Efficiently merge new data with existing datasets
- **Row Group Management**: Optimizing storage layout
- **Duplicate Handling**: Configurable duplicate detection and removal
- **Lock-based Concurrency**: Safe concurrent access to datasets
- **Cross-dataset Queries**: Query multiple datasets simultaneously

Documentation
-------------

.. toctree::
   :maxdepth: 2

   install
   tutorial
   purpose
   store
   api

Indices and Tables
------------------

* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
