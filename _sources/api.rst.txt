API Reference
=============

This section provides detailed API documentation for the main store components.

Indexer Functions
-----------------

.. autofunction:: oups.store.toplevel

.. autofunction:: oups.store.sublevel

.. autofunction:: oups.store.is_toplevel

Core Classes
------------

OrderedParquetDataset
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: oups.store.OrderedParquetDataset
   :members:
   :show-inheritance:

Store
~~~~~

.. autoclass:: oups.store.Store
   :members:
   :show-inheritance:

Write Operations
----------------

.. autofunction:: oups.store.write

Utility Functions
-----------------

.. autofunction:: oups.store.check_cmidx

.. autofunction:: oups.store.conform_cmidx

Type Definitions
----------------

The following are important type definitions used throughout the store module:

**Index Types**

Indexer classes are dataclasses decorated with ``@toplevel`` that define the schema for organizing datasets.

**Ordered Column Types**

The ``ordered_on`` parameter accepts:

- ``str``: Single column name
- ``Tuple[str]``: Multi-index column name (for hierarchical columns)

**Row Group Target Size Types**

The ``row_group_target_size`` parameter accepts:

- ``int``: Target number of rows per row group
- ``str``: Pandas frequency string (e.g., "1D", "1H") for time-based grouping

**Key-Value Metadata**

Custom metadata stored as ``Dict[str, str]`` alongside parquet files.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from oups.store import toplevel, Store, OrderedParquetDataset
    import pandas as pd

    # Define indexer schema
    @toplevel
    class MyIndex:
        category: str
        subcategory: str

    # Create store
    store = Store("/path/to/data", MyIndex)

    # Create sample data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=1000),
        "value": range(1000)
    })

    # Access dataset and write data
    key = MyIndex("stocks", "tech")
    dataset = store[key]
    dataset.write(df=df, ordered_on="timestamp")

Advanced Write Options
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from oups.store import write

    # Time-based row groups with duplicate handling
    write(
        "/path/to/dataset",
        ordered_on="timestamp",
        df=df,
        row_group_target_size="1D",  # Daily row groups
        duplicates_on=["timestamp", "symbol"],  # Drop duplicates
        max_n_off_target_rgs=2,  # Coalesce small row groups
        key_value_metadata={"source": "bloomberg", "version": "1.0"}
    )

Cross-Dataset Queries
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Query multiple datasets simultaneously
    keys = [MyIndex("stocks", "tech"), MyIndex("stocks", "finance")]

    for intersection in store.iter_intersections(
        keys,
        start=pd.Timestamp("2023-01-01"),
        end_excl=pd.Timestamp("2023-02-01")
    ):
        for key, df in intersection.items():
            print(f"Processing {key}: {len(df)} rows")

Hierarchical Indexing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from oups.store import toplevel, sublevel

    @sublevel
    class DateInfo:
        year: str
        month: str

    @toplevel
    class HierarchicalIndex:
        symbol: str
        date_info: DateInfo

    # This creates paths like: AAPL/2023-01/
    key = HierarchicalIndex("AAPL", DateInfo("2023", "01"))
