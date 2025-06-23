Store Architecture
==================

The ``oups.store`` module provides the core functionality for managing collections of ordered parquet datasets. It consists of three main components working together to provide efficient storage, indexing, and querying of time-series data.

Overview
--------

The store architecture is designed around three key components:

1. **Indexer**: Provides a schema-based indexing system for organizing datasets
2. **OrderedParquetDataset**: Manages individual parquet datasets with ordering validation
3. **Store**: Provides a collection interface for multiple indexed datasets

Main Components
---------------

Indexer
~~~~~~~

The indexer system allows you to define hierarchical schemas for organizing your datasets using dataclasses decorated with ``@toplevel`` and optionally ``@sublevel``. This provides a structured way to organize related datasets in common directories.

**Motivation**

Datasets are gathered in a parent directory as a collection. Each materializes as parquet files located in a child directory whose naming is derived from a user-defined index. By formalizing this index through dataclasses, index management (user scope) is dissociated from path management (*oups* scope).

**Decorators**

*oups* provides 2 class decorators for defining an indexing logic:

- ``@toplevel`` is compulsory, and defines naming logic of the first directory level
- ``@sublevel`` is optional, and can be used as many times as needed for sub-directories

**@toplevel Decorator**

The ``@toplevel`` decorator:

- Generates *paths* from attribute values (``__str__`` and ``to_path`` methods)
- Generates class instances (``from_path`` classmethod)
- Validates attribute values at instantiation
- Calls ``@dataclass`` with ``order`` and ``frozen`` parameters set as ``True``
- Accepts an optional ``fields_sep`` parameter (default ``-``) to define field separators
- Only accepts ``int`` or ``str`` attribute types
- If an attribute is a ``@sublevel``-decorated class, it must be positioned last

**@sublevel Decorator**

The ``@sublevel`` decorator:

- Is an alias for ``@dataclass`` with ``order`` and ``frozen`` set as ``True``
- Only accepts ``int`` or ``str`` attribute types
- If another deeper sub-level is defined, it must be positioned as last attribute

**Hierarchical Example**

.. code-block:: python

    from oups.store import toplevel, sublevel

    @sublevel
    class Sampling:
        frequency: str

    @toplevel
    class Measure:
        quantity: str
        city: str
        sampling: Sampling

    # Define different indexes for temperature in Berlin
    berlin_1D = Measure('temperature', 'berlin', Sampling('1D'))
    berlin_1W = Measure('temperature', 'berlin', Sampling('1W'))

    # When this indexer is connected to a Store, the directory structure will look like:
    # temperature-berlin/
    # ├── 1D/
    # │   ├── file_0000.parquet
    # │   └── file_0001.parquet
    # └── 1W/
    #     ├── file_0000.parquet
    #     └── file_0001.parquet

**Simple Example**

.. code-block:: python

    from oups.store import toplevel

    @toplevel
    class TimeSeriesIndex:
        symbol: str
        date: str

    # This creates a schema where datasets are organized as:
    # symbol-date/ (e.g., "AAPL-2023.01.01/")

OrderedParquetDataset
~~~~~~~~~~~~~~~~~~~~~

``OrderedParquetDataset`` is the core class for managing individual parquet datasets with strict ordering validation. It provides:

**Key Features:**

- **Ordered Storage**: Data is stored in row groups ordered by a specified column
- **Incremental Updates**: Efficiently merge new data with existing data
- **Row Group Management**: Automatic splitting and merging of row groups
- **Metadata Tracking**: Comprehensive metadata for each row group
- **Metadata Updates**: Add, update, or remove custom key-value metadata
- **Duplicate Handling**: Configurable duplicate detection and removal
- **Write Optimization**: Configurable row group sizes and merge strategies

**File Structure:**

.. code-block::

    parent_directory/
    ├── my_dataset/                # Dataset directory
    │   ├── file_0000.parquet      # Row group files
    │   └── file_0001.parquet
    ├── my_dataset_opdmd           # Metadata file
    └── my_dataset.lock            # Lock file

**Example:**

.. code-block:: python

    from oups.store import OrderedParquetDataset
    import pandas as pd

    # Create or load a dataset
    dataset = OrderedParquetDataset("/path/to/dataset", ordered_on="timestamp")

    # Write data
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=1000),
        "value": range(1000)
    })
    dataset.write(df=df)

    # Read data back
    result = dataset.to_pandas()

Store
~~~~~

The ``Store`` class provides a collection interface for managing multiple ``OrderedParquetDataset`` instances organized according to an indexer schema.

**Key Features:**

- **Schema-based Organization**: Uses indexer schemas for dataset discovery
- **Lazy Loading**: Datasets are loaded on-demand
- **Collection Interface**: Dictionary-like access to datasets
- **Cross-dataset Operations**: Advanced querying across multiple datasets
- **Automatic Discovery**: Finds existing datasets matching the schema

**Example:**

.. code-block:: python

    from oups.store import Store
    from oups.store import toplevel

    @toplevel
    class StockIndex:
        symbol: str
        year: str

    # Create store
    store = Store("/path/to/data", StockIndex)

    # Access datasets
    aapl_2023 = store[StockIndex("AAPL", "2023")]

    # Iterate over all datasets
    for key in store:
        dataset = store[key]
        print(f"Dataset {key} has {len(dataset)} row groups")

Advanced Features
-----------------

Write Method
~~~~~~~~~~~~

The ``write()`` function provides advanced data writing capabilities:

**Parameters:**

- ``row_group_target_size``: Control row group sizes (int or pandas frequency string)
- ``duplicates_on``: Specify columns for duplicate detection
- ``max_n_off_target_rgs``: Control row group coalescing behavior
- ``key_value_metadata``: Store custom metadata (supports add/update/remove operations)

**Example:**

.. code-block:: python

    from oups.store import write

    # Write with time-based row groups and metadata
    write(
        "/path/to/dataset",
        ordered_on="timestamp",
        df=df,
        row_group_target_size="1D",  # One row group per day
        duplicates_on=["timestamp", "symbol"],
        key_value_metadata={
            "source": "market_data",
            "version": "2.1",
            "processed_by": "data_pipeline"
        }
    )

    # Update existing metadata (add new, update existing, remove with None)
    write(
        "/path/to/dataset",
        ordered_on="timestamp",
        df=new_df,
        key_value_metadata={
            "version": "2.2",        # Update existing
            "last_updated": "2023-12-01",  # Add new
            "processed_by": None     # Remove existing
        }
    )

iter_intersections
~~~~~~~~~~~~~~~~~~

The ``iter_intersections()`` method enables efficient querying across multiple datasets with overlapping ranges:

**Key Features:**

- **Range Queries**: Query specific ranges (time, numeric, etc.) across multiple datasets
- **Intersection Detection**: Automatically finds overlapping row groups
- **Memory Efficient**: Streams data without loading entire datasets
- **Synchronized Iteration**: Iterates through multiple datasets in sync

**Example:**

.. code-block:: python

    # Query multiple datasets for overlapping data
    keys = [StockIndex("AAPL", "2023"), StockIndex("GOOGL", "2023")]

    for intersection in store.iter_intersections(
        keys,
        start=pd.Timestamp("2023-01-01"),
        end_excl=pd.Timestamp("2023-02-01")
    ):
        for key, df in intersection.items():
            print(f"Data from {key}: {len(df)} rows")

Best Practices
--------------

1. **Indexer Design**: Design your indexer schema to match your data access patterns
2. **Ordered Column**: Choose an appropriate column for ordering (typically timestamp)
3. **Row Group Size**: Balance between query performance and storage efficiency
4. **Duplicate Handling**: Use ``duplicates_on`` when data quality is a concern
5. **Metadata**: Use key-value metadata to store important dataset information


See Also
--------

- :doc:`api` - Complete API reference
- :doc:`quickstart` - Getting started guide
