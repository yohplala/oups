Why *oups*?
===========

Purpose
-------

*oups* (Ordered Updatable Parquet Store) is designed for managing large collections of ordered datasets, particularly time-series data. It provides a comprehensive framework for efficient storage, indexing, and querying of structured data with validated ordering.

**Key Design Goals:**

* **Schema-driven Organization**: Use dataclass schemas to automatically organize and discover datasets
* **Ordered Storage Validation**: Verify strict ordering within datasets for optimal query performance
* **Efficient Updates**: Support incremental data updates with intelligent merging strategies
* **Memory Efficiency**: Minimize memory footprint during read/write operations
* **Concurrent Access**: Provide safe concurrent access through file-based locking
* **Flexible Querying**: Enable cross-dataset queries and range-based data retrieval

**Core Features:**

* **Hierarchical Indexing**: Define complex organizational schemas using ``@toplevel`` decorated dataclasses
* **Row Group Management**: Use of parquet file structure to optimize both storage and query performance
* **Duplicate Handling**: Configurable duplicate detection and removal
* **Metadata Support**: Rich metadata storage alongside datasets
* **Range Queries**: Efficient querying across time ranges and multiple datasets simultaneously

Use Cases
---------

You may think of using *oups* for:

* **Financial Time Series**: Managing market data, trading records, and risk metrics across multiple instruments
* **IoT Data Collection**: Organizing sensor data from multiple devices and locations
* **Analytics Pipelines**: Storing intermediate and final results of data processing workflows
* **Research Data**: Managing experimental datasets with complex organizational requirements

Alternatives
------------

Several alternatives exist for managing dataset collections:

**Arctic (MongoDB-based)**
   - Provides powerful time-series storage
   - Requires MongoDB infrastructure
   - More complex deployment and maintenance

**PyStore (Dask-based)**
   - Supports parallelized operations
   - Less flexible organizational schemas
   - `Performance concerns <https://github.com/ranaroussi/pystore/issues/56>`_ in some scenarios

**DuckDB or DataFusion**
   - Excellent query performance
   - SQL-based querying

**Direct Parquet + File Management**
   - Maximum control over file structure
   - Requires implementing indexing, updates, and concurrency manually
   - This is how *oups* started

*oups* Advantages
------------------

Compared to these alternatives, *oups* offers:

* **Pure Python Implementation**: No external database dependencies
* **Flexible Duplicate Handling**: User-defined logic for handling duplicate rows
* **Automated Path Management**: Schema-driven directory organization
* **Incremental Updates**: Efficient merging of new data with existing datasets
* **Ordering Validation**: Built-in verification of data ordering for optimal performance
* **Simple API**: An interface not requiring SQL knowledge
* **Lock-based Concurrency**: Safe concurrent access without complex coordination

Example Comparison
-------------------

**Traditional Approach:**

.. code-block:: python

    # Manual path management
    path = f"/data/{symbol}/{year}/{month}/data.parquet"

    # Manual duplicate handling
    existing_df = pd.read_parquet(path)
    new_df = pd.concat([existing_df, new_data])
    new_df = new_df.drop_duplicates().sort_values('timestamp')
    new_df.to_parquet(path)

**With oups:**

.. code-block:: python

    @toplevel
    class DataIndex:
        symbol: str
        year: str
        month: str

    store = Store("/data", DataIndex)
    key = DataIndex("AAPL", "2023", "01")

    # Automatic path management, duplicate handling, and ordering
    store[key].write(
        df=new_data,
        ordered_on='timestamp',
        duplicates_on=['timestamp', 'symbol']
    )
