Tutorial
========

This guide will get you started with the ``oups.store`` module for managing ordered parquet datasets.

Basic Concepts
--------------

The store module is built around three key concepts:

1. **Indexer**: Defines how datasets are organized using dataclass schemas
2. **OrderedParquetDataset**: Individual datasets with validated ordering
3. **Store**: Collection manager for multiple datasets

Let's walk through a complete example.

Understanding Types and Parameters
----------------------------------

Before diving into examples, it's helpful to understand the key types and parameters used throughout the oups library:

**Index Types**

Indexer classes are dataclasses decorated with ``@toplevel`` that define the schema for organizing datasets. They can optionally include ``@sublevel`` classes for hierarchical organization.

**Ordered Column Types**

The ``ordered_on`` parameter accepts:

- ``str``: Single column name (e.g., ``"timestamp"``)
- ``Tuple[str]``: Multi-index column name for hierarchical columns (e.g., ``("date", "time")``)

**Row Group Target Size Types**

The ``row_group_target_size`` parameter accepts:

- ``int``: Target number of rows per row group (e.g., ``10000``)
- ``str``: Pandas frequency string for time-based grouping (e.g., ``"1D"`` for daily, ``"1H"`` for hourly)

**Key-Value Metadata**

Custom metadata stored as ``Dict[str, str]`` alongside parquet files. This can include source information, processing parameters, or any other relevant metadata.

Setting Up an Indexer
---------------------

First, define how you want to organize your datasets using a class decorated with ``@toplevel``:

.. code-block:: python

    from oups.store import toplevel

    @toplevel
    class WeatherIndex:
        country: str
        city: str

This creates a schema where datasets will be organized in directories like ``germany-berlin/``, ``france-paris/``, etc.

Creating a Store
-----------------

Create a store instance that will manage your collection of datasets:

.. code-block:: python

    from oups.store import Store
    import os

    # Define the base directory for your data collection
    data_path = os.path.expanduser('~/Documents/data/weather_data')

    # Create the store
    store = Store(data_path, WeatherIndex)

Working with Datasets
----------------------

**Writing Data**

.. code-block:: python

    import pandas as pd

    # Create an index for Berlin weather data
    berlin_key = WeatherIndex('germany', 'berlin')

    # Create sample data
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=30, freq='D'),
        'temperature': range(20, 50),
        'humidity': range(30, 60)
    })

    # Get reference to the dataset (initializes the dataset if it doesn't exist)
    berlin_dataset = store[berlin_key]

    # Write the data with timestamp ordering
    berlin_dataset.write(df=df, ordered_on='timestamp')

The directory structure will now look like:

.. code-block::

    weather_data/
    ├── germany-berlin/
    │   ├── file_0000.parquet
    │   └── file_0001.parquet
    ├── germany-berlin_opdmd
    └── germany-berlin.lock

**Reading Data**

.. code-block:: python

    # Read all data back as a pandas DataFrame
    result_df = berlin_dataset.to_pandas()
    print(f"Dataset has {len(result_df)} rows")

    # Check dataset metadata
    print(f"Ordered on: {berlin_dataset.ordered_on}")
    print(f"Number of row groups: {len(berlin_dataset)}")

Adding More Data
-----------------

**Incremental Updates**

.. code-block:: python

    # Add more recent data
    new_df = pd.DataFrame({
        'timestamp': pd.date_range('2023-02-01', periods=15, freq='D'),
        'temperature': range(15, 30),
        'humidity': range(40, 55)
    })

    # This will merge with existing data in the correct order
    berlin_dataset.write(df=new_df, ordered_on='timestamp')

**Adding Another City**

.. code-block:: python

    # Add data for Paris
    paris_key = WeatherIndex('france', 'paris')
    paris_df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=25, freq='D'),
        'temperature': range(25, 50),
        'humidity': range(35, 60)
    })

    store[paris_key].write(df=paris_df, ordered_on='timestamp')

Exploring Your Store
---------------------

**List All Datasets**

.. code-block:: python

    print(f"Total datasets: {len(store)}")

    for key in store:
        dataset = store[key]
        print(f"{key}: {len(dataset)} row groups")

**Query Multiple Datasets**

.. code-block:: python

    # Query data from multiple cities for a specific time range
    keys = [WeatherIndex('germany', 'berlin'), WeatherIndex('france', 'paris')]

    start_date = pd.Timestamp('2023-01-15')
    end_date = pd.Timestamp('2023-01-25')

    for intersection in store.iter_intersections(keys, start=start_date, end_excl=end_date):
        for key, df in intersection.items():
            print(f"Data from {key}: {len(df)} rows")
            print(f"Temperature range: {df['temperature'].min()}-{df['temperature'].max()}")

Advanced Features
-----------------

**Hierarchical Indexing**

For more complex organization, you can create hierarchical indexers using ``@sublevel``:

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
    store_hierarchical = Store("/path/to/financial_data", HierarchicalIndex)

**Time-based Row Groups**

.. code-block:: python

    from oups.store import write

    # Organize data into daily row groups
    write(
        store[berlin_key],
        ordered_on='timestamp',
        df=df,
        row_group_target_size='1D'  # One row group per day
    )

**Advanced Write Options**

The ``write`` function supports many advanced options for optimizing storage and handling duplicates:

.. code-block:: python

    from oups.store import write

    # Advanced write with all options
    write(
        "/path/to/dataset",
        ordered_on="timestamp",
        df=df,
        row_group_target_size="1D",  # Daily row groups
        duplicates_on=["timestamp", "symbol"],  # Drop duplicates based on these columns
        max_n_off_target_rgs=2,  # Coalesce small row groups
        key_value_metadata={
            "source": "bloomberg",
            "version": "1.0",
            "processed_by": "data_pipeline"
        }
    )

**Handling Duplicates**

.. code-block:: python

    # Remove duplicates based on timestamp and location
    write(
        store[berlin_key],
        ordered_on='timestamp',
        df=df_with_duplicates,
        duplicates_on=['timestamp']  # Drop rows with same timestamp
    )

**Custom Metadata**

.. code-block:: python

    # Add metadata to your dataset
    write(
        store[berlin_key],
        ordered_on='timestamp',
        df=df,
        key_value_metadata={
            'source': 'weather_station_001',
            'units': 'celsius',
            'version': '1.0'
        }
    )

    # Update existing metadata (add new, update existing, remove with None)
    write(
        store[berlin_key],
        ordered_on='timestamp',
        df=new_df,
        key_value_metadata={
            'version': '1.1',        # Update existing
            'last_updated': '2023-12-01',  # Add new
            'processed_by': None     # Remove existing
        }
    )

**Cross-Dataset Queries**

For more complex scenarios, you can query multiple datasets simultaneously:

.. code-block:: python

    # Define a financial indexer for cross-dataset queries
    @toplevel
    class StockIndex:
        category: str
        subcategory: str

    # Query multiple datasets simultaneously
    keys = [StockIndex("stocks", "tech"), StockIndex("stocks", "finance")]

    for intersection in store.iter_intersections(
        keys,
        start=pd.Timestamp("2023-01-01"),
        end_excl=pd.Timestamp("2023-02-01")
    ):
        for key, df in intersection.items():
            print(f"Processing {key}: {len(df)} rows")

Next Steps
----------

- Learn more about indexing and explore the complete :doc:`store` architecture documentation
- Review the full :doc:`api` reference
- Understand the :doc:`purpose` and design philosophy
