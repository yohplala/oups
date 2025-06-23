Installation
============

*oups* can be installed from source using poetry or pip. The library requires Python 3.10 or higher.

.. note::
    *oups* is currently not published to PyPI, so installation must be done from the source repository.

Using Poetry (Recommended)
---------------------------

For development or if you prefer poetry:

.. code-block:: bash

    # Install poetry if you don't have it
    curl -sSL https://install.python-poetry.org | python3 -

    # Clone and install oups
    git clone https://github.com/yohplala/oups
    cd oups
    poetry install

    # Run tests
    poetry run pytest

Using Pip (Development Install)
-------------------------------

For development with editable install:

.. code-block:: bash

    git clone https://github.com/yohplala/oups
    cd oups
    pip install -e .

    # Run tests
    pytest

Dependencies
------------

*oups* automatically installs these required dependencies:

* `pandas <https://pandas.pydata.org/>`_ (>=2.2.3) - Data manipulation
* `numpy <https://numpy.org/>`_ (>=2.0) - Numerical computations
* `fastparquet <https://github.com/dask/fastparquet>`_ (>=2023.10.1) - Parquet file handling
* `numba <https://numba.pydata.org/>`_ (>=0.61.2) - JIT compilation for performance
* `sortedcontainers <http://www.grantjenks.com/docs/sortedcontainers/>`_ (>=2.4.0) - Efficient data structures
* `flufl-lock <https://fluflock.readthedocs.io/>`_ (>=8.2.0) - File locking
* `joblib <https://joblib.readthedocs.io/>`_ (>=1.3.2) - Parallel processing
* `cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ (>=3.1.1) - Serialization
* `arro3-core <https://github.com/tradingsolutions/arro3>`_ (>=0.4.6) - Arrow data processing
* `arro3-io <https://github.com/tradingsolutions/arro3>`_ (>=0.4.6) - Arrow I/O operations

Verification
------------

To verify your installation:

.. code-block:: python

    import oups
    print(oups.__version__)

    # Basic functionality test
    from oups.store import toplevel

    @toplevel
    class TestIndex:
        name: str
        version: int

    # Test creating and using the index
    test_idx = TestIndex("example", 1)
    print(f"Index string representation: {test_idx}")
    print("Installation successful!")
