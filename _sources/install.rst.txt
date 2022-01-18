Install
=======

`Install poetry <https://python-poetry.org/docs/master/#installation>`_, then *oups*, first cloning its repository.

.. code-block:: bash

    # Install poetry.
    curl -sSL https://install.python-poetry.org | python3 -
    # Clone oups repo.
    git clone https://github.com/yohplala/oups
    # Install oups.
    cd oups
    poetry install
    # Test.
    poetry run pytest

Requirements will be taken care of by `poetry`.

* `sortedcontainers <http://www.grantjenks.com/docs/sortedcontainers/>`_
* `pandas <https://pandas.pydata.org/>`_
* `vaex <https://vaex.io/docs/index.html>`_
* `fastparquet <https://github.com/dask/fastparquet>`_
