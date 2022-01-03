Install
=======

`Install poetry <https://python-poetry.org/docs/master/#installation>`_, then *oups*, first cloning its repository.

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -
    git clone https://github.com/yohplala/oups
    cd oups
    poetry install


Requirements will be taken care of by `poetry`.

* `sortedcontainers <http://www.grantjenks.com/docs/sortedcontainers/>`_
* `pandas <https://pandas.pydata.org/>`_
* `vaex <https://vaex.io/docs/index.html>`_
* fastparquet, specific branch (`PR pending <https://github.com/dask/fastparquet/pull/712>`_)
