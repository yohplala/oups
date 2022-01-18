Why *oups*?
===========

Purpose
-------

Targeting the management of 'large-size' collections of ordered datasets (more specifically time series), *oups* provides convenience class and functions to ease their identification, creation, update, and loading.
These datasets may contain different data (from different channels or feeds) or be the results of different processings of the same raw data.

* *oups* most notably hides path management. By decorating a likewise ``dataclasss`` with ``@toplevel`` decorator, this class is turned into an index generator, with all attributes and functions so that path to related datasets can be generated.
* it also provides an *efficient* update logic suited for ordered datasets (low memory footprint).

Alternatives
------------

Other libraries out there already exist to manage collections of datasets,

* many that I have not tested, for instance `Arctic <https://github.com/man-group/arctic>`_,
* one that I have tested, `pystore <https://github.com/ranaroussi/pystore>`_. Being based on Dask, it supports parallelized reading/writing out of the box. Its update logic can be reviewed in `collection.py <https://github.com/ranaroussi/pystore/blob/ed9beca774312811527c80d199c3cf437623477b/pystore/collection.py#L181>`_. Not elaborating about its `possible performance issues <https://github.com/ranaroussi/pystore/issues/56>`_, and only focusing on this logic applicability, current procedure implies that any duplicate rows be dropped, except last (duplicate considering all columns, but not the index, the latter being necessarily a ``DatetimeIndex`` as per *pystore* implementation). But this hard-coded logic `may not suit all dataflows <https://github.com/ranaroussi/pystore/issues/43>`_.

In comparison, current version of *oups*,

* is not based on Dask but directly on `fastparquet <https://fastparquet.readthedocs.io/en/latest/>`_. No parallelized reading/writing is yet possible.
* provides an *efficient* update function with a user-defined logic for (optionally) dropping duplicates.
