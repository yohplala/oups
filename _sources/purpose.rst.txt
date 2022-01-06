Why *oups*?
===========

Purpose
-------

Being confronted with the management of 'large-size' collections of ordered datasets (more specifically time series), I identified the need to handle them 'easily' (identification, creation, update, loading).
These datasets may contain different data (from different channels or feeds) or be the results of different processings of the same raw data.
To ease their management, i.e. organize how storing these datasets, a first step in `oups` has then been the implementation of ``@toplevel`` and ``@sublevel`` class decorators.

Alternatives
------------

Other libraries out there already exist to manage collections of datasets,

* many that I have not tested, for instance `Arctic <https://github.com/man-group/arctic>`_,
* one that I have tested, `pystore <https://github.com/ranaroussi/pystore>`_. Being based on Dask, it supports parallelized reading/writing out of the box. Its update logic can be reviewed in `collection.py <https://github.com/ranaroussi/pystore/blob/ed9beca774312811527c80d199c3cf437623477b/pystore/collection.py#L181>`_. Not elaborating about its `possible performance issues <https://github.com/ranaroussi/pystore/issues/56>`_, and only focusing on this logic applicability, current procedure implies that any duplicate rows be dropped, except last (duplicate considering all columns, but not the index, the latter being necessarily a ``Datetimeindex`` as per *pystore* implementation). But this hard-coded logic `may not suit all dataflows <https://github.com/ranaroussi/pystore/issues/43>`_.

In comparison, current version of *oups*,

* is not based on Dask but directly on `fastparquet <https://fastparquet.readthedocs.io/en/latest/>`_. No parallelized reading/writing is yet possible.
* only appends new data, without dropping duplicates. It is however a target to propose an *update* function with a user-defined logic for dropping duplicates.
