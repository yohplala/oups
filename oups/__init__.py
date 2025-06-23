#!/usr/bin/env python3
"""
Created on Wed Dec  1 18:35:00 2021.

@author: yoh

"""
# Import version dynamically from Poetry
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .aggstream import AggStream
from .aggstream import by_x_rows
from .store import OrderedParquetDataset
from .store import Store
from .store import conform_cmidx
from .store import is_toplevel
from .store import sublevel
from .store import toplevel


try:
    __version__ = version("oups")
except PackageNotFoundError:
    # Package is not installed, likely in development
    __version__ = "development"
