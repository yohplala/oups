#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import asdict
import pytest

from oups import is_toplevel, sublevel, toplevel
from oups.defines import DIR_SEP


# check error message when using a class not being a toplevel.
# check with a directory that can't be materialize into a key.
