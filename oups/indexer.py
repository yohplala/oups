#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import dataclass, is_dataclass, fields
from functools import partial
from typing import Any, Iterator, List


TYPE_ACCEPTED = {int, float, str}
DIR_SEP = '/'

def _is_dataclass_instance(obj):
    return is_dataclass(obj) and not isinstance(obj, type)

def _dataclass_to_dict(obj):
    # Shallow copy, not possible to use 'asdict()'.
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

def _dataclass_instance_to_lists(obj) -> Iterator[List[Any]]:
    """
    Generator.
    Return items as lists of fields values.
    For each dataclass instance found, its fields values are returned again as
    a list in a next item, and so on...

    Parameters
    obj : dataclass
        May contain nested dataclass objects.
    Returns
    Iterator[List[Any]]
        Yields list of fields values.
    """
    fields = list(_dataclass_to_dict(obj).values())
    if fields:
        yield fields
        for field in fields:
            if _is_dataclass_instance(field):
                yield from _dataclass_instance_to_lists(field)

def _validate_top_level(top_level):
    """
    Validate a 'top_level'-decorated data class instance.
     - check only 'int', 'str', 'float' or another dataclass instance are used;
     - check that there is at most only one dataclass instance per nesting
       level, and if present, it is not the 1st field, nor the last field.
     Raise a TypeError if instance is not compliant.

    Parameters
    top_level : top_level dataclass
    """
    for fields_ in _dataclass_instance_to_lists(top_level):
        number_of_fields = len(fields_)
        for counter, field in enumerate(fields_):
            if _is_dataclass_instance(field):
                if not counter:
                    # A dataclass instance cannot be only field.
                    raise TypeError('A dataclass instance cannot be the only \
field of a level.')
                if counter+1 != number_of_fields:
                    # A dataclass instance is identified not in last position.
                    raise TypeError('A dataclass instance is only possible in \
last position.')
            if not ((type(field) in TYPE_ACCEPTED)
                    or _is_dataclass_instance(field)):
                raise TypeError(f'Field type {type(field)} not possible.')

def _dataclass_instance_to_str(top_level, as_path:bool=False):
    levels_sep = DIR_SEP if as_path else top_level._fields_sep
    to_str = []
    for fields_ in _dataclass_instance_to_lists(top_level):
        # Relying on the fact that only the tail can be a dataclass instance.
        to_str.append(top_level._fields_sep.join(map(str,fields_[:-1])))
    to_str[-1] += f'{top_level._fields_sep}{str(fields_[-1])}'
    return levels_sep.join(to_str)

# Not using '.' for 'fields_sep' as '.' can also be found in floats. It will
# not be possible then to re-create from a string the indexer class. 
def top_level(index_class=None, *, fields_sep:str='-'):
    def tweak(index_class):
        # Wrap with `@dataclass`.
        # TODO
        # When python 3.10 is more wide spread, set 'slot=True' to save RAM.
        index_class = dataclass(index_class, order= True, frozen=True)
        # Copy of original __init__ to call it without recursion.
        index_class_init = index_class.__init__
        def __init__(self, *args, **kws):
            object.__setattr__(self, "_fields_sep", fields_sep)
            index_class_init(self, *args, **kws)
            # Validate dataclass instance.
            _validate_top_level(self)
    
        index_class.__init__ = __init__
        index_class.__str__ = _dataclass_instance_to_str
        _dataclass_instance_to_str_p = partial(_dataclass_instance_to_str,
                                               as_path=True)
        index_class.to_path = property(_dataclass_instance_to_str_p)
        return index_class

    if index_class:
        # Calling decorator without other parameters.
        return tweak(index_class)
    # Calling decorator with other parameters.
    return tweak
