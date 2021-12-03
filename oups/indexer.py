#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import dataclass, is_dataclass, fields
from functools import partial
from typing import Any, Iterator, List

from oups.defines import DIR_SEP


# Float removed to prevent having '.' in field values.
TYPE_ACCEPTED = {int, str}

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
    forbidden_chars = (DIR_SEP, top_level._fields_sep)
    for fields_ in _dataclass_instance_to_lists(top_level):
        number_of_fields = len(fields_)
        for counter, field in enumerate(fields_):
            if _is_dataclass_instance(field):
                if not counter:
                    # A dataclass instance cannot be the only field.
                    # Detecting if it is in last position suffice, except if
                    # there is only one field, in which case it is also in
                    # 1st position.
                    raise TypeError('A dataclass instance cannot be the only \
field of a level.')
                if counter+1 != number_of_fields:
                    # A dataclass instance cannot be in last position.
                    raise TypeError('A dataclass instance is only possible in \
last position.')
            if not ((type(field) in TYPE_ACCEPTED)
                    or _is_dataclass_instance(field)):
                raise TypeError(f'Field type {type(field)} not possible.')
            field_as_str = str(field)
            if any([symb in field_as_str for symb in forbidden_chars]):
                raise ValueError(f'Use of a forbidden character among \
{forbidden_chars} is not possible in field {field_as_str}.')
    return

def _dataclass_instance_to_str(top_level, as_path:bool=False) -> str:
    levels_sep = DIR_SEP if as_path else top_level._fields_sep
    to_str = []
    for fields_ in _dataclass_instance_to_lists(top_level):
        # Relying on the fact that only the tail can be a dataclass instance.
        to_str.append(top_level._fields_sep.join(map(str,fields_[:-1])))
    to_str[-1] += f'{top_level._fields_sep}{str(fields_[-1])}'
    return levels_sep.join(to_str)

def _dataclass_fields_types_to_lists(cls) -> List[List[Any]]:
    """
    Return the type of each field, one list per level, and all levels in a
    list.

    Parameters
    cls : dataclass
        A dataclass instance or a dataclass.

    Returns
    List[List[Any]]
        List of fields types lists, one list per level.
    """
    types = [[field.type for field in fields(cls)]]
    while is_dataclass(last := types[-1][-1]):
        types.append([field.type for field in fields(last)])
    return types

def _dataclass_instance_from_str(cls, string:str, level_sep:str):
    types = _dataclass_fields_types_to_lists(cls)
    # Split string depending 'level_sep', into different levels.
    string_as_list = string.split(level_sep)
    # Manages last level first.
    level_types = types.pop() # remove last element
    level_length = len(level_types)
    level = list(map(level_types, string_as_list[-level_length:]))
    while types:
        string_as_list = string_as_list[:level_length]
        level_types = types.pop() # remove last element
        level_length = len(level_types)-1
        # Relying on the fact that a dataclass is necessarily the last field.
        level = list(map(level_types[:-1],
                              string_as_list[-level_length:])) \
                     + level_types[-1](*level)
    return level

# Not using '.' for 'fields_sep' as '.' can also be found in floats.
# It is not accepted by vaex. 
def top_level(index_class=None, *, fields_sep:str='-'):
    def tweak(index_class):
        # Wrap with `@dataclass`.
        # TODO
        # When python 3.10 is more wide spread, set 'slot=True' to save RAM.
        index_class = dataclass(index_class, order= True, frozen=True)
        # Copy of original __init__ to call it without recursion.
        index_class_init = index_class.__init__
        def __init__(self, *args, **kws):
#            object.__setattr__(self, "_fields_sep", fields_sep)
            index_class_init(self, *args, **kws)
            # Validate dataclass instance.
            _validate_top_level(self)
        # Set modified __init__, new methods, and new attributes.
        index_class.__init__ = __init__
        # Need to define '_fields_sep' as class attribute, as used in
        # 'from_path' and 'from_str'.
        index_class._fields_sep = fields_sep
        index_class.__str__ = _dataclass_instance_to_str
        _dataclass_instance_to_str_p = partial(_dataclass_instance_to_str,
                                               as_path=True)
        index_class.to_path = property(_dataclass_instance_to_str_p)
        _dataclass_instance_from_path = partial(_dataclass_instance_from_str,
                                                level_sep=DIR_SEP)
        index_class.from_path = classmethod(_dataclass_instance_from_path)
        _dataclass_instance_from_str_p = partial(_dataclass_instance_from_str,
                                                 level_sep=fields_sep)
        index_class.from_path = classmethod(_dataclass_instance_from_str_p)
        return index_class

    if index_class:
        # Calling decorator without other parameters.
        return tweak(index_class)
    # Calling decorator with other parameters.
    return tweak
