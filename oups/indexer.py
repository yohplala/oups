#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import dataclass, is_dataclass, fields
from functools import partial
import re
from typing import Any, Iterator, List

from oups.defines import DIR_SEP


# Float removed to prevent having '.' in field values.
TYPE_ACCEPTED = {int, str}
# Characters forbidden in field value.
# 'fields_sep' is also included before check.
FORBIDDEN_CHARS = (DIR_SEP, '.')

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

def _validate_top_level_obj(top_level):
    """
    Validate a 'top_level'-decorated data class instance.
     - check field type is only among 'int', 'str' or another dataclass
       instance;
     - check that there is at most only one dataclass instance per nesting
       level, and if present, it is not the 1st field, nor the last field.
     Raise a TypeError if instance is not compliant.

    Parameters
    top_level : top_level dataclass
    """
    forbidden_chars = (top_level._fields_sep, *FORBIDDEN_CHARS)
    for fields_ in _dataclass_instance_to_lists(top_level):
        number_of_fields = len(fields_)
        for counter, field in enumerate(fields_):
            if _is_dataclass_instance(field):
                # If a dataclass instance.
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
            else:
                # If not a dataclass instance.
                field_as_str = str(field)
                if any([symb in field_as_str for symb in forbidden_chars]):
                    raise ValueError(f'Use of a forbidden character among \
{forbidden_chars} is not possible in {field_as_str}.')
            if not ((type(field) in TYPE_ACCEPTED)
                    or _is_dataclass_instance(field)):
                raise TypeError(f'Field type {type(field)} not possible.')
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

def _dataclass_instance_from_str(cls, string:str, fields_sep:str):
    types = _dataclass_fields_types_to_lists(cls)
    # Split string depending 'fields_sep' and 'DIR_SEP', into different fields.
    strings_as_list = re.split(fr'{DIR_SEP}|\{fields_sep}', string)
    # Manages last level first.
    level_types = types.pop() # remove last element
    level_length = len(level_types)
    level = [field_type(field_as_string) for field_type, field_as_string
             in zip(level_types, strings_as_list[-level_length:])]
    while types:
        strings_as_list = strings_as_list[:-level_length]
        level_types = types.pop() # remove last element
        level_length = len(level_types)-1
        # Relying on the fact that a dataclass is necessarily the last field.
        level = [field_type(field_as_string) for field_type, field_as_string
                 in zip(level_types[:-1], strings_as_list[-level_length:])]\
                + [level_types[-1](*level)]
    return cls(*level)

def toplevel(index_class=None, *, fields_sep:str='-'):
    """
    Decorate a class into a dataclass with methods and attributes to use it
    as a dataset index.
    Decorated class has to be defined as one would define a class decorated by
    '@dataclass'.
    '@dataclass' is actually called when decorating with '@toplevel' with
    parameters set to:
        - order=True,
        - frozen=True
    
    'top_level' is to be used as a decorator, with or without parameter
    'fields_sep'.

    Class instanciation is checked.
      - An instance can only be composed with `int`, 'str' or a dataclass
        object coming in last position;
      - Value of attribute can not incorporate forbidden characters like '/'
        and 'self._fields_sep'.

    Parameters
    fields_sep : str, default '.'
        Character to use as separator between fields of the dataclass.

    Returns
    Decorated class.
    """
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
            _validate_top_level_obj(self)
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
                                                fields_sep=fields_sep)
        index_class.from_path = classmethod(_dataclass_instance_from_path)
        index_class.from_str = classmethod(_dataclass_instance_from_path)
        return index_class

    if index_class:
        # Calling decorator without other parameters.
        return tweak(index_class)
    # Calling decorator with other parameters.
    return tweak

def sublevel(index_class):
    """
    Decorator to be used as an alias of '@dataclass' decorator, with parameters
    set to:
        - order=True,
        - frozen=True
    """ 
    # Wrap with `@dataclass`.
    # TODO
    # When python 3.10 is more wide spread, set 'slot=True' to save RAM.
    return dataclass(index_class, order= True, frozen=True)
