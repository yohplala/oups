#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:35:00 2021
@author: yoh
"""
from dataclasses import dataclass, is_dataclass, fields
from functools import partial
import re
from typing import Any, Iterator, List, Type, Union

from oups.defines import DIR_SEP


# Float removed to prevent having '.' in field values.
TYPE_ACCEPTED = {int, str}
# Default fields separator, if not modified by user.
DEFAULT_FIELDS_SEP = '-'
# Characters forbidden in field value.
# 'fields_sep' is also included at runtime before check.
FORBIDDEN_CHARS = (DIR_SEP, '.')


def _is_dataclass_instance(obj:Any) -> bool:
    # Check if a class is an instance of a dataclass and not a dataclass
    # itself, as per
    # https://docs.python.org/3/library/dataclasses.html#dataclasses.is_dataclass
    return is_dataclass(obj) and not isinstance(obj, type)

def _dataclass_instance_to_dict(obj:dataclass) -> dict:
    # Shallow copy, not possible to use 'asdict()', as per
    # https://docs.python.org/3/library/dataclasses.html#dataclasses.asdict
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

def _dataclass_instance_to_lists(obj:dataclass) -> Iterator[List[Any]]:
    """
    Yields items as lists of fields values.
    As a new dataclass instance is found, its fields values are yielded in a
    next item, and so on...

    Parameters
    obj : dataclass
        May contain nested dataclass instances.
    Returns
    Iterator[List[Any]]
        Yields list of fields values.
    """
    fields = list(_dataclass_instance_to_dict(obj).values())
    if fields:
        yield fields
        for field in fields:
            if _is_dataclass_instance(field):
                yield from _dataclass_instance_to_lists(field)

def _validate_toplevel_instance(toplevel:dataclass):
    """
    Validate a 'toplevel'-decorated data class instance.
     - Check field type is only among 'int', 'str' or another dataclass
       instance;
     - Check that there is at most only one dataclass instance per nesting
       level, and if present, it is not the 1st field, nor the last field.
     Raise a TypeError or ValueError if instance is not compliant.

    Parameters
    toplevel : dataclass
    """
    forbidden_chars = (toplevel.fields_sep, *FORBIDDEN_CHARS)
    for fields_ in _dataclass_instance_to_lists(toplevel):
        number_of_fields = len(fields_)
        for counter, field in enumerate(fields_):
            if _is_dataclass_instance(field):
                # If a dataclass instance.
                if not counter:
                    # A dataclass instance cannot be the only field.
                    # Detecting if it is in last position suffice, except if
                    # there is only one field, in which case it is also in
                    # 1st position.
                    raise TypeError('a dataclass instance cannot be the only \
field of a level.')
                if counter+1 != number_of_fields:
                    # A dataclass instance cannot be in last position.
                    raise TypeError('a dataclass instance is only possible in \
last position.')
            else:
                # If not a dataclass instance.
                field_as_str = str(field)
                if any([symb in field_as_str for symb in forbidden_chars]):
                    raise ValueError(f'use of a forbidden character among \
{forbidden_chars} is not possible in {field_as_str}.')
            if not ((type(field) in TYPE_ACCEPTED)
                    or _is_dataclass_instance(field)):
                raise TypeError(f'field type {type(field)} not possible.')
    return

def _dataclass_instance_to_str(toplevel:dataclass, as_path:bool=False) -> str:
    """
    Returns a dataclass instance as a string, in which the different levels are
    either separated with 'fields_sep' or with DIR_SEP.

    Parameters
    toplevel : dataclass
    as_path : bool, default False
        Defines separator to be used between levels of the dataclass.
        If True, use DIR_SEP ('/');
        If False, use 'fields_sep'

    Returns
    str
        All fields values, joined:
            - At a same level, using 'fields_sep';
            - Between different levels, either 'fields_sep', either DIR_SEP,
              depending 'as_path'.
    """
    levels_sep = DIR_SEP if as_path else toplevel.fields_sep
    to_str = []
    for fields_ in _dataclass_instance_to_lists(toplevel):
        # Relying on the fact that only the tail can be a dataclass instance.
        to_str.append(toplevel.fields_sep.join(map(str,fields_[:-1])))
    to_str[-1] += f'{toplevel.fields_sep}{str(fields_[-1])}'
    return levels_sep.join(to_str)

def _dataclass_fields_types_to_lists(cls:Type[dataclass]) -> List[List[Any]]:
    """
    Return the type of each field, one list per level, and all levels in a
    list.

    Parameters
    cls : Type[dataclass]
        A dataclass instance or a dataclass.

    Returns
    List[List[Any]]
        List of fields types lists, one list per level.
    """
    types = [[field.type for field in fields(cls)]]
    while is_dataclass(last := types[-1][-1]):
        types.append([field.type for field in fields(last)])
    return types

def _dataclass_instance_from_str(cls:Type[dataclass], string:str)\
    -> Union[dataclass, None]:
    """
    Returns a dataclass instance derived from input string.
    Level separator can either be DIR_SEP or 'cls.fields_sep'.
    If dataclass '__init__' fails, `None` is returned.

    Parameters
    cls : Type[dataclass]
        Dataclass to be used for generating dataclass instance.
    string : str
        String representation of the dataclass instance (using either
        'cls.fields_sep' of DIR_SEP)

    Returns
    Union[dataclass, None]
        Dataclass instance derived from input string.
    """
    types = _dataclass_fields_types_to_lists(cls)
    # Split string depending 'fields_sep' and 'DIR_SEP', into different fields.
    fields_sep = cls.fields_sep
    strings_as_list = re.split(fr'{DIR_SEP}|\{fields_sep}', string)
    # Manages last level first.
    level_types = types.pop() # remove last element
    level_length = len(level_types)
    try:
        level = [field_type(field_as_string) for field_type, field_as_string
                 in zip(level_types, strings_as_list[-level_length:])]
        while types:
            strings_as_list = strings_as_list[:-level_length]
            level_types = types.pop() # remove last element
            level_length = len(level_types)-1
            # Relying on the fact that a dataclass is necessarily the last
            # field.
            level = [field_type(field_as_string)
                     for field_type, field_as_string
                     in zip(level_types[:-1],
                            strings_as_list[-level_length:])]\
                    + [level_types[-1](*level)]
        return cls(*level)
    except (TypeError, ValueError):
        # TypeError if the number of arguments for instantiation of a
        # dataclass is not correct (meaning the split has not been done
        # with the right 'fields_sep' character).
        # ValueError if there is a type mismatch, for instance when 'int'
        # is initialized from a string.
        return None

def _get_depth(obj:Union[dataclass,Type[dataclass]]) -> int:
    """
    Returns number of levels, including 'toplevel'.
    To be decorated with '@property'.
    """
    depth=1
    level=obj
    while is_dataclass(level := fields(level)[-1].type):
        depth+=1
    return depth

class TopLevel(type):
    """
    TopLevel metaclass defining class properties of '@toplevel'-decorated
    class.
    """
    @property
    def fields_sep(cls) -> str:
        return cls._fields_sep
    @property
    def depth(cls) -> int:
        return cls._depth

def toplevel(index_class=None, *, fields_sep:str=DEFAULT_FIELDS_SEP):
    """
    Decorate a class into a dataclass with methods and attributes to use it
    as a dataset index.
    Decorated class has to be defined as one would define a class decorated by
    '@dataclass'.
    '@dataclass' is actually called when decorating with '@toplevel' with
    parameters set to:
        - order=True,
        - frozen=True
    
    'toplevel' is to be used as a decorator, with or without parameter
    'fields_sep'.

    Class instantiation is checked.
      - An instance can only be composed with `int`, 'str' or a dataclass
        object coming in last position;
      - Value of attribute can not incorporate forbidden characters like '/'
        and 'self.fields_sep'.

    Parameters
    fields_sep : str, default '.'
        Character to use as separator between fields of the dataclass.

    Returns
    Decorated class.
    
    Attributes
    fields_sep: str
        Fields separator (can't assign).
    depth: int
        Number of levels, including 'toplevel' (can't assign).
    """
    def tweak(index_class):
        # Re-create 'index_class' as a 'TopLevel'-inheriting class to equip it
        # with class properties 'depth' and 'fields_sep'
        # (as per https://stackoverflow.com/questions/5120688)
        # Explicitely add property to OtherClass.__dict__
        # (as per https://stackoverflow.com/questions/70233891)
        d = dict(index_class.__dict__)
        d.update({'fields_sep': TopLevel.fields_sep, 'depth': TopLevel.depth})
        index_class = TopLevel(index_class.__name__,  index_class.__bases__, d)
        # Wrap with `@dataclass`.
        # TODO
        # When python 3.10 is more wide spread, set 'slot=True' to save RAM.
        index_class = dataclass(index_class, order= True, frozen=True)

        # Equip 'index_class' with what is needed to be a 'toplevel'.
        # Dunders: modified '__init__', modified '__str__'
        # Copy of original __init__ to call it without recursion.
        index_class_init = index_class.__init__
        def __init__(self, *args, **kws):
#            object.__setattr__(self, "_fields_sep", fields_sep)
            index_class_init(self, *args, **kws)
            # Validate dataclass instance.
            _validate_toplevel_instance(self)
        index_class.__init__ = __init__
        index_class.__str__ = _dataclass_instance_to_str

        # Class properties: 'fields_sep', 'depth'
        index_class._fields_sep = fields_sep
        index_class._depth = _get_depth(index_class)

        # Class instance properties: 'to_path'
        _dataclass_instance_to_str_p = partial(_dataclass_instance_to_str,
                                               as_path=True)
        index_class.to_path = property(_dataclass_instance_to_str_p)
        _dataclass_instance_from_path = partial(_dataclass_instance_from_str)

        # Classmethods: 'from_str', 'from_path'.
        index_class.from_path = classmethod(_dataclass_instance_from_path)
        index_class.from_str = classmethod(_dataclass_instance_from_path)

        return index_class

    if index_class:
        # Calling decorator without other parameters.
        return tweak(index_class)
    # Calling decorator with other parameters.
    return tweak

def is_toplevel(toplevel) -> bool:
    """
    Returns 'True' if 'toplevel' (class or instance) has been decorated with
    '@toplevel'. It checks presence 'fields_sep' attribute and 'from_path'
    method.
    """
    return (hasattr(toplevel, 'fields_sep')
            and callable(getattr(toplevel, 'from_path', None)))

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
