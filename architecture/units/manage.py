from typing import List, Set, Dict, Iterator
from collections.abc import MutableSequence
from .base import BaseUnit
from ..utils.list_operator import find_index as global_find_index, ReadOnlyList

def check_units(units : List[BaseUnit]):
    ids = set()

    for unit in units:
        if unit.id in ids:
            raise TypeError(f"Unit {unit.id} duplicates!")
        ids.add(unit.id)

    return ids

def check_contain(unit : BaseUnit | str, ids : Set):
    _id = unit
    if isinstance(unit, BaseUnit):
        _id = unit.id
    return _id in ids

def find_index(unit_id : str, units : List[BaseUnit]):
    def access_key(unit : BaseUnit):
        return unit.id
    return global_find_index(unit_id, units, access_key)


class ReadOnlyUnit(ReadOnlyList):
    def __init__(self, units : List[BaseUnit], *args, **kwargs):
        super().__init__(units)
        self.__ids = check_units(units)
        
        self.__lookup : Dict[str, BaseUnit] = dict()
        self.__build_lookup(units)

    def __build_lookup(self, units : List[BaseUnit]):
        for unit in units:
            self.__lookup[unit.id] = unit

    def __contains__(self, unit : BaseUnit | str):
        return check_contain(unit, self.__ids)
    
    def find(self, unit_id : str):
        return self.__lookup[unit_id]
    
    def __iter__(self) -> Iterator[BaseUnit]:
        return super().__iter__()

class MutableUnit(MutableSequence):
    def __init__(self, units : List[BaseUnit] = [],*args, **kwargs):
        super().__init__()
        self._units = units
        self.__ids = check_units(units)

    def __len__(self):
        return len(self._units)
    
    def __getitem__(self, index):
        return self._units[index]
    
    def __contains__(self, unit : BaseUnit | str):
        return check_contain(unit, self.__ids)
    
    def __iter__(self):
        return iter(self._units)
    
    def __delitem__(self, index):
        pass

    def __setitem__(self, index, value):
        pass

    def insert(self, index, value):
        pass

    def append(self, unit : BaseUnit):
        if not isinstance(unit, BaseUnit):
            raise AttributeError("unit must be inheriented from BaseUnit!")

        if unit.id in self.__ids:
            raise ValueError(f"Unit {unit.id} existed!")
        
        self._units.append(unit)
        self.__ids.add(unit.id)
    
    def pop(self, unit : BaseUnit | str):
        _id = unit
        if isinstance(unit, BaseUnit):
            _id = unit.id

        if _id not in self.__ids:
            raise ValueError(f"Unit {_id} does not exist!")
        
        index = find_index(_id, self._units)
        if index == -1:
            raise ValueError(f"Unexpected error, not found value!")
        
        self._units.pop(index)
        self.__ids.remove(_id)

def convert_mutable_to_readonly(obj : MutableUnit) -> ReadOnlyUnit:
    return ReadOnlyUnit(list(iter(obj)))