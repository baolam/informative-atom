from collections.abc import MutableSequence
from typing import List, Set
from .base import BaseRelation
from ..utils.list_operator import find_index as global_find_index, ReadOnlyList

def check_exist(relations : List[BaseRelation]):
    _ids = set()

    for relation in relations:
        if relation.id in _ids:
            raise TypeError(f"Relation {relation.id} existed!")
        _ids.add(relation.id)

    return _ids

def check_contain(relation : BaseRelation | str, relations : Set):
    _id = relation
    if isinstance(relation, BaseRelation):
        _id = relation.id
    return _id in relations

def find_index(relation_id : str, relations : List[BaseRelation]):
    def access_key(relation : BaseRelation):
        return relation.id
    return global_find_index(relation_id, relations, access_key)

class ReadOnlyRelation(ReadOnlyList):
    def __init__(self, relations : List[BaseRelation], *args , **kwargs):
        super().__init__(relations, *args, **kwargs)
        self.__ids = check_exist(relations)

    def __contains__(self, relation : BaseRelation | str):
        return check_contain(relation, self.__ids)
    
    def __iter__(self) -> BaseRelation:
        return super().__iter__()

class MutableRelation(MutableSequence):
    def __init__(self, relations : List[BaseRelation] = [], *args, **kwargs):
        super().__init__()
        self._relations = relations
        self.__ids = check_exist(relations)

    def __len__(self):
        return len(self._relations)
    
    def __iter__(self):
        return iter(self._relations)
    
    def __getitem__(self, index):
        return self._relations[index]
    
    def __contains__(self, relation : BaseRelation | str):
        return check_contain(relation, self.__ids)
    
    def __delitem__(self, index):
        pass

    def __setitem__(self, index, value):
        pass

    def insert(self, index, value):
        pass

    def append(self, relation : BaseRelation):
        if not isinstance(relation, BaseRelation):
            raise AttributeError("relation must be inheriented from BaseRelation!")
        
        if relation.id in self.__ids:
            raise ValueError(f"Relation {relation.id} existed!")
        
        self._relations.append(relation)
        self.__ids.add(relation.id)
    
    def pop(self, relation : BaseRelation | str):
        _id = relation
        if isinstance(relation, BaseRelation):
            _id = relation.id
        
        if _id not in self.__ids:
            raise ValueError(f"Relation {_id} does not exist!")

        index = find_index(_id, self._relations)
        if index == -1:
            raise ValueError(f"Unexpected error, not found value!")
        
        self._relations.pop(index)
        self.__ids.remove(_id)

def convert_mutable_to_readonly(obj : MutableRelation) -> ReadOnlyRelation:
    return ReadOnlyRelation(list(iter(obj)))