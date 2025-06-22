from collections import defaultdict
from typing import Dict, List
from .base import BaseRelation
from .manage import ReadOnlyRelation
from ..utils.list_operator import ReadOnlyList


class AdjacencyList():
    def __init__(self, relations : List[BaseRelation], *args, **kwargs):
        super().__init__()

        self.__manage_ids : Dict[str, List[str]] = defaultdict(list)
        self.__lookup : Dict[str, BaseRelation] = dict()
        self.__build_dict(relations)

    def __build_dict(self, relations : List[BaseRelation]):
        for relation in relations:
            self.__manage_ids[relation._from].append(relation._to)
            self.__lookup[relation.id] = relation

    def __contains__(self, relation : BaseRelation):
        if relation._from not in self.__manage_ids:
            return False
        return relation._to in self.__manage_ids[relation._from]
    
    def out_nodes(self, node : str):
        """
        Trả về các node là đầu ra của node được cung cấp
        """
        if node not in self.__manage_ids:
            return ReadOnlyList([])

        return ReadOnlyList(self.__manage_ids[node])
    
    def in_relations(self, node_str : str) -> ReadOnlyRelation:
        """
        Trả về các cạnh vào của một node được cung cấp
        """
        relations = []

        for relation in self.__lookup.values():
            if relation._to == node_str:
                relations.append(relation)
        
        return ReadOnlyRelation(relations)
    
    def out_relations(self, node_str : str) -> ReadOnlyRelation:
        """
        Trả về các cạnh ra của một node được cung cấp
        """
        relations = []

        for relation in self.__lookup.values():
            if relation._from == node_str:
                relations.append(relation)

        return ReadOnlyRelation(relations)