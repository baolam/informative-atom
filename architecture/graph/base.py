from abc import ABC, abstractmethod
from typing import List
from ..units.base import BaseUnit
from ..units.manage import MutableUnit
from ..relations.base import BaseRelation
from ..relations.manage import MutableRelation
from ..relations.adjacency_list import AdjacencyList


class BaseGraph(ABC):
    def __init__(self, _id : str = None ,*args, **kwargs):
        super().__init__()
        self.__id = _id
        self._units = MutableUnit([])
        self._relations = MutableRelation([])

    @property
    def id(self):
        return self.__id

    @property
    def units(self):
        return self._units
    
    @property
    def relations(self):
        return self._relations
    
    def adjacency_list(self):
        """
        Chuyển đổi Relation dưới dạng lưu trữ các danh sách kề
        """
        return AdjacencyList(self.relations)
    
    @abstractmethod
    def add_unit(self, unit : BaseUnit):
        # raise NotImplementedError("add_unit method must be implemented!")
        pass
    
    @abstractmethod
    def del_unit(self, unit : BaseUnit):
        # raise NotImplementedError("del_unit method must be implemented!")
        pass
    
    @abstractmethod
    def add_relation(self, relation : BaseRelation):
        # raise NotImplementedError("add_relation method must be implemented!")
        pass
    
    @abstractmethod
    def del_relation(self, relation : BaseRelation):
        # raise NotImplementedError("del_relation method must be implemented!")
        pass