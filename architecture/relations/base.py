from abc import ABC, abstractmethod
from typing import Any


class BaseRelation(ABC):
    def __init__(self, _from : str, _to : str ,_id : str = None, metadata : dict = {}, *args, **kwargs):
        super().__init__()
        self.__id = _id
        self.__from = _from
        self.__to = _to
        self.__metadata = metadata

    @property
    def id(self):
        return self.__id

    @property
    def _from(self):
        return self.__from
    
    @property
    def _to(self):
        return self.__to
    
    @property
    def metadata(self):
        return self.__metadata
    
    @abstractmethod
    def add_meta(self, key : str, value : Any):
        # raise NotImplementedError("add_meta method must be implemented!")
        pass
    
    @abstractmethod
    def pop_meta(self, key : str) -> Any:
        # raise NotImplementedError("pop_meta method must be implemented!")
        pass

    @abstractmethod
    def update_meta(self, key : str, value : Any, *args, **kwargs):
        pass