from abc import ABC, abstractmethod
from ..units.base import BaseUnit
from ..units.manage import MutableUnit


class Layer(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._units = MutableUnit()
    
    @property
    def units(self):
        return self._units
    
    @abstractmethod
    def add_unit(self, unit : BaseUnit ,*args, **kwargs):
        # raise NotImplementedError("add_unit method must be implemented!")
        pass
    
    @abstractmethod
    def del_unit(self, unit : BaseUnit, *args, **kwargs):
        # raise NotImplementedError("del_unit method must be implemented!")
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        # raise NotImplementedError("forward method must be implemented!")
        pass