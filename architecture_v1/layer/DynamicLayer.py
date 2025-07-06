from typing import List
from abc import ABC, abstractmethod
from .base import Layer
from ..units.base import BaseUnit
from .StaticLayer import StaticLayer


class DynamicLayer(Layer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_unit(self, unit, *args, **kwargs):
        self._units.append(unit)

    def del_unit(self, unit, *args, **kwargs):
        self._units.pop(unit)

    def as_static(self, *args, **kwargs):
        """
        Trả về bản dựng tĩnh của lớp
        """
        return StaticLayer(self.units)
    
    def __add__(self, units : List[BaseUnit]):
        for unit in units:
            self.add_unit(unit)
    
    def forward(self, *args, **kwargs):
        raise ValueError("DynamicLayer can not forward!")