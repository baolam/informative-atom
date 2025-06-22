from abc import ABC
from .base import Layer
from .StaticLayer import StaticLayer


class DynamicLayer(ABC, Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_unit(self, unit, *args, **kwargs):
        self._units.append(unit)

    def del_unit(self, unit, *args, **kwargs):
        self._units.pop(unit)

    def as_static_layer(self, *args, **kwargs):
        return StaticLayer(self.units)