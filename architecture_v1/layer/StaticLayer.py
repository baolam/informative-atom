from abc import ABC
from .base import Layer
from ..units.manage import ReadOnlyUnit, MutableUnit, convert_mutable_to_readonly

def convert_mutable(obj : MutableUnit | ReadOnlyUnit):
    if isinstance(obj, MutableUnit):
        return convert_mutable_to_readonly(obj)
    return obj


class StaticLayer(Layer, ABC):
    def __init__(self, units : MutableUnit | ReadOnlyUnit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._units = convert_mutable(units)

    def add_unit(self, unit, *args, **kwargs):
        raise ValueError("Add_unit method is not allowed!")

    def del_unit(self, unit, *args, **kwargs):
        raise ValueError("Add_unit method is not allowed!")
    
    def save(self, *args, **kwargs):
        raise NotImplementedError("save method must be implemented!")