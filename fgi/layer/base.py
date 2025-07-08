from abc import ABC
from typing import List
from . import Layer
from ..units import Unit


class StaticLayer(Layer, ABC):
    """
    Lớp tĩnh. Đây là lớp cơ sở dành cho lan truyền liên đơn vị
    """
    def __init__(self, units : List[Unit], _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        # Tiến hành tạo mới nhằm tránh trường hợp thay đổi units làm thay đổi bên trong
        self._units = units

        # Tiến hành cập nhật cho quản lí
        self.metadata = ("components", { unit.id : unit.metadata for unit in units })

    @property
    def units(self):
        return super().units

    @units.setter
    def units(self, unit : Unit):
        raise AttributeError("Việc cố gắng thay đổi units là bị cấm!")
    
    def _del_unit(self, runit):
        raise AttributeError("Việc cố gắng thay đổi units là bị cấm")

    @classmethod
    def from_units(cls, units : List[Unit], _id = None, *args, **kwargs):
        return cls(units=units, _id=_id, *args, **kwargs)