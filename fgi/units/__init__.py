from typing import Tuple, Any
from abc import ABC, abstractmethod
from ..utils.dict_operator import create_update, delete_key
from ..utils.id_generator import generate_id


class Unit(ABC):
    def __init__(self, _id : str, *args, **kwargs):
        if _id is None:
            _id = generate_id()
        
        super().__init__()
        self._id = _id
        self._metadata = dict()

        # Tiến hành cập nhật tên class vào quản lý
        self.metadata = ("type", self.__class__.__name__)
    
    @property
    def id(self):
        return self._id
    
    @property
    def metadata(self):
        return self._metadata
    
    @metadata.setter
    def metadata(self, pair : Tuple[str, Any]):
        create_update(self._metadata, pair)

    @metadata.deleter
    def metadata(self, key):
        delete_key(self._metadata, key)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError("Phương thức load chưa được cài đặt!")

from .base import SoftUnit, HardUnit
from .represent import SoftRepresentUnit, HardRepresentUnit
from .co_represent import CoRepresentUnit
from .property import PropertyUnit
from .co_property import CoPropertyUnit, ChooseOptions, Regression, Boolean

__all__ = [
    "SoftUnit",
    "HardUnit",
    "CoRepresentUnit",
    "PropertyUnit",
    "CoPropertyUnit",
    "ChooseOptions",
    "Regression",
    "Boolean",
    "SoftRepresentUnit",
    "HardRepresentUnit"
]