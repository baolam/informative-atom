from abc import ABC
from . import Unit, SoftUnit
from torch import nn


class RepresentUnit(Unit, ABC):
    """
    Đơn vị biểu diễn cho dữ liệu đầu vào
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)


class SoftRepresentUnit(SoftUnit, RepresentUnit, ABC):
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)


class HardRepresentUnit(RepresentUnit, ABC):
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)