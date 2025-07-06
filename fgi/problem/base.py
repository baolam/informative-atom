import os
from abc import ABC
from torch import nn
from . import Problem
from ..layer.base import StaticLayer
from ..layer.CoPropertyLayer import CoPropertyLayer
from inspect import getmembers
from ..utils.save_load import save_management
from .. import MANAGEMENT_EXT


def filter_layer(member_value):
    return issubclass(type(member_value), StaticLayer)

def get_layer_names(cls):
    """
    Trả về tên khoá lưu trữ
    """
    filter_members = getmembers(cls, filter_layer)
    layers = []

    for name, _ in filter_members:
        layers.append(name)

    return layers


def get_unit_id(cls, layers):
    units = []
    for layer_name in layers:
        layer : StaticLayer = getattr(cls, layer_name)
        units += list(layer.units)
    return units

def get_co_property_layer(cls, layers):
    properties = []
    for layer_name in layers:
        layer : CoPropertyLayer = getattr(cls, layer_name)
        if issubclass(type(layer), CoPropertyLayer):
            properties += layer.properties
    return properties


class NonCodeProblem(Problem, nn.Module, ABC):
    """
    Triển khai giải quyết vấn đề không dùng đến Code, 
    nghĩa là hành vi triển khai là dùng AI.
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)

    def _update_additional_infor(self):
        """
        Cập nhật dữ liệu sau khi hoàn thành quá trình khởi tạo
        """
        self._layer_names = get_layer_names(self)
        self._unit_ids = get_unit_id(self, self._layer_names)

        # Tiến hành thêm dữ liệu vào metadata
        self.metadata = ("detail", { layer_name : getattr(self, layer_name).metadata for layer_name in self._layer_names })
        self.metadata = ("properties", get_co_property_layer(self, self._layer_names))

    @property
    def layer_names(self):
        return self._layer_names

    @property
    def units(self):
        return self._unit_ids

    def save(self, problem_folder, *args, **kwargs):
        """
        Triển khai lưu trữ theo từng thành phần.
        Cấu trúc thư mục lưu trữ, lưu quản lí, lưu thực nội dung
        """
        # Triển khai chỉ đơn thuần là lưu quản lí
        identifier = f"{self.id}"
        management = f"{problem_folder}/{identifier}{MANAGEMENT_EXT}"

        save_management(self.metadata, management)


class CodeProblem(Problem, ABC):
    """
    Triển khai giải quyết vấn đề dùng thuần thuật toán
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)