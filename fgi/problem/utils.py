from ..layer.base import StaticLayer
from ..layer.CoPropertyLayer import CoPropertyLayer
from inspect import getmembers


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
