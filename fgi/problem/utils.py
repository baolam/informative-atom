from ..layer import Layer
from ..layer.base import StaticLayer
from ..units import Unit
from ..units.co_property import CoPropertyUnit
from ..layer.CoPropertyLayer import CoPropertyLayer
from inspect import getmembers


def filter_layer(member_value):
    return issubclass(type(member_value), StaticLayer)

def filter_unit(member_value):
    return issubclass(type(member_value), Unit) and (not issubclass(type(member_value), Layer))

def get_layer_names(cls):
    """
    Trả về tên khoá lưu trữ
    """
    filter_members = getmembers(cls, filter_layer)
    layers = []

    for name, _ in filter_members:
        layers.append(name)

    return layers

def get_unit_key(cls):
    """
    Trả về tên các unit, khoá truy cập ứng riêng với nó
    """
    filter_members = getmembers(cls, filter_unit)
    units = []
    
    for name, unit in filter_members:
        units.append(name)
        
    # Trong layer cũng có chứa unit, cần phải lấy ra lun
    layers = get_layer_names(cls)
    
    for name in layers:
        layer : StaticLayer = getattr(cls, name)
        units += layer._units

    return units

def get_unit_ids(cls):
    """
    Trả về các ids ứng với units
    """
    filter_members = getmembers(cls, filter_unit)
    ids = []
    
    for name, unit in filter_members:
        ids.append(unit.id)
    
    # Trong layer cũng có chứa unit, cần phải lấy ra lun
    layers = get_layer_names(cls)
    
    for name in layers:
        layer : StaticLayer = getattr(cls, name)
        ids += list(layer.units)
        
    return ids

def retrieve_coproperty(cls, layer_names):
    """
    Giả sử chỉ có một lớp tổng hợp
    """
    result = []
    
    for layer_name in layer_names:
        layer : CoPropertyLayer = getattr(cls, layer_name)
        if issubclass(type(layer), CoPropertyLayer):
            result.append(layer)
    
    return result

def retrieve_coproperty_units(cls, unit_names):
    result = []
    
    for unit_name in unit_names:
        if issubclass(type(unit_name), Unit):
            unit = unit_name
        else:
            unit : CoPropertyUnit = getattr(cls, unit_name)
        
        if issubclass(type(unit), CoPropertyUnit):
            result.append(unit)
    
    return result

# def get_unit_id(cls, layers):
#     units = []

#     for layer_name in layers:
#         layer : StaticLayer = getattr(cls, layer_name)
#         units += list(layer.units)
    
#     return units

def get_co_property_layer(cls, layers):
    """
    Lấy ra tất cả các properties được chỉ định, khi đầu vào là layers
    """
    properties = []

    for layer_name in layers:
        layer : CoPropertyLayer = getattr(cls, layer_name)
        if issubclass(type(layer), CoPropertyLayer):
            properties += layer.properties
    
    return properties

def get_properties(cls, units):
    """
    Lấy ra tất cả các properties được chỉ định cho đơn vị
    tổng hợp biểu diễn.
    """
    properties = []

    for unit_name in units:
        if issubclass(type(unit_name), Unit):
            unit = unit_name
        else:
            unit : CoPropertyUnit = getattr(cls, unit_name)
            
        if issubclass(type(unit), CoPropertyUnit):
            properties.append(unit.metadata["property"])

    return properties