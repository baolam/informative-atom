from typing import Dict, Any
from .ForwardLayer import NonCodeForwardLayer
from ..units.co_property import CoPropertyUnit
from torch import Tensor


class CoPropertyLayer(NonCodeForwardLayer):
    """
    Lớp tổng hợp tính chất cho mục đích khai thác
    """
    def __init__(self, units, _id = None, *args, **kwargs):
        assert all(issubclass(type(unit), CoPropertyUnit) for unit in units)
        super().__init__(units, _id, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        temp = dict()
        for unit in self._units:
            y = unit(x)
            temp[unit.metadata["property"]] = y
        return temp

    def intepret(self, y : Dict[str, Tensor], *args, **kwargs) -> Dict[str, Any]:
        """
        Diễn giải lại kết quả đầu ra của tổng hợp tính chất
        """
        temp = {}
        for output, unit in zip(y.values(), self._units):
            temp.update(unit.intepret(output))
        return temp
    
    @property
    def properties(self):
        """
        Trả về tập các tên thuộc tính
        """
        return [unit.metadata["property"] for unit in self._units]