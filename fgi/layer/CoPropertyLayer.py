from typing import Dict, Any, Tuple
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
        output = []
        for unit in self._units:
            y = unit(x)
            output.append(y)
        return tuple(output)

    def intepret(self, y : Tuple[Tensor], *args, **kwargs) -> Dict[str, Any]:
        """
        Diễn giải lại kết quả đầu ra của tổng hợp tính chất
        """
        temp = {}
        for output, unit in zip(y, self._units):
            temp.update(unit.intepret(output))
        return temp
    
    @property
    def properties(self):
        """
        Trả về tập các tên thuộc tính
        """
        return [unit.metadata["property"] for unit in self._units]