from torch import Tensor
from typing import Dict, Any
from .ForwardLayer import ForwardLayer
from ..units.CombineProperty import CombineProperty
from ..units.manage import ReadOnlyUnit
from ..utils.list_operator import ReadOnlyList

def check_sat_unit(units : ReadOnlyUnit):
    assert all(isinstance(unit, CombineProperty) for unit in units), "Unit must be inheriented from CombineProperty class!"


class IntepretationLayer(ForwardLayer):
    """
    Lớp diễn giải ý nghĩa, kết quả lan truyền của các đơn vị
    """
    def __init__(self, units, *args, **kwargs):
        check_sat_unit(units)
        super().__init__(units, *args, **kwargs)

    def intepret(self, x, *args, **kwargs) -> Dict[str, Any]:
        """
        Đây là gom nhóm và diễn dịch kết quả cho mục đích khai thác phía sau
        """
        collector = {}

        for unit in self._units:
            result = unit.intepret(x, *args, **kwargs)   
            collector.update(**result)         

        return collector
    
    def forward(self, x, *args, **kwargs) -> Dict[str, Tensor]:
        output = {}

        for unit in self._units:
            output[unit.metadata["as_name"]] = unit(x)

        return output
    
    def properties(self):
        """
        Trả về tập các tên gọi cho các Unit diễn dịch tương ứng
        """
        propers = []

        for unit in self._units:
            propers.append(unit.metadata["as_name"])

        return propers