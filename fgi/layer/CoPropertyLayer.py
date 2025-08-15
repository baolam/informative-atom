from typing import Dict, Any, Tuple
from .ForwardLayer import NonCodeForwardLayer
from ..units.co_property import CoPropertyUnit
from torch import Tensor


class CoPropertyLayer(NonCodeForwardLayer):
    """
    Lớp tổng hợp tính chất cho mục đích khai thác.

    Mô hình hoạt động của lớp này tuân theo mô hình giải quyết vấn đề và ở bước
    cuối: Hiểu vấn đề (đưa ra được biểu hiện và xác định được các biểu hiện tính chất), 
    khai thác vấn đề (tổng hợp các biểu hiện tính chất và biệt hoá cho mục đích khai thác khác nhau)

    Do đặc điểm chồng lấp trạng thái mà lớp khai thác có thêm phương phức diễn giải kết quả phục vụ
    cho các bài toán khai thác ở cấp cao. Mô hình triển khai ở đây là OOP.

    Ví dụ:
    Các tính chất của khuôn mặt con người sẽ có:
    + Mắt
    + Mũi
    + Miệng
    + ...
    ==> Cách diễn giải sẽ trả lời cho câu hỏi các tính chất đó sẽ như thế nào?
    ==> Phục vụ cho mục đích khai thác cho lập trình thuần, các tính chất được xem như key, 
    các biểu hiện xem như value.
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