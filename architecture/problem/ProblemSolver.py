from abc import ABC, abstractmethod
from dataclasses import _DataclassT
from typing import Dict, Any
from ..utils.id_management import generate_id
from ..utils.list_operator import ReadOnlyList
from ..units.base import HybridUnit
from ..units.manage import ReadOnlyUnit
from ..units.behavior import NonCodingBehavior, CodingBehavior
from ..layer.IntepretationLayer import IntepretationLayer
from ..layer.RepresentationLayer import RepresentationLayer


class ProblemBehavior(NonCodingBehavior, ABC):
    """
    Đây là cài đặt mô hình AI cho vấn đề.
    Bản chất là định hướng cách lan truyền qua các lớp.
    """
    def __init__(self, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)

    @abstractmethod
    def intepret(self, *args, **kwargs):
        """
        Diễn giải kết quả đầu ra của mô hình
        """
        pass

    @abstractmethod
    def interpretation_layer(self) -> IntepretationLayer:
        """
        Cài đặt lớp diễn giải
        """
        pass

    @abstractmethod
    def representation_layer(self) -> RepresentationLayer:
        """
        Cài đặt lớp biểu diễn đầu vào
        """
        pass

    @abstractmethod
    def units(self) -> ReadOnlyUnit:
        """
        Trả về tập các unit
        """
        pass


class ProbelmCoding(ABC, CodingBehavior):
    """
    Đây là cài đặt thuật toán thông thường cho vấn đề.

    Có thể dẫn cài đặt ràng buộc để xác định tính khả thi của đầu ra
    """
    def __init__(self, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)


class ProblemSolver(HybridUnit, ABC):
    """
    Đây là lớp dựng chính cho áp dụng cho giải quyết vấn đề
    """
    def __init__(self, coding_behavior : ProbelmCoding = None, non_coding_behavior : ProblemBehavior = None, metadata=..., *args, **kwargs):
        _id = generate_id()
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

    @abstractmethod
    def as_entity(self, *args, **kwargs) -> _DataclassT:
        """
        Trả về một thực thể của một lớp. Dùng cho tổ chức và gọi hành vi tương
        tác.

        Ví dụ problem = ProblemSolver()
        Trong ProblemSolver có class Person. Việc gọi p = problem.Person(thuộc tính) sẽ dựng nên
        một thực thể có các gì đó.

        Phương thức này sẽ tự động điền các thuộc tính vào Person + kiểm tra sự hợp lý thuộc tính
        """
        pass

    @abstractmethod
    def satisfy_rules(self, raw_property : Dict[str, Any], *args, **kwargs) -> bool:
        """
        Kiểm tra sự hợp lý của các thuộc tính thô dùng cho lớp dựng
        """
        pass

    @abstractmethod
    def recognize(self, *args, **kwargs):
        """
        Triển khai đặc điểm nhận dạng cho vấn đề (dùng cho AI)
        """
        pass

    @property
    def units(self):
        return self._non_code.units()
    
    @property
    def raw_properties(self) -> ReadOnlyList:
        """
        Tập tên gọi các tính chất thô
        """
        return self._non_code.interpretation_layer.properties()