from abc import ABC, abstractmethod
from ..units import Unit


class Problem(Unit, ABC):
    """
    Bộ dựng của một Problem(Vấn đề)
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    @abstractmethod
    def recognize_unknown(self, *args, **kwargs):
        """
        Phát hiện chưa biết. Đây là phương thức nhận đầu vào
        là biểu hiện, xác định xem biểu hiện đó có thể xử lí 
        được bằng vấn đề phụ trách hay không
        """
        pass

    @property
    @abstractmethod
    def _as_object(self, *args, **kwargs):
        """
        Trả về một bản dựng của vấn đề (hình thức cài đặt hướng
        đối tượng).
        Dấu _ ám thị cho việc đây là phương thức cài đặt ở các lớp kế thừa
        """
        pass

    @abstractmethod
    def as_instance(self, *args, **kwargs):
        """
        Trả về một thực thể (một biểu hiện) của vấn đề (là thực thể thực sự của vấn đề).
        Có thể dùng cho các bài toán quản lý, tương tác các thực thể để tìm ra đánh giá phù hợp
        nhất, ...

        Ngoài ra, tương tác với môi trường, ta cũng có thể yêu cầu tự chọn hành động tương tác.        
        """
        pass
    
    @property
    @abstractmethod
    def units(self):
        """
        Do vấn đề được định nghĩa là phối hợp tham gia của các đơn vị nên
        cần trả về tập các đơn vị tham gia cho một số bài toán
        """
        pass

from .base import NonCodeProblem, CodeProblem
from .vision import *

__all__ = [
    "NonCodeProblem",
    "CodeProblem",
    "ImageRepresent",
    "DepthRepresent",
    "ColorFilter",
    "EdgeRepresent"
]