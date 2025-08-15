"""
    Một số hành vi khai thác đơn vị Unit.

    Các lớp khai thác ở dưới ở dưới dựa trên ý tưởng về kiểu dữ liệu nguyên thuỷ.
    + Số (Float, cài đặt là Regression)
    + Lựa chọn (Mảng, Options)
    + Quyết định (Đúng sai, cài đặt là Boolean)
"""

from abc import ABC, abstractmethod
from .base import SoftUnit
from typing import List, Dict, Any
from torch import nn, randn, softmax, sigmoid, Tensor, matmul, empty


class CoPropertyUnit(SoftUnit, ABC):
    """
    Kết hợp các tính chất cho mục đích khai thác
    """
    def __init__(self, from_units : int, phi_dim, _id = None, *args, **kwargs):
        """
        from_units (phi_dim), nhận đầu vào từ bao nhiêu đơn vị
        """
        super().__init__(_id, *args, **kwargs)
        if from_units > 1:
            # Thứ tự kết hợp các tính chất có ảnh hưởng đến kết quả
            # khai thác
            self._position = nn.Parameter(empty(from_units, phi_dim))

            # Chỉ số đánh giá
            self._weighted = nn.Parameter(randn(from_units))

            # Tinh chỉnh lại kết quả tổng hợp cho phù hợp
            # với bài toán
            self._enhance = nn.Sequential(
                nn.Linear(phi_dim, phi_dim),
                nn.ReLU()
            )

            self._initalize_weight()
        
        self.metadata = ("one_unit?", from_units == 1)

    def _initalize_weight(self):
        nn.init.kaiming_normal_(self._position)
        nn.init.uniform_(self._weighted)

    def forward(self, x, *args, **kwargs):
        if not self.metadata["one_unit?"]:
            # Giai đoạn thêm mã vị trí vào tính chất
            x = x + self._position
            x = nn.functional.leaky_relu(x)

            # Giai đoạn tổng hợp kết quả
            weighted = softmax(self._weighted, dim = 0)
            x = matmul(weighted, x)

            # Giai đoạn tăng cường biểu diễn
            x = self._enhance(x)

        return x
    
    @abstractmethod
    def intepret(self, y : Tensor ,*args, **kwargs) -> Dict[str, Any]:
        """
        Dịch nghĩa kết quả hoạt động tổng hợp tính chất (AI) sang phi-AI.
        Dưới dạng các biểu tượng mà người lập trình thao tác được
        """
        pass


class ChooseOptions(CoPropertyUnit):
    """
    Dùng kết quả tổng hợp cho mục đích lựa chọn trong nhiều lựa chọn
    """
    def __init__(self, from_units, options : List[str], property_name : str, phi_dim, _id = None, *args, **kwargs):
        super().__init__(from_units, phi_dim = phi_dim, _id = _id, *args, **kwargs)
        self._decides = nn.Linear(phi_dim, len(options))

        self.metadata = ("property", property_name)
        self.metadata = ("options", options.copy())
    
    def forward(self, x, skip_activate : bool = True, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = self._decides(x)
        if not skip_activate:
            x = softmax(x, dim = 1)
        return x
    
    def intepret(self, y : Tensor ,*args, **kwargs):
        """
        y (Tensor) là kết quả của đầu ra
        """
        assert y.size()[0] == 1, "Cho một biểu hiện duy nhất"
        z = y.max(dim=1)
        prob = z.values.item()
        inde = z.indices.item()
        return { self.metadata["property"] : self.metadata["options"][inde], f"{self.metadata["property"]}.raw" : prob }


class Regression(CoPropertyUnit):
    """
    Dùng kết quả tổng hợp cho ước đoán ra một số
    """
    def __init__(self, from_units, phi_dim, property_name : str, _id = None, *args, **kwargs):
        super().__init__(from_units, phi_dim = phi_dim, _id = _id, *args, **kwargs)
        self._predicted = nn.Linear(phi_dim, 1)

        self.metadata = ("property", property_name)
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = self._predicted(x)
        return x
    
    def intepret(self, y, *args, **kwargs):
        assert y.size()[0] == 1, "Cho một biểu hiện duy nhất"
        return { self.metadata["property"] : y.item() }


class Boolean(Regression):
    """
    Dùng tổng hợp tính chất cho đưa ra quyết định đúng/sai
    """
    def __init__(self, from_units, phi_dim, property_name, threshold : float, _id = None, *args, **kwargs):
        super().__init__(from_units, phi_dim = phi_dim, property_name = property_name, _id = _id, *args, **kwargs)
        self.metadata = ("threshold", threshold)
    
    def forward(self, x, skip_activate : bool = True, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        if not skip_activate:
            x = sigmoid(x)
        return x
    
    def intepret(self, y, *args, **kwargs):
        assert y.size()[0] == 1, "Cho một biểu hiện duy nhất"
        prob = y.item()
        return { self.metadata["property"] : prob >= self.metadata["threshold"], f"{self.metadata["property"]}.raw" : prob }