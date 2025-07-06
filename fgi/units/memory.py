from abc import ABC, abstractmethod
from . import Unit
from .base import SoftUnit
from torch import nn, randn, tensor, softmax, matmul


class MemoryUnit(Unit, ABC):
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    @abstractmethod
    def avatar(self):
        """
        Lấy mã đại diện cho đơn vị ghi nhớ
        """
        pass


class SoftMemoryUnit(MemoryUnit, SoftUnit):
    """
    Đơn vị ghi nhớ, triển khai hành vi dùng AI
    """
    def __init__(self, _id, phi_dim : int, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._beta = nn.Parameter(tensor(1.))
        self._patterns = nn.Parameter(randn(phi_dim, phi_dim) * 0.001)

    def forward(self, x ,*args, **kwargs):
        weighted = self._beta * x @ self._patterns.T
        weighted = softmax(weighted, dim=1)
        x = matmul(weighted, self._patterns)
        return x
    
    def avatar(self):
        """
        Đây xem như một vector đại diện cho đơn vị ghi nhớ.
        Dùng cho các bài toán cấp cao liên quan đến truy xuất ra đơn vị ghi nhớ
        """
        return self._patterns.mean(dim = 1)