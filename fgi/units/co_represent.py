from .base import SoftUnit
from .memory import SoftMemoryUnit
from torch import nn, randn, softmax, cat, matmul


class CoRepresentUnit(SoftUnit):
    """
    Đơn vị tổng hợp biểu diễn
    """
    def __init__(self, _id, from_units : int, phi_dim, *args, **kwargs):
        """
        from_units (int), số đơn vị nhận đầu vào làm tổng hợp
        """
        super().__init__(_id, *args, **kwargs)
        if from_units <= 1:
            raise ValueError("Chỉ tổng hợp từ hai đơn vị trở lên")

        self._memory = SoftMemoryUnit(_id, phi_dim)
        self._weighted = nn.Parameter(randn(from_units))
        self._combine = nn.Linear(2 * phi_dim, phi_dim)
        self._activate = nn.SiLU()
    
    def forward(self, x ,*args, **kwargs):
        # Tổng hợp các đặc trưng lại với nhau
        weighted = softmax(self._weighted, dim=0)
        x = matmul(weighted, x)
    
        # Tiến hành truy xuất mẫu gợi nhớ
        p = self._memory(x)

        # Tiến hành nối lại đặc trưng
        fusion = cat((x, p), dim=1)

        # Tiến hành sử dụng
        fusion = self._combine(fusion)
        fusion = self._activate(fusion)

        return fusion