from .base import SoftUnit
from .memory import SoftMemoryUnit
from torch import nn, sigmoid


class PropertyUnit(SoftUnit):
    """
    Đơn vị đại diện cho tính chất
    """
    def __init__(self, phi_dim, dropout : float = 0.2, _id = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._memory = SoftMemoryUnit(_id, phi_dim, dropout)
        self._evaluation = nn.Linear(phi_dim, 1)
        self._norm = nn.LayerNorm(phi_dim)

    def forward(self, x, *args, **kwargs):
        # Truy xuất lại mẫu nhớ
        p = self._memory(x)
        
        # Đánh giá mức độ phù hợp của biểu hiện
        z = self._evaluation(x)
        z = sigmoid(z)

        # Kết hợp thông tin đánh giá và mẫu nhớ
        x = p * z

        # Chuẩn hoá
        x = self._norm(x)

        return x