from .base import SoftUnit
from .memory import SoftMemoryUnit
from torch import nn, sigmoid


class PropertyUnit(SoftUnit):
    """
    Đơn vị đại diện cho tính chất
    """
    def __init__(self, phi_dim, _id = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._memory = SoftMemoryUnit(_id, phi_dim, *args, **kwargs)

        self._evaluation = nn.Linear(phi_dim, 1)
        self._projection = nn.Linear(1, phi_dim)
        self._activate = nn.ReLU()

    def evaluate(self, x, skip_activate : bool = True, *args, **kwargs):
        """
        Biểu hiện x đầu vào phản ánh bao nhiêu mức độ phù hợp với tính chất đại diện
        """
        x = self._evaluation(x)
        if not skip_activate:
            x = sigmoid(x)
        return x

    def forward(self, x, *args, **kwargs):
        # Truy xuất lại mẫu nhớ
        p = self._memory(x)
        
        # Đánh giá mức độ phù hợp của biểu hiện
        z = self.evaluate(p, skip_activate=True)
        z = self._projection(z)

        # Kết hợp thông tin đánh giá và mẫu nhớ
        x = p + z
        x = self._activate(x)

        return x