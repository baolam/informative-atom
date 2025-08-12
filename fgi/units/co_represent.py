from .base import SoftUnit
from .memory import SoftMemoryUnit
from torch import nn, randn, softmax, cat, matmul


class EnhanceRepresentUnit(SoftUnit):
    """
    Tăng cường biểu diễn dựa trên biểu hiện của đơn vị nhớ
    """
    def __init__(self, phi_dim, _id = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._memory = SoftMemoryUnit(_id, phi_dim=phi_dim, *args, **kwargs)
        
        # Bộ đánh giá để kết hợp, (đánh giá dựa trên độ khác biệt)
        self._lin_diff = nn.Sequential(
            nn.Linear(phi_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, *args, **kwargs):
        p = self._memory(x)

        z = x - p
        z = self._lin_diff(z)

        t = x * z + (1 - z) * p
        return t

    def embedding(self, x, *args, **kwargs):
        """
        Lấy đại diện cho x đầu vào
        """
        p = self._memory(x)
        return p

    def proba_use_memory(self, x, *args, **kwargs):
        """
        Một biểu hiện đầu vào, một biểu hiện bổ sung, 
        hàm này trả về kết quả phản ánh mức độ dùng biểu hiện bổ sung
        """
        p = self._memory(x)
        
        z = x - p
        z = self._lin_diff(z)

        return 1 - z


class CoRepresentUnit(EnhanceRepresentUnit):
    """
    Đơn vị tổng hợp biểu diễn
    """
    def __init__(self, from_units : int, phi_dim, _id = None, *args, **kwargs):
        """
        from_units (int), số đơn vị nhận đầu vào làm tổng hợp
        """
        super().__init__(phi_dim, _id, *args, **kwargs)
        if from_units <= 1:
            raise ValueError("Chỉ tổng hợp từ hai đơn vị trở lên")

        self._weighted = nn.Parameter(randn(from_units))
        
        self._initalize_weights()

    def _initalize_weights(self):
        nn.init.uniform_(self._weighted)
    
    def forward(self, x ,*args, **kwargs):
        # Tổng hợp các đặc trưng lại với nhau
        weighted = softmax(self._weighted, dim=0)
        x = matmul(weighted, x)

        return super().forward(x)