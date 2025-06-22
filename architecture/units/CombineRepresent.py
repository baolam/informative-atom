from torch import nn, randn, matmul, cat
from .base import SoftUnit
from .MemoryUnit import MemoryUnit
from .behavior import NonCodingBehavior
from ..utils.id_management import generate_id
from ..utils.dict_operator import add_meta, update_meta, pop_meta


class DefaultCombine(NonCodingBehavior):
    def __init__(self, _id=None, mem_unit : MemoryUnit = None, phi_dim : int = None, m_dim : int = None, *args, **kwargs):
        """
        
        Args:
        phi_dim (int), kích thước biểu diễn, kích thước tính toán chính
        m_dim (int), kích thước mà đơn vị tổng hợp nhận đầu vào
        components (int), số thành phần của đơn vị nhớ
        """
        super().__init__(_id, *args, **kwargs)
        # if not isinstance(components, int):
        #     raise TypeError("components must be int!")
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        if not isinstance(m_dim, int):
            raise TypeError("m_dim must be int!")
        if not isinstance(mem_unit, MemoryUnit):
            raise TypeError("mem_unit must be inheriented from MemoryUnit!")
        
        # Thành phần tham gia của đơn vị vấn đề
        self._mem = mem_unit

        self.score = nn.Parameter(randn(m_dim))
        self.eval_f = nn.Softmax(dim=0)

        # Thành phần đánh giá, kết hợp memory vào tính toán
        self.lin1 = nn.Linear(phi_dim, phi_dim)
        self.act1 = nn.ReLU()

        # Thành phần thao tác Fusion
        self.lin2 = nn.Linear(phi_dim * 2, phi_dim)
        self.act2 = nn.ReLU()

        # Thành phần chuẩn hoá
        self.layer_norm = nn.LayerNorm(phi_dim)

    def forward(self, x, *args, **kwargs):
        """
        Tiến hành tổng hợp lại kết quả bằng cách đánh giá đầu vào và tổng hợp
        """
        weighted_avg = self.eval_f(self.score)
        x = matmul(weighted_avg, x)
        
        y = self._mem(x)
        y = self.lin1(y)
        y = self.act1(y)

        # Kích thước của x và y là như nhau
        # Có thể tiến hành kết hợp để cho ra kết quả
        fusion = cat((x, y), dim=1)
        fusion = self.lin2(fusion)
        fusion = self.act2(fusion)

        return self.layer_norm(x + fusion)
    
    def recognize(self, *args, **kwargs):
        return super().recognize(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class CombineRepresent(SoftUnit):
    def __init__(self, metadata=..., *args, **kwargs):
        _id = generate_id()
        behavior = DefaultCombine(_id, *args, **kwargs)
        super().__init__(_id, behavior, metadata, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value