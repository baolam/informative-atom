from math import log
from ...units.base_behavior import NonCodingBehavior
from torch import zeros, arange, exp, nn, float as tFloat, sin, cos


class PositionalEncoding(NonCodingBehavior):
    """
    Bộ phận sinh mã vị trí dùng cho mục đích có phân biệt thứ 
    tự. Kết quả từ nghiên cứu Attention is all you need, bộ mã 
    hoá vị trí.
    """
    def __init__(self, dropout : int = None, max_len : int = None, phi_dim : int = None, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self.dropout = nn.Dropout(dropout)

        # Khởi tạo ma trận trọng số vị trí
        pe = zeros(max_len, phi_dim)
        position = arange(0, max_len, dtype=tFloat).unsqueeze(1)
        div_term = exp(arange(0, max_len, 2)).float() * (-log(10000.0) / phi_dim)

        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, q, *args, **kwargs):
        x = q + self.pe[:, :x.size()[1], :]
        x = self.dropout(x)
        return x