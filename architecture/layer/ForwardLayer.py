from .StaticLayer import StaticLayer
from torch import stack, nn


class ForwardLayer(StaticLayer, nn.Module):
    """
    Lớp lan truyền, bản chất cũng là Module (kế thừa hành vi trong thư viện torch)
    """
    def __init__(self, units, *args, **kwargs):
        super().__init__(units, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Cách lan truyền này cần cài đặt tối ưu để tăng tốc huấn luyện
        """
        return stack([ unit(*args, **kwargs) for unit in self._units ], dim=1)
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    
    def as_cluster(self):
        """
        Đây là phương thức trả về một Kn graph (đồ thị đầy đủ) của các
        unit tham gia
        """
        pass