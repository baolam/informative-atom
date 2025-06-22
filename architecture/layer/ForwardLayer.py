from .StaticLayer import StaticLayer
from torch import stack


class ForwardLayer(StaticLayer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(units, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Cách lan truyền này cần cài đặt tối ưu để tăng tốc huấn luyện
        """
        return stack([ unit(*args, **kwargs) for unit in self._units ], dim=1)
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)