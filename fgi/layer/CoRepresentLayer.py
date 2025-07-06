from .ForwardLayer import NonCodeForwardLayer
from ..units.co_represent import CoRepresentUnit


class CoRepresentLayer(NonCodeForwardLayer):
    """
    Lớp tổng hợp biểu diễn
    """
    def __init__(self, units, _id = None, *args, **kwargs):
        assert all(issubclass(type(unit), CoRepresentUnit) for unit in units)
        super().__init__(units, _id, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = x.mean(dim = 1)
        return x