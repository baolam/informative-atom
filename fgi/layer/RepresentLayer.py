from .ForwardLayer import NonCodeForwardLayer
from ..units.represent import RepresentUnit


class RepresentLayer(NonCodeForwardLayer):
    """
    Các đơn vị biểu diễn được tổ hợp lại thành một lớp
    """
    def __init__(self, units, output_dim : int, _id = None, *args, **kwargs):
        assert all(issubclass(type(unit), RepresentUnit) for unit in units)
        super().__init__(units, _id, *args, **kwargs)
        self.metadata = ("phi_dim", output_dim)
    
    def forward(self, x, *args, **kwargs):
        return super().forward(x, phi_dim = self.metadata["phi_dim"], *args, **kwargs)