from .ForwardLayer import NonCodeForwardLayer
from ..units.represent import RepresentUnit


class RepresentLayer(NonCodeForwardLayer):
    """
    Các đơn vị biểu diễn được tổ hợp lại thành một lớp
    """
    def __init__(self, units, _id, *args, **kwargs):
        assert all(issubclass(type(unit), RepresentUnit) for unit in units)
        super().__init__(units, _id, *args, **kwargs)