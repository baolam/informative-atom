from .ForwardLayer import ForwardLayer
from ..units.RepresentUnit import RepresentUnit


class RepresentationLayer(ForwardLayer):
    """
    Lớp input layer, đóng vai trò biểu diễn dữ liệu đầu vào
    """
    def __init__(self, units, *args, **kwargs):
        if len(units) == 0:
            raise TypeError("units mustn't be empty!")
        if any(not isinstance(unit, RepresentUnit) for unit in units):
            raise TypeError("units's element must be inheriented from RepresentUnit!")
        super().__init__(units, *args, **kwargs)