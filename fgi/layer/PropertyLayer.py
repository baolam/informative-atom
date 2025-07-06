from .ForwardLayer import NonCodeForwardLayer
from ..units.property import PropertyUnit


class PropertyLayer(NonCodeForwardLayer):
    """
    Lớp tính chất
    """
    def __init__(self, units, _id, *args, **kwargs):
        assert all(issubclass(type(unit), PropertyUnit) for unit in units)
        super().__init__(units, _id, *args, **kwargs)