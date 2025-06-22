from .ForwardLayer import ForwardLayer


class RepresentationLayer(ForwardLayer):
    """
    Lớp input layer, đóng vai trò biểu diễn dữ liệu đầu vào
    """
    def __init__(self, units, *args, **kwargs):
        super().__init__(units, *args, **kwargs)