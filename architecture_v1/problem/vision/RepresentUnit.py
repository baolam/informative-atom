from ...utils.id_management import generate_id
from .represent_behavior import *
from ...units.RepresentUnit import RepresentUnit


class EdgeUnit(RepresentUnit):
    def __init__(self, _id=None, metadata=..., *args, **kwargs):
        _id = generate_id()
        coding_behavior = AllDirectionEdge(_id, *args, **kwargs)
        non_coding_behavior = DefaultBehavior(_id, img_shape=coding_behavior.output_shape, num_heads=1, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)


class AutoRepresentUnit(RepresentUnit):
    def __init__(self, _id=None, metadata=..., *args, **kwargs):
        _id = generate_id()
        non_coding_behavior = DefaultBehavior(_id, *args, **kwargs) 
        coding_behavior = SkipCoding(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

    def represent(self, x, *args, **kwargs):
        # return super().represent(*args, **kwargs)
        return self.forward(x, *args, **kwargs)
    

class DepthUnit(RepresentUnit):
    def __init__(self, _id=None, metadata=..., *args, **kwargs):
        _id = generate_id()
        coding_behavior = GrayScaleImage(_id, *args, **kwargs)
        non_coding_behavior = DefaultBehavior(_id, img_shape=coding_behavior.output_shape, num_heads=1 ,*args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)


class FlattenUnit(RepresentUnit):
    """
    Đây là đơn vị duỗi ảnh thành 1 vector duy nhất. Dùng cho trường hợp
    ảnh đầu vào nhỏ (kích thước nhỏ hơn 32x32.)

    Đơn vị này sẽ có ích trong trường hợp.
    + Trích xuất đặc trưng toàn cục.
    + Chuyển đổi định dạng
    """
    def __init__(self, _id=None, metadata=..., *args, **kwargs):
        _id = generate_id()
        coding_behavior = SkipCoding(_id, *args, **kwargs)
        non_coding_behavior = FlattenImage(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)
    
    def represent(self, x, *args, **kwargs):
        # return super().represent(*args, **kwargs)
        return self.forward(x, *args, **kwargs)