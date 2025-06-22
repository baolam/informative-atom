from torch import nn, randn, max
from .base import SoftUnit
from .behavior import NonCodingBehavior
from ..utils.id_management import generate_id
from ..utils.dict_operator import add_meta, update_meta, pop_meta


class DefaultBehavior(NonCodingBehavior):
    def __init__(self, _id=None, phi_dim : int = None, components : int = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        if not isinstance(components, int):
            raise TypeError("components must be int!")
        self.W = nn.Parameter(randn(components, phi_dim))
        self._act = nn.ELU()
        self.norm = nn.LayerNorm(phi_dim)

    def forward(self, x ,*args, **kwargs):
        """
        Hình thành trọng số biểu diễn x dựa trên phản ánh biểu hiện đã lưu trữ
        """
        z = x * self._act(x) + self.recognize().values
        z = self.norm(z)
        return z

    def recognize(self, *args, **kwargs):
        """
        Cài đặt chiến lược nhận diện, phân biệt đơn vị bằng cách chọn các đặc trưng, mean, 
        hoặc các phương thức khác.

        Chiến lược mặc định là lấy max các chiều thành phần
        """
        return max(self.W, dim=0)
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class MemoryUnit(SoftUnit):
    def __init__(self, metadata=...,*args, **kwargs):
        _id = generate_id()
        behavior = DefaultBehavior(_id,*args, **kwargs)
        super().__init__(_id, behavior, metadata, *args, **kwargs)
    
    @property
    def representation(self):
        """
        Xem như biểu diễn, bản chất của đơn vị ghi nhớ phụ trách
        """
        return self._behavior.recognize().values

    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value