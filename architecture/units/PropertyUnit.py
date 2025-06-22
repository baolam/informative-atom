from torch import nn, randn, matmul
from .base import SoftUnit
from .behavior import NonCodingBehavior
from ..utils.id_management import generate_id
from ..utils.dict_operator import add_meta, update_meta, pop_meta


class DefaultBehavior(NonCodingBehavior):
    def __init__(self, _id=None, components : int = None, phi_dim : int = None, beta_scaling : int = 1.0, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        if not isinstance(components, int):
            raise TypeError(f"components must be int!")
        if not isinstance(phi_dim, int):
            raise TypeError(f"phi_dim must be int!")

        self.beta = beta_scaling
        self.d = phi_dim

        # Xem đây như là tập truy vấn cơ sở (Y)
        self.Y = nn.Parameter(randn(components, phi_dim))
        self.act_attn = nn.Softmax(dim=1)

        # Trọng số học kiểm tra Exist
        self.exist = nn.Linear(phi_dim, 1)
        self.act_exist = nn.Sigmoid()

        # Các trọng số dùng cho học kết hợp đặc trưng
        self.fusion = nn.Bilinear(phi_dim, phi_dim, phi_dim)
        self.act = nn.ReLU()

    def forward(self, q ,*args, **kwargs):
        """
        Lan truyền tính toán
        """
        # Kích thước đầu vào của x (Batch_size, phi_dim)
        p = self.exist(q)
        p = self.act_exist(p)

        attn_weights = matmul(q, self.Y) * self.beta / (self.d ** 0.5)
        attn = self.act_attn(attn_weights)

        z = matmul(attn, self.Y)

        fusion = self.fusion(q, z)
        fusion = self.act(fusion)

        return p * fusion
    

class PropertyUnit(SoftUnit):
    def __init__(self, metadata=..., *args, **kwargs):
        _id = generate_id()
        behavior = DefaultBehavior(_id, *args, **kwargs)
        super().__init__(_id, behavior, metadata, *args, **kwargs)

    @property
    def raw_memory(self):
        """
        Truy cập bộ nhớ riêng được lưu giúp cho biệt hoá tính chất
        """
        return self._behavior.Y
    
    # def add_meta(self, key, value, *args, **kwargs):
    #     pass

    # def pop_meta(self, key, *args, **kwargs):
    #     pass

    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value