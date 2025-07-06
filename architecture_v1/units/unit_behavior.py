from torch import nn, randn, matmul, cat
from .base_behavior import NonCodingBehavior

class MemoryBehavior(NonCodingBehavior):
    def __init__(self, _id=None, phi_dim : int = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        
        self._lin = nn.Linear(phi_dim, phi_dim, bias=False)
        self._act = nn.ReLU()

    def forward(self, x ,*args, **kwargs):
        """
        Hình thành trọng số biểu diễn x dựa trên phản ánh biểu hiện đã lưu trữ
        """
        z = self._lin(x) + self.hidden_behavior.max(dim=0).values
        z = self._act(z)
        return z
    
    @property
    def hidden_behavior(self):
        """
        Tượng trưng cho hành vi ẩn của đơn vị
        """
        return self._lin.weight

    def recognize(self, *args, **kwargs):
        """
        Cài đặt chiến lược nhận diện, phân biệt đơn vị bằng cách chọn các đặc trưng, mean, 
        hoặc các phương thức khác.

        Chiến lược mặc định là lấy max các chiều thành phần
        """
        return self.lin.weight.max(dim=0)
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class CombineRepresentBehavior(NonCodingBehavior):
    def __init__(self, _id=None, mem_unit = None, phi_dim : int = None, m_dim : int = None, *args, **kwargs):
        """
        
        Args:
        phi_dim (int), kích thước biểu diễn, kích thước tính toán chính
        m_dim (int), kích thước mà đơn vị tổng hợp nhận đầu vào
        components (int), số thành phần của đơn vị nhớ
        """
        super().__init__(_id, *args, **kwargs)
        # if not isinstance(components, int):
        #     raise TypeError("components must be int!")
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        if not isinstance(m_dim, int):
            raise TypeError("m_dim must be int!")
        # if not issubclass(type(mem_unit)):
        #     raise TypeError("mem_unit must be inheriented from MemoryUnit!")
        
        # Thành phần tham gia của đơn vị vấn đề
        self._mem = mem_unit

        self.score = nn.Parameter(randn(m_dim))
        self.eval_f = nn.Softmax(dim=0)

        # Thành phần đánh giá, kết hợp memory vào tính toán
        self.lin1 = nn.Linear(phi_dim, phi_dim)
        self.act1 = nn.ReLU()

        # Thành phần thao tác Fusion
        self.lin2 = nn.Linear(phi_dim * 2, phi_dim)
        self.act2 = nn.ReLU()

        # Thành phần chuẩn hoá
        self.layer_norm = nn.LayerNorm(phi_dim)

    def forward(self, x, *args, **kwargs):
        """
        Tiến hành tổng hợp lại kết quả bằng cách đánh giá đầu vào và tổng hợp
        """
        weighted_avg = self.eval_f(self.score)
        x = matmul(weighted_avg, x)
        
        y = self._mem(x)
        y = self.lin1(y)
        y = self.act1(y)

        # Kích thước của x và y là như nhau
        # Có thể tiến hành kết hợp để cho ra kết quả
        fusion = cat((x, y), dim=1)
        fusion = self.lin2(fusion)
        fusion = self.act2(fusion)

        return self.layer_norm(x + fusion)
    
    def recognize(self, *args, **kwargs):
        return super().recognize(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class PropertyBehavior(NonCodingBehavior):
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

        attn_weights = matmul(q, self.Y.T) * self.beta / (self.d ** 0.5)
        attn = self.act_attn(attn_weights)

        z = matmul(attn, self.Y)

        fusion = self.fusion(q, z)
        fusion = self.act(fusion)

        return p * fusion
    
    def recognize(self, *args, **kwargs):
        return max(self.Y, dim=0)
    
    def save(self, *args, **kwargs):
        pass
