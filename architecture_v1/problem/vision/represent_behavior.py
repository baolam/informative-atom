from torch import nn, tensor, zeros, cat
from torch.nn import functional as F
from ...units.base_behavior import NonCodingBehavior, CodingBehavior

class SkipCoding(CodingBehavior):
    """
    Đây là hành vi mà bỏ qua Coding giúp tích hợp vào trong dạng đơn vị Hybrid
    """
    def __init__(self, _id=None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        return x
    
    def recognize(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class ConvolutionBehavior(NonCodingBehavior):
    """
    Áp dụng bộ lọc vào ảnh
    """
    def __init__(self, _id=None, img_shape : tuple = None, kernel = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._img_shape = img_shape
        self._kernel = tensor(kernel).expand(1, img_shape[2], img_shape[1], img_shape[0])
    
    def forward(self, x ,*args, **kwargs):
        return F.conv2d(x, self._kernel, padding=1, groups=1)
    
    @property
    def output_shape(self):
        return self._img_shape


class AllDirectionEdge(ConvolutionBehavior):
    """
    Laplacian, tách biên theo mọi hướng
    """
    def __init__(self, _id=None, img_shape = None, *args, **kwargs):
        super().__init__(_id, img_shape, [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], *args, **kwargs)


class SharpenImage(ConvolutionBehavior):
    """
    Dùng Kernel tăng cường ảnh
    """
    def __init__(self, _id=None, img_shape = None, *args, **kwargs):
        super().__init__(_id, img_shape, [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], *args, **kwargs)


class EmboosImage(ConvolutionBehavior):
    """
    Dùng Emboss Kernel để tạo hiệu ứng 3D cho ảnh
    """
    def __init__(self, _id=None, img_shape = None, *args, **kwargs):
        super().__init__(_id, img_shape, [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ], *args, **kwargs)


class EdgeEnhance(ConvolutionBehavior):
    """
    Dùng Kernel cho tăng cường biên ảnh
    """
    def __init__(self, _id=None, img_shape = None, *args, **kwargs):
        super().__init__(_id, img_shape, [
            [-1, -1, -1],
            [-1, 9, 1],
            [-1, -1, -1]
        ], *args, **kwargs)


class GrayScaleImage(ConvolutionBehavior):
    """
    Dùng Kernel để chuyển ảnh màu thành ảnh xám
    """
    def __init__(self, _id=None, img_shape = None, *args, **kwargs):
        super().__init__(_id, img_shape, [
                [[0.299]], [[0.587]], [[0.114]]
            ]
            , *args, **kwargs)
    

class DefaultBehavior(NonCodingBehavior):
    """
    Đây là hành vi biểu diễn ảnh cơ bản
    """
    def __init__(self, _id=None, img_shape : tuple = None, patch_size : int = None, phi_dim : int = None, num_heads : int = 2, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        if not isinstance(patch_size, int):
            raise TypeError("patch_size must be int!")
        
        H, W, C = img_shape
        assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"

        self.__num_patches = (H // patch_size) * (W // patch_size)
        self.proj = nn.Conv2d(
            C, 
            phi_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Bổ sung kiến trúc, lấy ý tưởng lớp đầu của ViT
        self._cls_token = nn.Parameter(zeros(1, 1, phi_dim))
        self._pos_embed = nn.Parameter(zeros(1, self.num_patches + 1, phi_dim))
        self._attn = nn.MultiheadAttention(phi_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(phi_dim)
    
    @property
    def num_patches(self):
        return self.__num_patches

    def _default_forward(self, x, *args, **kwargs):
        B = x.shape[0]

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self._cls_token.expand(B, -1, -1)
        x = cat([cls_token, x], dim=1)

        x = x + self._pos_embed

        # Tiến hành chuẩn hoá
        x = self.norm(x)

        # Tiến hành attention
        x, attn_weights = self._attn(x, x, x)
        return x, attn_weights

    def forward(self, x ,*args, **kwargs):
        x, _ = self._default_forward(x, *args, **kwargs)
        return x[:, 0]
    
    def recognize(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

class FlattenImage(NonCodingBehavior):
    def __init__(self, _id=None, img_shape : tuple = None, phi_dim : int = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        H, W, C = img_shape

        self._flatten = nn.Flatten()
        self._linear = nn.Linear(H * W * C, phi_dim)
    
    def forward(self, x, *args, **kwargs):
        x = self._flatten(x)
        x = self._linear(x)
        return x

    def recognize(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass