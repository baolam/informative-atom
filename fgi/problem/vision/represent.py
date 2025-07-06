"""
Tuỳ vào từng bài toán mà cách hình thành đơn
vị biểu diễn sẽ khác nhau và như vậy cũng sẽ có
nhiều loại đặc trưng khác nhau.

Ví dụ:
Đặc trưng cạnh, đặc trưng chuyển động, đặc trưng màu sắc,
...
"""
from typing import Tuple
from torch import nn, zeros, cat, tensor
from kornia.color import rgb_to_grayscale, rgb_to_hsv
from kornia.filters import canny, in_range
from ...units.represent import SoftRepresentUnit
from ...utils.get_config import get_parameter_through_function
from .utils import normalize_hsv, check_input_hsv


class ImageRepresent(SoftRepresentUnit):
    """
    Dùng cấu tạo một vài lớp đầu trong ViT cho cơ sở hoạt động
    
    Tham khảo cài đặt PatchEmbedding ở, https://medium.com/@fernandopalominocobo/demystifying-visual-transformers-with-pytorch-understanding-patch-embeddings-part-1-3-ba380f2aa37f
    """
    def __init__(self, _id, img_shape, patch_size : int, 
        num_heads : int, phi_dim : int, patch_dropout : int = 0.2, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        H, W, C = img_shape
        assert H % patch_size == 0 and W % patch_size == 0

        # Cắt bức ảnh ra thành các phần bằng nhau và tích chập
        self._patch_embedding = nn.Sequential(
            nn.Conv2d(
                C,
                phi_dim,
                kernel_size=patch_size,
                stride=patch_size
            ),
            nn.Flatten(dim=2)
        )

        num_patches = (H // patch_size) * (W // patch_size)

        # Một số thành phần cấu tạo của lớp đầu vào
        self._cls_token = nn.Parameter(zeros(1, 1, phi_dim))
        self._pos_embed = nn.Parameter(zeros(1, num_patches + 1, phi_dim))
        self._dropout = nn.Dropout(patch_dropout)

        # Hoạt động tăng cường và phối hợp
        self._attn = nn.MultiheadAttention(phi_dim, num_heads, batch_first=True)
        self._norm = nn.LayerNorm(phi_dim)
    
    def _forward(self, x, *args, **kwargs):
        cls_token = self._cls_token.expend(x.shape[0], -1, -1)

        # Hình thành biểu diễn xong cho ảnh, lần đầu
        x = self._patch_embedding(x)
        x = x.permute(0, 2, 1)
        x = cat([cls_token, x], dim=1)

        x = self._pos_embed + x
        x = self._dropout(x)

        # Tiến hành dùng Attention Mechanism để tổng
        # hợp lại kết quả
        # x (B, num_patches, phi_dim)
        x, attn_weights = self._attn(x, x, x)
        x = self._norm(x)

        return x, attn_weights
    
    def forward(self, x, *args, **kwargs):
        # Lấy kết quả từ cls làm kết quả đại diện
        x, _ = self._forward(x, *args, **kwargs)
        return x[:, 0]
    
    def noncode_forward(self, x, *args, **kwargs):
        raise NotImplementedError


class DepthRepresent(ImageRepresent):
    """
    Đơn vị biểu diễn độ sâu ảnh, bản chất là thao tác trên ảnh xám
    """
    def __init__(self, _id, img_shape, patch_size, num_heads, phi_dim, patch_dropout = 0.2, *args, **kwargs):
        super().__init__(_id, img_shape, patch_size, num_heads, phi_dim, patch_dropout, *args, **kwargs)
    
    def forward(self, x, *args, **kwargs):
        x = rgb_to_grayscale(x)
        x = super().forward(x, *args, **kwargs)


class EdgeRepresent(ImageRepresent):
    """
    Đơn vị biểu diễn cạnh ảnh,
    Áp dụng thuật toán Canny
    """
    def __init__(self, _id, img_shape, patch_size, num_heads, 
            phi_dim, patch_dropout = 0.2, *args, **kwargs):
        super().__init__(_id, img_shape, patch_size, num_heads, phi_dim, patch_dropout, *args, **kwargs)

        self.metadata = ("canny_config", get_parameter_through_function(canny))
    
    def noncode_forward(self, x, *args, **kwargs):
        x = rgb_to_grayscale(x)
        x, edges = canny(x, **self.metadata["canny_config"])
        return x, edges

    def forward(self, x, *args, **kwargs):
        x, _ = self.noncode_forward(x, *args, **kwargs)
        x = super().forward(x, *args, **kwargs)
        return x
    

class ColorFilter(ImageRepresent):
    """
    Đơn vị biểu diễn phản xạ theo một nhóm màu nhất định.
    Cài đặt bộ lọc màu.
    """
    def __init__(self, _id, img_shape, patch_size, num_heads, 
        phi_dim, lower_hsv : Tuple[float, float, float], upper_hsv : Tuple[float, float, float], patch_dropout = 0.2, *args, **kwargs):
        super().__init__(_id, img_shape, patch_size, num_heads, phi_dim, patch_dropout, *args, **kwargs)

        check_input_hsv(lower_hsv)
        check_input_hsv(upper_hsv)

        # Cho phép hiển thị cài đặt trong metadata, việc thay đổi sẽ ko gây ảnh 
        # hưởng đến hành vi cố định của đơn vị
        self.metadata = ("filter_config", {
            "lower" : lower_hsv,
            "upper" : upper_hsv
        })

        # Đăng ký tham số nhận
        self.register_buffer("_lower", tensor(normalize_hsv(lower_hsv)).expand(1, 1, -1))
        self.register_buffer("_upper", tensor(normalize_hsv(upper_hsv)).expand(1, 1, -1))

    def noncode_forward(self, x, *args, **kwargs):
        x = rgb_to_hsv(x)
        x = in_range(x, self._lower, self._upper)
        return x

    def forward(self, x, *args, **kwargs):
        x = self.noncode_forward(x, *args, **kwargs)
        x = super().forward(x, *args, **kwargs)
        return x