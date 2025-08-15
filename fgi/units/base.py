from abc import ABC
from . import Unit
from ..utils.save_load import save_management_ext, save_contentai_ext
from torch import nn


class SoftUnit(Unit, nn.Module, ABC):
    """
    Loại hình đơn vị dành cho cài đặt hành vi không dùng đến AI.
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    # def forward(self, *args, **kwargs):
    #     raise NotImplementedError("Phương thức forward phải được cài đặt!")

    def save(self, folder_path : str, save_management : bool = True, *args, **kwargs):
        """
        Lưu trữ dữ liệu gồm dữ liệu quản lí, dữ liệu nội dung.
        """
        if save_management:
            save_management_ext(self.metadata, self.id, folder_path)
        save_contentai_ext(self, self.id, folder_path)


class HardUnit(Unit, ABC):
    """
    Loại hình đơn vị dành cho những tác vụ không liên quan đến AI
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    # def forward(self, *args, **kwargs):
    #     """
    #     Cài đặt hành vi thuật toán cho đơn vị
    #     """
    #     raise NotImplementedError("Phương thức forward phải được cài đặt!")
    
    def save(self, folder_path, *args, **kwargs):
        save_management_ext(self.metadata, self.id, folder_path)


# class HybridUnit(Unit, nn.Module, ABC):
#     """
#     Loại đơn vị kết hợp hành vi cho cả hai.
#     Bản thân là Hybrid nhưng hành vi AI vẫn được ưu tiên hơn
#     """
#     def __init__(self, _id, *args, **kwargs):
#         super().__init__(_id, *args, **kwargs)
    
#     def forward(self, *args, **kwargs):
#         """
#         Định hình cách hoạt động của đơn vị, phi AI trước rùi đến AI
#         Hay AI rùi đến phi AI.
#         """
#         raise NotImplementedError("Phương thức forward phải được cài đặt!")

#     def save(self, *args, **kwargs):
#         raise NotImplementedError("Phương thức save phải được cài đặt!")

"""
    Mối quan hệ giữa NonCodeUnit và CodeUnit.
    + CodeUnit là loại đơn vị mà hành vi của nó là mã lập trình cứng (thuật toán thường).
    + NonCodeUnit là loại đơn vị mà hành vi của nó là thuật toán AI.

    Trong một quy trình hoạt động cho giải quyết vấn đề, các đơn vị Unit có thể
    tương tác với nhau theo hai hình thức chính sau:
    + thuật toán AI (NonCodeUnit) lấy kết quả từ thuật toán thường (CodeUnit).
    + thuật toán thường (CodeUnit) tận dụng kết quả từ thuật toán AI (NonCodeUnit).

    Trong một quy trình học cho thuật toán AI. CodeUnit có thể tham gia vào quá trình
    này thông qua sự tham gia của bộ chuyển đổi hành vi.

    Ví dụ cho quy trình là những đơn vị Rule (đơn vị phản ánh luật). Quá trình học là
    quá trình kiểm tra xem Output có tuân theo luật được định nghĩa (hành vi cứng) hay không.

    Sự tham gia của hai loại đơn vị trong giải quyết vấn đề.
    Hành vi của thuật toán AI có thể dùng các đơn vị phản ánh luật (bản chất là CodeUnit) để kiểm chứng tính
    khả thi.
    Hành vi của thuật toán thường có thể dùng bộ chuyển đổi hành vi để cho nó tham gia vào quy trình giải
    quyết dùng AI.

    Cách chuyển đổi hành vi không AI sang AI. 
    --> Chuyển đổi bằng cách dùng ánh xạ dạng tra cứu, cái này xảy ra khi không gian biểu hiện của hành vi không AI là
    hữu hạn.
"""