import os
from abc import ABC
from torch import nn
from . import Problem
from ..utils.save_load import save_management_ext
from .utils import *


class NonCodeProblem(Problem, nn.Module, ABC):
    """
    Triển khai giải quyết vấn đề không dùng đến Code, 
    nghĩa là hành vi triển khai là dùng AI.
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self.metadata = ("call_update", False)

    def _update_additional_infor(self):
        """
        Cập nhật dữ liệu sau khi hoàn thành quá trình khởi tạo
        """
        self._layer_names = get_layer_names(self)
        self._unit_ids = get_unit_id(self, self._layer_names)

        # Tiến hành thêm dữ liệu vào metadata
        self.metadata = ("detail", { layer_name : getattr(self, layer_name).metadata for layer_name in self._layer_names })
        self.metadata = ("properties", get_co_property_layer(self, self._layer_names))
        self.metadata = ("call_update", True)

    @property
    def layer_names(self):
        assert self.metadata["call_update"], "Chưa gọi phương thức _update_additional_infor"
        return self._layer_names

    @property
    def units(self):
        assert self.metadata["call_update"], "Chưa gọi phương thức _update_additional_infor"
        return self._unit_ids

    def save(self, problem_folder, *args, **kwargs):
        """
        Triển khai lưu trữ theo từng thành phần.
        Cấu trúc thư mục lưu trữ, lưu quản lí, lưu thực nội dung
        """
        # Triển khai chỉ đơn thuần là lưu quản lí
        save_management_ext(self.metadata, self.id, problem_folder)


class CodeProblem(Problem, ABC):
    """
    Triển khai giải quyết vấn đề dùng thuần thuật toán
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    
    def save(self, problem_folder, *args, **kwargs):
        save_management_ext(self.metadata, self.id, problem_folder)