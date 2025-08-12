import os

from typing import List, Tuple, Dict, Any
from abc import ABC
from torch import nn, load as torch_load, save as torch_save
from . import Problem
from .exploit import Exploiter
from ..utils.save_load import save_management_ext
from .utils import *

def save_discretize_mode(cls, problem_folder, *args, **kwargs):
    if not os.path.exists(problem_folder):
        os.makedirs(problem_folder)
    if not os.path.exists(f"{problem_folder}/units"):
        os.makedirs(f"{problem_folder}/units")
    if not os.path.exists(f"{problem_folder}/layers"):
        os.makedirs(f"{problem_folder}/layers")
        
    # Dữ liệu quản lý
    management_data = cls.metadata

    # Lưu phân rã các thành phần kiến trúc
    # Cần phải đi qua từng thành phần kiến trúc để lưu lại cho sử dụng sau
    # Lưu trữ đơn vị
    units : List[Unit] = cls._unit_keys
    for unit_name in units:
        if issubclass(type(unit_name), Unit):
            unit = unit_name
        else:
            unit : Unit = getattr(cls, unit_name)

        unit.save(f"{problem_folder}/units", save_management=False)
        management_data.update({ f"unit_{unit.id}" : unit.metadata })
    
    # Lưu trữ layer
    for layer_name in cls.metadata["layers"]:
        layer : Layer = getattr(cls, layer_name)
        layer.save(f"{problem_folder}/layers")
        management_data.update({ f"layer_{layer.id}" : layer.metadata })

    # Triển khai chỉ đơn thuần là lưu quản lí
    save_management_ext(cls.metadata, cls.id, problem_folder)


class NonCodeProblem(Problem, nn.Module, ABC):
    """
    Triển khai giải quyết vấn đề không dùng đến Code, 
    nghĩa là hành vi triển khai là dùng AI.
    """
    def __init__(self, _id, exploiter = None, *args, **kwargs):
        super().__init__(_id, exploiter, *args, **kwargs)
        self.metadata = ("call_update", False)

    def after_init(self):
        """
        Cập nhật dữ liệu sau khi hoàn thành quá trình khởi tạo
        """
        self.metadata = ("call_update", True)

        # Biến tạm dành cho phục vụ các nhiệm vụ quản lí
        self._unit_keys = get_unit_key(self)

        # Tiến hành thêm dữ liệu vào metadata
        self.metadata = ("layers", get_layer_names(self))
        self.metadata = ("units", get_unit_ids(self))
        self.metadata = ("properties", get_properties(self, self._unit_keys))

    @property
    def layer_names(self):
        assert self.metadata["call_update"], "Chưa gọi phương thức update"
        if "layers" in self.metadata:
            return self.metadata["layers"]
        return []

    @property
    def units(self):
        assert self.metadata["call_update"], "Chưa gọi phương thức update"
        if "units" in self.metadata:
            return self.metadata["units"]
        return []

    def save(self, problem_path, discretize_mode : bool = True, *args, **kwargs):
        """
        Triển khai lưu trữ theo từng thành phần.
        Cấu trúc thư mục lưu trữ, lưu quản lí, lưu thực nội dung

        discretize_mode là chỉ định cách lưu trữ theo hình thức phân rã các
        thành phần (đơn vị) 
        """
        # Kiểm tra tạo thư mục khi chưa tồn tại
        if discretize_mode:
            save_discretize_mode(self, problem_path, *args, **kwargs)
        else:
            if not os.path.isfile(problem_path):
                raise ValueError(f"{problem_path} phải là file!")
            torch_save(self.state_dict(), problem_path)            

    def as_instance(self, x, *args, **kwargs) -> Tuple[Exploiter, Dict[str, Any]]:
        """
        Cài đặt mặc định là diễn giải từ đơn vị CoProperty.
        Trả về kết quả là một tuple, cái đầu tiên là thực thể từ template khai thác, thứ hai là dữ liệu thô
        """
        # Ở đây cần phải lọc lại để lấy chính xác property, bản chất là lấy các CoPropertyUnit
        # Lấy ở cấp độ unit (trực tiếp) và gián tiếp thông qua các layer
        co_property = retrieve_coproperty_units(self, self._unit_keys)
        co_property += retrieve_coproperty(self, self.metadata["layers"])

        data = {  }

        for unit in co_property:
            data.update(**unit.intepret(x))

        # Cập nhật thêm thông số ngoài
        data.update(**kwargs)

        return self._exploiter(**data), data
    
    @classmethod
    def load(cls, problem_path, *args, **kwargs):
        """
        Lấy thông tin về kiến trúc mô hình, bộ trọng số đã giải quyết.

        Chế discretize_mode là chế độ load mô hình bằng cách sắp xếp lại units.
        problem_path : đường dẫn trực tiếp đến file lưu trữ hoặc tới thư mục
        """
        discretize_mode = os.path.isfile(problem_path)
        if discretize_mode:
            instance = cls(*args, **kwargs)
            instance.load_state_dict(torch_load(problem_path))

            return instance
        raise NotImplementedError("Chưa cài đặt cách load từ các thành phần rời rạc")


class CodeProblem(Problem, ABC):
    """
    Triển khai giải quyết vấn đề dùng thuần thuật toán
    """
    def __init__(self, _id, exploiter = None, *args, **kwargs):
        super().__init__(_id, exploiter, *args, **kwargs)
    
    def save(self, problem_folder, *args, **kwargs):
        save_management_ext(self.metadata, self.id, problem_folder)