from torch import Tensor, optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from ..units.base import BaseUnit, HardUnit
from ..problem.ProblemSolver import ProblemSolver


class Learner:
    """
    Lớp dựng hình thức học cho đơn vị, chỉ dành cho nhóm đơn vị kế thừa từ SoftUnit và HybridUnit
    """
    def __init__(self, problem : ProblemSolver, *args, **kwargs):
        self._problem = problem

        # Khởi tạo một bản lookup gồm các thông số như rứa
        # optim là tập các thông số dùng cho định hướng hành vi học
        self._lookup = { unit.id : dict(learnable=False, unit=unit, optim={}) for unit in problem.units }
        self._loss_component = { property_name : lambda x: x for property_name in problem.raw_properties }
        
    @property
    def infor(self):
        return self._lookup
    
    @infor.setter
    def infor(self, infor : Tuple[BaseUnit | str, Dict[str, Any]]):
        unit, value = infor
        _id = unit
        if isinstance(unit, BaseUnit):
            _id = unit.id
        if _id not in self._lookup:
            raise ValueError(f"{_id} does not exist!")
        if isinstance(unit, HardUnit):
            raise ValueError(f"{_id} is inheriented from HardUnit which cannot be learned!")
        self._lookup[_id].update(**value)

    def parameters(self):
        """
        Trả về tập các thông số dùng cho tối ưu tiến trình AI
        """
        params = []

        for __, data in self._lookup:
            unit : BaseUnit = data["unit"]
            if data["learnable"]:
                params.append({ "param" : unit.parameters() }.update(**data["optim"]))

        return params
    
    def forward(self, *args, **kwargs):
        pass

    def train(self, epochs : int, loader : DataLoader, optimizer : optim.Optimizer, *args, **kwargs):
        pass

    def assign_loss(self, property_name, loss_fn):
        if property_name not in self._loss_component:
            raise ValueError(f"{property_name} does not exist!")
        
        self._loss_component[property_name] = loss_fn

    def combine_loss(self, *args, **kwargs) -> Tensor:
        """
        Đây là hàm cài đặt tính tổng các lỗi thành phần,
        mục tiêu là hỗ trợ lan truyền tính toán của backward
        """
        pass