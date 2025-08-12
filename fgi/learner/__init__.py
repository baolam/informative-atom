from typing import Dict, Tuple, Callable
from abc import ABC, abstractmethod
from ..layer.ForwardLayer import NonCodeForwardLayer
from ..units.base import SoftUnit
from ..problem.base import NonCodeProblem
from torch.optim import Optimizer
from torch import Tensor
from ..utils.after_init import AutoAfterInitMeta


class Learner(ABC, metaclass=AutoAfterInitMeta):
    """
    Triển khai bộ học cho vấn đề. Cấp độ thao tác là vấn đề
    """
    def __init__(self, problem : NonCodeProblem ,*args, **kwargs):
        assert issubclass(type(problem), NonCodeProblem)
        super().__init__()
        self._problem = problem

        # Khoá bảo vệ, ngăn thay đổi
        self.__block_mode = False

        # Tiến hành chuẩn bị cho việc học
        # Bảng tra cứu đơn vị đơn vị nào nên học hay không
        self._learnable : Dict[str, bool] = { _id : False for _id in problem.units }

        # Bảng điền hàm loss
        # Tối ưu chỉ dùng duy nhất 1 loại xuyên suốt quá trình
        self._loss_infor = { _property : lambda x : x for _property in problem.metadata["properties"] } 

        # Bảng cập nhật metric
        self._metric_infor = { _property : lambda x : x for _property in problem.metadata["properties"] }

    def after_init(self, *args, **kwargs):
        """
        Cấu hình mặc định là tất cả đơn vị tham gia đều học
        """
        for _id in self.learnable.keys():
            self.learnable = (_id, True)

    @property
    def learnable(self):
        return self._learnable
    
    @learnable.setter
    def learnable(self, infor : Tuple[SoftUnit | str, bool]):
        unit, status = infor
        if issubclass(type(unit), SoftUnit):
            unit = unit.id
        assert unit in self._learnable, f"Đơn vị {unit} không tham gia vào quá trình học!"
        assert not self.__block_mode, "Chức năng bị khoá"
        self._learnable[unit] = status

    @property
    def loss_infor(self):
        return self._loss_infor
    
    @loss_infor.setter
    def loss_infor(self, infor : Tuple[str, Callable]):
        _property, loss_fn = infor
        assert _property in self._loss_infor
        assert not self.__block_mode, "Chức năng bị khoá"
        self._loss_infor[_property] = loss_fn

    def compile(self, *args, **kwargs):
        """
        biên dịch, khoá chức năng là cách thức tổng hợp lỗi các thành phần
        """
        # Sau khi tiến hành cập nhật thì tiến hành khoá thay đổi, 
        # định danh các đơn vị, bộ số học
        self.__block_mode = True
        self._update_learnable_state()

        # Điền các hình thức cần thiết cho tối ưu
        # self._optimizer : Optimizer = optimizer_cls(self._problem.parameters(), *args, **kwargs)
        # self._device = device

    @abstractmethod
    def _aggerate_loss(self, y_hat, y, *args, **kwargs) -> Dict[str, Tensor]:
        """
        Cách thức tiến hành tổng hợp lỗi thành phần
        """
        pass

    def _update_learnable_state(self):
        # Đảm bảo có ít nhất một đơn vị cần học
        assert any(code for code in self._learnable.values()), "Phải có ít nhất 1 đơn vị cần đào tạo!"

        # Tiến hành truy cập qua các Unit trong các Layer, đối chiếu False mà cho
        # ngăn và chặn lan truyền
        for layer_name in self._problem.layer_names:
            # Đối với mỗi layer name, lấy ra các Unit
            layer : NonCodeForwardLayer = getattr(self._problem, layer_name)
            # Tiến hành lặp qua các Unit
            for unit in layer._units:
                # Cập nhật required_grad toàn bộ đơn vị
                for p in unit.parameters():
                    p.requires_grad = self._learnable[unit.id]

    @staticmethod
    def total_parameters(problem : NonCodeProblem):
        return sum(p.numel() for p in problem.parameters())
    
    @staticmethod
    def total_learnable_parameters(problem : NonCodeProblem):
        return sum(p.numel() for p in problem.parameters() if p.requires_grad)
    

from .LightningLearner import LightningLearner

__all__ = [
    "Learner",
    "LightningLearner"
]