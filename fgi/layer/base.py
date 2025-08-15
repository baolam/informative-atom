from abc import ABC
from typing import List
from . import Layer
from ..units import Unit


class StaticLayer(Layer, ABC):
    """
    Thực hiện nhiệm vụ nhóm và quản lí một tập các Unit.

    StaticLayer, một khi đã nhận một tập các Unit thì chức năng thêm/sửa/xoá sẽ bị vô
    hiệu hoá. Điều này phục vụ cho đảm bảo tính nhất quán cho thao tác tối ưu, tính toán
    đạo hàm, hoạt động của các đối tượng khác.
    
    Bất cứ phương pháp hoạt động nào liên quan đến một tập các đơn vị
    nên ưu tiên kế thừa lớp này.
    """
    def __init__(self, units : List[Unit], _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        # Nên tiến hành cập nhật quản lí các units ở đây, nghĩa là do units truyền vào là một Array, thao tác
        # thay đổi units có thể dẫn đến ảnh hưởng units truyền vào đây.
        # Nên có cách kiểm soát hoạt động ở lớp này.
        self._units = units

        # Lưu trữ dữ liệu nội bộ của lớp
        self.metadata = ("components", { unit.id : unit.metadata for unit in units })

    @property
    def units(self):
        return super().units

    @units.setter
    def units(self, unit : Unit):
        raise AttributeError("Việc cố gắng thay đổi units là bị cấm!")
    
    def _del_unit(self, runit):
        raise AttributeError("Việc cố gắng thay đổi units là bị cấm")

    @classmethod
    def from_units(cls, units : List[Unit], _id = None, *args, **kwargs):
        """
        Dựng nên lớp từ bản thân lớp và tập units được cho phép.
        """
        return cls(units=units, _id=_id, *args, **kwargs)