from abc import ABC
from typing import List
from ..units import Unit
from ..utils.save_load import save_management_ext

class Layer(Unit, ABC):
    """
    Layer, (khái niệm triển khai phối hợp nhóm các đơn vị cùng hoạt động với nhau).

    Đây là lớp hành vi cơ sở của một Layer. Chứa các nhóm phương thức liên quan đến
    quản lí units như thêm, sửa, xoá và hành vi lưu trữ, tải nội dung.
    """
    def __init__(self, _id, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        self._units : List[Unit] = [] 

    @property
    def units(self) -> tuple[str]:
        """
        Để bảo vệ các đơn vị tham gia trong Layer.
        Chỉ trả về danh sách các id dưới dạng tuple
        """
        return tuple(unit.id for unit in self._units)
    
    @units.setter
    def units(self, unit : Unit):
        """
        Thêm vào một đơn vị mới.
        Phải kiểm tra đơn vị đó chưa tồn tại
        """
        assert issubclass(type(unit), Unit), "Không phải là đơn vị"
        assert unit.id not in self.units, "Đơn vị đã tồn tại"
        self._units.append(unit)

    def _del_unit(self, runit : Unit | str):
        """
        Xoá một đơn vị
        """
        unit = runit
        if isinstance(runit, str):
            # Tiến hành truy vấn ngược ra unit
            units = tuple(filter(lambda u : u.id == runit, self._units))
            assert len(unit) == 1
            unit = units[0]
        self._units.remove(unit)

    def __add__(self, unit):
        self.units = unit
    
    def __sub__(self, unit):
        self._del_unit(unit)

    def save(self, layer_folder ,*args, **kwargs):
        save_management_ext(self.metadata, self.id, layer_folder)


from .base import StaticLayer
from .CoPropertyLayer import CoPropertyLayer
from .PropertyLayer import PropertyLayer
from .RepresentLayer import RepresentLayer
from .CoRepresentLayer import CoRepresentLayer

__all__ = [
    "StaticLayer",
    "CoPropertyLayer",
    "PropertyLayer",
    "RepresentLayer",
    "CoRepresentLayer"
]