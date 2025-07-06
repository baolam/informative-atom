from .base import SoftUnit
from ..utils.id_management import generate_id
from ..utils.dict_operator import add_meta, update_meta, pop_meta
from .unit_behavior import MemoryBehavior


class MemoryUnit(SoftUnit):
    def __init__(self, *args, **kwargs):
        _id = generate_id()
        behavior = MemoryBehavior(_id,*args, **kwargs)
        super().__init__(_id, behavior, *args, **kwargs)
    
    @property
    def representation(self):
        """
        Xem như biểu diễn, bản chất của đơn vị ghi nhớ phụ trách
        """
        return self._behavior.recognize().values

    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value