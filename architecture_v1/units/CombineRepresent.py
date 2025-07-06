from .base import SoftUnit
from ..utils.id_management import generate_id
from ..utils.dict_operator import add_meta, update_meta, pop_meta
from .unit_behavior import CombineRepresentBehavior


class CombineRepresent(SoftUnit):
    def __init__(self, *args, **kwargs):
        _id = generate_id()
        behavior = CombineRepresentBehavior(_id, *args, **kwargs)
        super().__init__(_id, behavior, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value