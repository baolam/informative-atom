from .base import HybridUnit
from ..utils.dict_operator import add_meta, update_meta, pop_meta


class RepresentUnit(HybridUnit):
    def __init__(self, _id=None, coding_behavior = None, non_coding_behavior = None, *args, **kwargs):
        super().__init__(_id, coding_behavior, non_coding_behavior, *args, **kwargs)

    def represent(self, *args, **kwargs):
        x = self._code.forward(*args, **kwargs)
        x = self._non_code.forward(x, *args, **kwargs)
        return x

    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)

    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value

    def intepret(self, *args, **kwargs):
        pass