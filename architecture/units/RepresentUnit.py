from .base import HybridUnit

class RepresentUnit(HybridUnit):
    def __init__(self, _id=None, coding_behavior = None, non_coding_behavior = None, metadata=..., *args, **kwargs):
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

    def represent(self, *args, **kwargs):
        x = self._code.forward(*args, **kwargs)
        x = self._non_code.forward(x, *args, **kwargs)
        return x
    
    def add_meta(self, key, value, *args, **kwargs):
        if key in self.metadata:
            raise TypeError(f"{key} existed!")
        self._metadata[key] = value
    
    def update_meta(self, key, value, *args, **kwargs):
        if key not in self._metadata:
            raise TypeError(f"{key} does not exist!")
        self._metadata.update({ key : value })
    
    def pop_meta(self, key, *args, **kwargs):
        if key not in self._metadata:
            raise TypeError(f"{key} does not exist!")
        return self._metadata.pop(key)
    
    def intepret(self, *args, **kwargs):
        pass