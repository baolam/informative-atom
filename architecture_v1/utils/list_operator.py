from typing import List, Any, Callable
from collections.abc import Sequence

def find_index(key : Any, keys : List[Any], access_key : Callable = None):
    for i, _key in enumerate(keys):
        if access_key: _key = access_key(_key)
        if key == _key:
            return i
    return -1


class ReadOnlyList(Sequence):
    def __init__(self, data : List[Any], *args, **kwargs):
        super().__init__()
        self._data = data

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __contains__(self, value):
        return value in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def _as_list(self):
        return self._data