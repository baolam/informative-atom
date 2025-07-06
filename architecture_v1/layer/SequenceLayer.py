from collections.abc import Sequence
from typing import List
from .ForwardLayer import ForwardLayer


class SequenceLayer(Sequence):
    def __init__(self, layers :  List[ForwardLayer], *args, **kwargs):
        super().__init__()
        self._layers = layers
    
    def __len__(self):
        return len(self._layers)
    
    def __getitem__(self, index):
        return self._layers[index]
    
    def __iter__(self):
        return iter(self._layers)
    
    def forward(self, *args, **kwargs):
        """
        Cần cài đặt tối ưu lan truyền ở đây
        """
        out = self._forward_input_layer(*args, **kwargs)
        for layer in self._layers[1:]:
            out = layer(out)
        return out
    
    def _forward_input_layer(self, *args, **kwargs):
        pass