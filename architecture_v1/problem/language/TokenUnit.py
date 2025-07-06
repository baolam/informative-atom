from typing import List
from torch import nn, zeros, tensor, Tensor, stack, long as tLong, randn
from ...layer.ForwardLayer import ForwardLayer
from ...units.base import SoftUnit
from ...units.unit_behavior import MemoryBehavior
from ...units.manage import ReadOnlyUnit


class TokenUnit(SoftUnit):
    """
    Đơn vị hành vi cho một Token (đóng vai trò là đơn vị biểu diễn)
    """
    class TokenBehavior(MemoryBehavior):
        """
        Hành vi AI của một Token (dùng cài đặt mặc định có trong đơn vị nhớ)
        """
        def __init__(self, _id=None, phi_dim = None, components = None, *args, **kwargs):
            super().__init__(_id, phi_dim, components, *args, **kwargs)
            self._beta = nn.Parameter(1.0)
            self._act = nn.Softmax(dim=1)
            self._norm = nn.LayerNorm(phi_dim)

        def forward(self, x, *args, **kwargs):
            # Cài đặt theo hình thức của Hopefield
            x = self._beta * x * self.hidden_behavior.T
            x = self._act(x)
            x = x * self.hidden_behavior
            x = self._norm(x)
            return x

    def __init__(self, _id=None, metadata={}, phi_dim : int = None, components : int = None, *args, **kwargs):
        if not isinstance(_id, str):
            raise TypeError(f"{_id} must be str!, (not None or something like that!)")

        behavior = self.TokenBehavior(_id, phi_dim=phi_dim, components=components)
        super().__init__(_id, behavior, metadata, *args, **kwargs)
    
    @property
    def embedding(self):
        return self._behavior.recognize().values
    

def check_sat(units : List[TokenUnit]):
    assert all(issubclass(type(unit), TokenUnit) for unit in units), "Unit must be inheriented from TokenUnit"

"""
Dùng để lưu các bộ từ dùng cho việc sinh từ
"""
class TokenDictionary(ForwardLayer):
    def __init__(self, units : ReadOnlyUnit, pad_token = "<PAD>", max_len : int = None, *args, **kwargs):
        check_sat(units)

        super().__init__(units, *args, **kwargs)

        # Chuyển đổi mỗi token thành mã chỉ số index
        self._vocab = { unit.id : i + 4 for i, unit in enumerate(units) }
        self._vocab.update({ "<UNK>" : 3, "<BOS>" : 0, pad_token : 1, "<EOS>" : 2 })

        # Cho phép chuyển đổi ngược index thành token
        self._index = { 3 : "<UNK>", 0 : "<BOS>", 1 : pad_token, 2 : "<EOS>" }
        self._index.update({ i + 4 : unit.id for i, unit in enumerate(units) })

        # Lưu trữ độ dài tối đa
        self._max_len = len(self._vocab)

    @property
    def vocab(self):
        return self._vocab
    
    @property
    def index(self):
        return self._index

    def token_to_index(self, token_lists : List[List[str]]) -> Tensor:
        """
        Chuyển đổi các Token thành các chỉ số Index
        """
        indexed_sequences = [
            [self._vocab.get(token, len(self._units) + 1) for token in tokens]
            for tokens in token_lists
        ]

        max_len = max(len(seq) for seq in indexed_sequences)
        if self._max_len < max_len:
            raise ValueError("Max_len exceeded!")

        padded_sequences = [
            seq + [0] * (self._max_len - len(seq))
            for seq in indexed_sequences
        ]

        return tensor(padded_sequences, dtype=tLong)
    
    def index_to_token(self, indicies : List[int]):
        raw_output = [ self._index[index] for index in indicies ]
        return raw_output

    @property
    def max_len(self):
        return self._max_len