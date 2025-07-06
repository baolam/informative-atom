from torch import nn, triu, ones, no_grad, tensor, long, cat
from ...units.base import SoftUnit
from ...units.base_behavior import NonCodingBehavior
from ...utils.id_management import generate_id
from .positional import PositionalEncoding
from .generation import GenerateStrategy
from .TokenUnit import TokenDictionary


class GrammarUnit(SoftUnit):
    """
    Đơn vị ngữ pháp.
    Đơn vị phụ trách việc sinh ra token.
    """

    class GrammarBehavior(NonCodingBehavior):
        """
        Cài đặt hành vi hoạt động của đơn vị ngữ pháp
        """    
        def __init__(self, token_dictionary : TokenDictionary, _id=None, phi_dim : int = None, *args, **kwargs):
            super().__init__(_id, *args, **kwargs)
            self._token_dictionary = token_dictionary
            self._positional_encoding = PositionalEncoding(_id = _id, max_len=token_dictionary.max_len, *args, **kwargs)
            
            # Tiến hành cài đặt chú ý để xử lí
            self._mha = nn.MultiheadAttention(embed_dim=phi_dim, batch_first=True)

            # Chuẩn hoá
            self._norm = nn.LayerNorm(phi_dim)

            # Bộ hình thành xác suất từ
            self._ffn = nn.Sequential(
                nn.Linear(phi_dim),
                nn.ReLU(),
                nn.Linear(phi_dim, self._token_dictionary.max_len)
            )

        def forward(self, q, internal_memory, skip_activate : bool = False, *args, **kwargs):
            """
            Args:
            q (Tensor) : truy vấn đầu vào
            internal_memory (TensorLong) : mã vị trí, dùng cho sinh từ

            Mechanism:
            Cơ chế sinh từ là one-to-many (dùng duy nhất q làm đầu vào chỉ định)
            """
            # Lấy một số tham số của internal_memory
            __, length = internal_memory.size()

            # ------------------------------------------------------
            # Có thể cài đặt tính toán nhanh ở đây
            # Hình thành ma trận trọng số ứng riêng với truy vấn
            embedding_matrix = self._token_dictionary(q)        
            # ------------------------------------------------------

            # Trả về vector biểu diễn ứng với chỉ số token đầu vào
            state = nn.functional.embedding(internal_memory, embedding_matrix, padding_idx=0)
            
            # Tiến hành tăng cường biểu diễn bằng mã vị trí
            state = self._positional_encoding(state)
            
            # Áp dụng cơ chế Attention
            attn_mask = triu(ones(length, length), diagonal=1).bool().to(q.device)
            state, _ = self._mha(state, state, state, attn_mask=attn_mask)
            state = self._norm(state)
            
            # Hình thành tính toán từ
            state = self._ffn(state)
            if not skip_activate:
                state = self._softmax(state)
            return state

    def __init__(self, _id=None, generator : GenerateStrategy = None, *args, **kwargs):
        _id = generate_id()
        behavior = self.GrammarBehavior(_id=_id, *args, **kwargs)
        super().__init__(_id, behavior=behavior, *args, **kwargs)

        # Chiến lược sinh token
        self._generator = generator
    
    def change_strategy(self, strategy : GenerateStrategy):
        """
        Cho phép cập nhật chiến lược sinh token.
        """
        self._generator = strategy

    def generate(self, q, *args, **kwargs):
        """
        Sinh ra token, thực thi bằng cách apply chiến lược
        """
        if q.size()[0] > 1:
            raise ValueError("q.shape must be (1, phi_dim)!")
        
        additional_result = []
        tokens = [0]

        internal_memory = tensor([[0]], dtype=long).unsqueeze(-1).to(q.device)

        for __ in range(self._behavior._token_dictionary.max_len):
            with no_grad():
                logits = self.forward(q, internal_memory)
                
                # Sử dụng chiến lược sinh token
                result, next_token = self._generator(logits[0, -1, :])

                additional_result.append(result)
                tokens.append(next_token)

                # Kiểm tra mã (<EOS>) thoát dừng
                if next_token == 2:
                    break

                # Ghép nối chuỗi để sinh từ tiếp theo
                internal_memory = cat([internal_memory, tensor([[next_token]], dtype=long).to(q.device)], dim=0)
        
        return self.index_to_token(tokens), additional_result
    
    def token_to_index(self, *args, **kwargs):
        return self._behavior.token_to_index(*args, **kwargs)
    
    def index_to_token(self, *args, **kwargs):
        return self._behavior.index_to_token(*args, **kwargs)