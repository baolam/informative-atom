from typing import Dict
from torch import nn, randn, matmul, Tensor, max
from .base import HybridUnit
from .behavior import NonCodingBehavior, CodingBehavior
from ..utils.id_management import generate_id
from ..utils.list_operator import ReadOnlyList
from ..utils.dict_operator import add_meta, pop_meta, update_meta


class DefaultBehavior(NonCodingBehavior):
    def __init__(self, _id=None, m_dim : int = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)
        if not isinstance(m_dim, int):
            raise TypeError("m_dim must be int!")
        # Trọng số đánh giá biểu hiện đầu vào
        self.attn_weights = nn.Parameter(randn(m_dim))
        self.act = nn.Softmax(dim=0)

    def forward(self, x ,*args, **kwargs):
        weights = self.attn_weights.view(-1, self.attn_weights.size()[0])
        score = self.act(weights)
        y = matmul(score, x)
        y = y.view(x.size()[0], -1)
        return y
    
    def recognize(self, *args, **kwargs):
        pass
    
    def save(self, *args, **kwargs):
        pass

class ScaleIntepreter(DefaultBehavior):
    def __init__(self, _id=None, m_dim = None, phi_dim : int = None, *args, **kwargs):
        super().__init__(_id, m_dim, *args, **kwargs)
        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        self.lin = nn.Linear(phi_dim, 1)
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = self.lin(x)
        return x


class ProbabilityIntepreter(DefaultBehavior):
    def __init__(self, _id=None, m_dim = None, phi_dim : int = None, *args, **kwargs):
        super().__init__(_id, m_dim, *args, **kwargs)
        self.lin = nn.Linear(phi_dim, 1)
        self.act = nn.Sigmoid()
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = self.lin(x)
        x = self.act(x)
        return x


class MultiClassIntepreter(DefaultBehavior):
    def __init__(self, _id=None, m_dim = None, phi_dim : int = None, num_classes : int = None, *args, **kwargs):
        super().__init__(_id, m_dim, *args, **kwargs)

        if not isinstance(phi_dim, int):
            raise TypeError("phi_dim must be int!")
        if not isinstance(num_classes, int):
            raise TypeError("num_classes must be int!")
        
        self.lin = nn.Linear(phi_dim, num_classes)
        self.act = nn.Softmax(dim=1)
    
    def forward(self, x, *args, **kwargs):
        x = super().forward(x, *args, **kwargs)
        x = self.lin(x)
        x = self.act(x)
        return x


class CombineProperty(HybridUnit):
    """
    Đơn vị tổng hợp các tính chất để giải quyết vấn đề.
    """
    def __init__(self, _id=None, coding_behavior = None, non_coding_behavior = None, metadata=..., *args, **kwargs):
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)
    
    def add_meta(self, key, value, *args, **kwargs):
        self._metadata = add_meta(self._metadata, key, value)
    
    def pop_meta(self, key, *args, **kwargs):
        self._metadata, popped_value = pop_meta(self._metadata, key)
        return popped_value
        
    def intepret(self, *args, **kwargs):
        raise NotImplementedError("intepret method must be implemented!")
    
    def update_meta(self, key, value, *args, **kwargs):
        self._metadata = update_meta(self._metadata, key, value)

        

class ScaleUtilization(CombineProperty):
    """
    Đây là ứng dụng kết hợp các tính chất để đưa ra một thuộc tính mang tính dự đoán, Float
    """
    def __init__(self, metadata={}, property_name : str = None, coding_behavior : CodingBehavior = None, *args, **kwargs):
        _id = generate_id()
        non_coding_behavior = ScaleIntepreter(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

        if not isinstance(property_name, str):
            raise TypeError("property_name must be str!")
        
        self.add_meta("as_name", property_name)
    
    def intepret(self, x : Tensor, *args, **kwargs) -> Dict[str, float]:
        if x.shape[0] > 1:
            raise ValueError("x.shape must be (1, phi_dim)")
        x : Tensor = self.forward(x, *args, **kwargs)
        num = x.item()
        return { self._metadata["as_name"] : num }
    

class BooleanUtilization(CombineProperty):
    """
    Đây là ứng dụng kết hợp các tính chất để đưa ra một thuộc tính mang tính đúng/sai
    """
    def __init__(self, coding_behavior=None, metadata={}, property_name : str = None, *args, **kwargs):
        _id = generate_id()
        non_coding_behavior = ProbabilityIntepreter(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

        if not isinstance(property_name, str):
            raise TypeError("property_name must be str!")
        
        self.add_meta("as_name", property_name)
        self.add_meta("threshold", 0.5)
    
    def intepret(self, x, *args, **kwargs) -> Dict[str, bool]:
        if x.shape[0] > 1:
            raise ValueError("x.shape must be (1, phi_dim)")
        x = self.forward(x, *args, **kwargs)

        thres = self._metadata["threshold"]
        proba = x.item()

        return { self._metadata["as_name"] : proba >= thres, f"{self._metadata["as_name"]}.raw" : proba }
    

class OptionUtilization(CombineProperty):
    """
    Đây là ứng dụng kết hợp các tính chất để đưa ra thuộc tính mang tính chọn lựa các giá trị
    """
    def __init__(self, coding_behavior=None, metadata={}, property_name : str = None, options : ReadOnlyList = None, *args, **kwargs):
        _id = generate_id()

        if not isinstance(options, ReadOnlyList):
            raise TypeError("options must be ReadOnlyList!")

        non_coding_behavior = MultiClassIntepreter(_id, num_classes=len(options) ,*args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, metadata, *args, **kwargs)

        if not isinstance(property_name, str):
            raise TypeError("property_name must be str!")
        
        self.add_meta("as_name", property_name)
        self.add_meta("options", options._as_list())
    
    def intepret(self, x ,*args, **kwargs):
        if x.shape[0] > 1:
            raise ValueError("x.shape must be (1, phi_dim)")
        x = self.forward(x, *args, **kwargs)

        max_obj = max(x, dim=1)
        proba = max_obj.values.item()
        index = max_obj.indices.item()

        option = self._metadata["options"][index]
        return { self._metadata["as_name"] : option, f"{self._metadata["as_name"]}.raw" : proba }