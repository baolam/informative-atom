from typing import Dict
from torch import Tensor, max
from .base import HybridUnit
from .base_behavior import CodingBehavior
from ..utils.id_management import generate_id
from ..utils.list_operator import ReadOnlyList
from ..utils.dict_operator import add_meta, pop_meta, update_meta
from .primitive import ScaleIntepreter, ProbabilityIntepreter, MultiClassIntepreter


class CombineProperty(HybridUnit):
    """
    Đơn vị tổng hợp các tính chất để giải quyết vấn đề.
    """
    def __init__(self, _id=None, coding_behavior = None, non_coding_behavior = None, *args, **kwargs):
        super().__init__(_id, coding_behavior, non_coding_behavior, *args, **kwargs)
    
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
    def __init__(self, property_name : str = None, coding_behavior : CodingBehavior = None, *args, **kwargs):
        _id = generate_id()
        non_coding_behavior = ScaleIntepreter(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, *args, **kwargs)

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
    def __init__(self, coding_behavior=None, property_name : str = None, *args, **kwargs):
        _id = generate_id()
        non_coding_behavior = ProbabilityIntepreter(_id, *args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, *args, **kwargs)

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
    def __init__(self, coding_behavior=None, property_name : str = None, options : ReadOnlyList = None, *args, **kwargs):
        _id = generate_id()

        if not isinstance(options, ReadOnlyList):
            raise TypeError("options must be ReadOnlyList!")

        non_coding_behavior = MultiClassIntepreter(_id, num_classes=len(options) ,*args, **kwargs)
        super().__init__(_id, coding_behavior, non_coding_behavior, *args, **kwargs)

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