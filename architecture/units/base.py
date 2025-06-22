from typing import Any
from abc import ABC, abstractmethod
from .behavior import CodingBehavior, NonCodingBehavior


class BaseUnit(ABC):
    def __init__(self, _id : str = None, metadata : dict = {}, *args, **kwargs):
        super().__init__()
        if not isinstance(_id, str):
            raise TypeError("_id must be str!")
        
        self.__id = _id
        self._metadata = metadata

    @property
    def id(self):
        return self.__id
    
    @property
    def metadata(self):
        return self._metadata

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def add_meta(self, key : str, value : Any, *args, **kwargs):
        # raise NotImplementedError("add_meta method must be implemented!")
        pass
    
    @abstractmethod
    def pop_meta(self, key : str, *args, **kwargs) -> Any:
        # raise NotImplementedError("pop_meta method must be implemented!")
        pass
    
    @abstractmethod
    def update_meta(self, key : str, value : Any, *args, **kwargs):
        # raise NotImplementedError("update_meta method must be implemented!")
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        # raise NotImplementedError("forward method must be implemented!")
        pass
    
    @abstractmethod
    def as_coding_view(self, *args, **kwargs) -> CodingBehavior:
        # raise NotImplementedError("as_coding_view method must be implemented!")
        pass
    
    @abstractmethod
    def as_model_view(self, *args, **kwargs) -> NonCodingBehavior:
        # raise NotImplementedError("as_model_view method must be implemented!")
        pass
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class HardUnit(BaseUnit, ABC):
    def __init__(self, _id = None, behavior : CodingBehavior = None, metadata = {}, *args, **kwargs):
        super().__init__(_id, metadata, *args, **kwargs)
        if not isinstance(behavior, CodingBehavior):
            raise TypeError("behavior must be inheriented from CodingBehavior!")
        self._behavior = behavior
    
    def forward(self, *args, **kwargs):
        return self._behavior(*args, **kwargs)
    
    def as_coding_view(self, *args, **kwargs):
        return self._behavior
    
    def as_model_view(self, *args, **kwargs):
        return None
    

class SoftUnit(BaseUnit, ABC):
    def __init__(self, _id = None, behavior : NonCodingBehavior = None, metadata = {}, *args, **kwargs):
        super().__init__(_id, metadata, *args, **kwargs)
        if not isinstance(behavior, NonCodingBehavior):
            raise TypeError("behavior must be inheriented from NonCodingBehavior!")
        self._behavior = behavior
    
    def forward(self, *args, **kwargs):
        return self._behavior(*args, **kwargs)
    
    def as_coding_view(self, *args, **kwargs):
        return None
    
    def as_model_view(self, *args, **kwargs):
        return self._behavior
    

class HybridUnit(BaseUnit, ABC):
    """
    Đây là hình thức kết hợp hành vi lập trình và phi lập trình trong đơn vị.
    Mục tiêu của hình thức này là tăng tính diễn giải của mô hình.
    """
    def __init__(self, _id = None, coding_behavior : CodingBehavior = None, non_coding_behavior : NonCodingBehavior = None, metadata = {}, *args, **kwargs):
        super().__init__(_id, metadata, *args, **kwargs)
        if not isinstance(coding_behavior, CodingBehavior):
            raise TypeError("coding_behavior must be inheriented from CodingBehavior!")
        if not isinstance(non_coding_behavior, NonCodingBehavior):
            raise TypeError("non_coding_behavior must be inheriented from NonCodingBehavior!")
        self._code = coding_behavior
        self._non_code = non_coding_behavior
    
    def as_coding_view(self, *args, **kwargs):
        return self._code
    
    def as_model_view(self, *args, **kwargs):
        return self._non_code
    
    def forward(self, *args, **kwargs):
        return self._non_code(*args, **kwargs)
    
    @abstractmethod
    def intepret(self, *args, **kwargs):
        """
        Diễn giải ý nghĩa của đơn vị.
        Do đơn vị chứa cả hai hành vi AI lẫn phi AI, nên tuỳ vào việc cài đặt mà cần diễn giải ý nghĩa để hai hành
        vi thống nhất với nhau
        """
        # raise NotImplementedError("intepret method must be implemented!")
        pass
    
    def _assign_coding_behavior(self, coding : CodingBehavior):
        self._code = coding

    def _assign_noncode_behavior(self, noncode : NonCodingBehavior):
        self._non_code = noncode