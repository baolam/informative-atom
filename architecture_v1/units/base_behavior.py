from abc import ABC, abstractmethod
from typing import Any
from torch import nn


class Behavior(ABC):
    """
    Đây là lớp dựng hành vi thuộc tính của một Behavior.

    Hành vi là một hành động được thực thi để biến X thành Y.
    """
    def __init__(self, _id : str = None, *args, **kwargs):
        super().__init__()
        self.__id = _id
    
    @property
    def id(self):
        return self.__id
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Là cách hành động ứng với dữ liệu đầu vào (hành động này là hành động để biến X thành Y)
        """
        # raise NotImplementedError("forward method must be implemented!")
        pass
    
    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Phương thức lưu hành động
        """
        # raise NotImplementedError("save method must be implemented!")
        pass
    
    @abstractmethod
    def recognize(self, *args, **kwargs) -> Any:
        """
        Nhận đầu vào, quyết định xem đầu vào đó có phù hợp với hành động đang phụ
        trách hay không?
        """
        raise NotImplementedError("recognize method must be implemented!")

    @staticmethod
    def load(*args, **kwargs):
        raise NotImplementedError("load method must be implemented!")

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)


class NonCodingBehavior(Behavior, nn.Module, ABC):
    def __init__(self, _id = None ,*args, **kwargs):
        super().__init__(_id, *args, **kwargs)
    


class CodingBehavior(Behavior, ABC):
    def __init__(self, _id = None, *args, **kwargs):
        super().__init__(_id, *args, **kwargs)