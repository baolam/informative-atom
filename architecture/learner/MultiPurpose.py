from typing import Tuple, Callable
from .Learner import Learner
from ..layer.IntepretationLayer import IntepretationLayer
from .utils import assign_loss
from ..utils.list_operator import ReadOnlyList

def check_sat(layer : IntepretationLayer):
    assert len(layer.units) > 1, "multi purpose only!"

class MultiPurpose(Learner):
    """
    Mục tiêu cho lớp này là dùng AI để cài đặt tối ưu các thuộc tính
    """
    def __init__(self, problem, *args, **kwargs):
        check_sat(problem.intepretation_layer)
        super().__init__(problem, *args, **kwargs)

    def __learnable_final_layer(self):
        properties = []
        
        for unit in self._problem.intepretation_layer:
            if self.infor[unit.id]["learnable"]:
                properties.append(unit.metadata["as_name"])

        self._learnbale_properties = ReadOnlyList(properties)

    @property
    def loss(self):
        return self._loss_component
    
    @loss.setter
    def loss(self, infor : Tuple[str, Callable]):
        self._loss_component = assign_loss(self._loss_component, infor)
    
    def forward(self, *args, **kwargs):
        return self._problem(*args, **kwargs)
    
    def train(self, epochs, loader, optimizer, *args, **kwargs):
        self.__learnable_final_layer()

        infors = []

        for __ in range(epochs):
            for x, y in loader:
                optimizer.step()
                
                y_hat = self.forward(x)
                loss, summary = self.combine_loss(y_hat, y)
                loss.backward()

                optimizer.step()
                infors.append(summary)

        return infors

    def combine_loss(self, y_hat, y, *args, **kwargs):
        loss = 0.

        summary_loss = { }
        for property_name in self._learnbale_properties:
            loss_fn = self._loss_component[property_name]
            loss_item = loss_fn(y_hat[property_name], y[property_name])
            loss = loss + loss_item

            summary_loss[property_name] = loss_item.item()

        summary_loss["total_loss"] = loss.item()
        return loss, summary_loss
    
    def compile(self, *args, **kwargs):
        pass