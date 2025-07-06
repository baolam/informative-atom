from .Learner import Learner
from ..layer.IntepretationLayer import IntepretationLayer

def check_intepreation(intepretation_layer : IntepretationLayer):
    assert len(intepretation_layer.units) != 1, "One purpose only!"

class OnePurpose(Learner):
    """
    Đây là lớp học dành cho những vấn đề mà tối ưu chỉ có một mục đích duy nhất.

    Nghĩa là số đơn vị diễn giải ở lớp Intepretation là một.
    """
    def __init__(self, problem, *args, **kwargs):
        check_intepreation(problem.noncode.intepretation_layer)
        super().__init__(problem, *args, **kwargs)

    def forward(self, x, *args, **kwargs):
        y = self._problem(x, *args, **kwargs)
        key = self._problem.metadata["as_name"]
        return y[key]
    
    @property
    def loss(self):
        return self._loss_component

    @loss.setter
    def loss(self, loss_fn):
        self._loss_component = loss_fn

    def train(self, epochs, loader, optimizer, *args, **kwargs):
        self._problem.noncode.eval()

        losses = []

        for __ in range(epochs):
            for x, y in loader:
                optimizer.zero_grad()

                y_hat = self.forward(x)
                loss = self.combine_loss(y_hat, y)
                loss.backward()

                optimizer.step()
            
            losses.append(loss.item())
        
        return losses

    def combine_loss(self, y_hat, y, *args, **kwargs):
        return self._loss_component(y_hat, y)
    
    def compile(self, *args, **kwargs):
        pass