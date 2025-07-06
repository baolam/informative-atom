from torch import nn, randn, matmul
from .base_behavior import NonCodingBehavior


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