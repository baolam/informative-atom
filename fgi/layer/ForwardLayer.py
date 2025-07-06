from torch import nn, empty, Tensor
from .base import StaticLayer
from ..units.base import SoftUnit
from ..utils.save_load import save_management_ext


class NonCodeForwardLayer(StaticLayer, nn.Module):
    """
    Cài đặt lớp lan truyền
    """
    def __init__(self, units, _id = None, *args, **kwargs):
        assert all(issubclass(type(unit), SoftUnit) for unit in units)
        super().__init__(units, _id ,*args, **kwargs)
        self._units = nn.ModuleList(units)
        
    def forward(self, x : Tensor, *args, **kwargs):
        if len(self._units) == 1:
            return self._units[0](x, *args, **kwargs)

        batch_size, phi_dim = x.size()[0], x.size()[1]
        if len(x.size()) == 3:
            phi_dim = x.size()[2]
            
        tempo = empty(len(self._units), batch_size, phi_dim, dtype=x.dtype, device=x.device)

        for i, unit in enumerate(self._units):
            tempo[i] = unit(x, *args, **kwargs)
        
        tempo = tempo.transpose(0, 1)
        return tempo

    def save(self, folder_name, *args, **kwargs):
        save_management_ext(self.metadata, self.id, folder_name)