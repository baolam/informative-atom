import pytest
import torch
from architecture.units.CombineRepresent import CombineRepresent
from architecture.units.MemoryUnit import MemoryUnit


def test_combine_represent():
    mem = MemoryUnit(metadata={}, phi_dim=128, components=64)

    with pytest.raises(TypeError):
        CombineRepresent(metadata={})
    with pytest.raises(TypeError):
        CombineRepresent(metadata={}, phi_dim=128)
    with pytest.raises(TypeError):
        CombineRepresent(metadata={}, phi_dim=128, m_dim=4)
    
    combine = CombineRepresent(metadata={}, phi_dim=128, m_dim=4, mem_unit=mem)

    x = torch.rand((32, 4, 128))
    y = combine(x)

    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 32
    assert y.shape[1] == 128