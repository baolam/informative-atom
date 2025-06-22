import pytest
import torch
from architecture.units.MemoryUnit import MemoryUnit

def test_raise_memory_unit():
    with pytest.raises(TypeError):
        MemoryUnit(metadata={})
        MemoryUnit(metadata={}, phi_dim=128)

def test_memory_unit():
    unit = MemoryUnit(metadata={}, components=64, phi_dim=128)
    assert isinstance(unit.id, str)
    assert unit.name == "MemoryUnit"
    assert unit.representation.shape[0] == 128

    x = torch.rand((32, 128))
    y = unit(x)

    assert y.shape[0] == 32
    assert y.shape[1] == 128