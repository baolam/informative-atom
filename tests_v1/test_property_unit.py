import pytest
from torch import nn, randn
from architecture_v1.units.PropertyUnit import PropertyUnit
from architecture_v1.units.base_behavior import NonCodingBehavior

def test_overall_property():
    with pytest.raises(TypeError):
        PropertyUnit(metadata={})
    with pytest.raises(TypeError):
        PropertyUnit(metadata={}, components=32)

def test_detail_property():
    my_proper = PropertyUnit(metadata={}, components=32, phi_dim=128)
    assert isinstance(my_proper.id, str)
    assert isinstance(my_proper._behavior, nn.Module)
    assert isinstance(my_proper._behavior, NonCodingBehavior)

    x = randn(12, 128)
    y = my_proper(x)

    assert y.shape[0] == 12
    assert y.shape[1] == 128    

    r = my_proper._behavior.recognize().values
    assert r.shape[0] == 128
    assert my_proper._behavior.save() == None

    memory = my_proper.raw_memory
    assert memory.shape[0] == 32
    assert memory.shape[1] == 128

def test_manage_property():
    my_proper = PropertyUnit(metadata={}, components=32, phi_dim=128)
    
    my_proper.add_meta("key", "haha")
    assert my_proper.metadata["key"] == "haha"
    assert "key" in my_proper.metadata

    my_proper.pop_meta("key")
    assert "key" not in my_proper.metadata

    with pytest.raises(TypeError):
        my_proper.update_meta("key", "hello_world")