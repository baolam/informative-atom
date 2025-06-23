import pytest
import torch
from architecture.layer.DynamicLayer import DynamicLayer
from architecture.layer.RepresentationLayer import RepresentationLayer
from architecture.layer.IntepretationLayer import IntepretationLayer
from architecture.units.CombineProperty import BooleanUtilization, OptionUtilization, ScaleUtilization
from architecture.utils.list_operator import ReadOnlyList
from . import *

def build_unit(_id):
    return FakeSoft(_id, FakeNonCodeBehavior(_id), metadata={})

def test_dynamic_layer():
    dl = DynamicLayer()
    dl + [build_unit("12"), build_unit("24"), build_unit("36"), build_unit("48"), build_unit("57")]

    assert len(dl.units) == 5
    assert "12" in dl.units
    assert "6" not in dl.units

    with pytest.raises(ValueError):
        dl.forward()
    
    dl.del_unit("12")
    assert "12" not in dl.units

def test_static_layer():
    dl = DynamicLayer()
    dl + [build_unit("12"), build_unit("24"), build_unit("36"), build_unit("48"), build_unit("57")]

    with pytest.raises(TypeError):
        sl = dl.as_static()

def test_intepretation_layer():
    scale = ScaleUtilization(property_name="age", coding_behavior=FakeBehavior("1"), phi_dim=128, m_dim=4, metadata={})
    boolean1 = BooleanUtilization(property_name="sex", coding_behavior=FakeBehavior("2"), phi_dim=128, m_dim=4, metadata={})
    boolean2 = BooleanUtilization(property_name="happy", coding_behavior=FakeBehavior("3"), phi_dim=128, m_dim=4, metadata={})
    options = OptionUtilization(property_name="school", coding_behavior=FakeBehavior("4"), phi_dim=128, m_dim=4, options=ReadOnlyList(["a", "b", "c"]), metadata={})

    dl = DynamicLayer()
    dl + [scale, boolean1, boolean2, options]

    intepretation = IntepretationLayer(dl.units)
    x = torch.randn(32, 4, 128)
    y = intepretation(x)

    assert isinstance(y, dict)
    assert y["age"].shape[0] == 32 and y["age"].shape[1] == 1
    assert y["sex"].shape[0] == 32 and y["sex"].shape[1] == 1
    assert y["happy"].shape[0] == 32 and y["happy"].shape[1] == 1
    assert y["school"].shape[0] == 32 and y["school"].shape[1] == 3

    x = torch.randn(1, 4, 128)
    raws = intepretation.intepret(x)

    print(raws)
    assert isinstance(raws, dict)
    assert isinstance(raws["age"], float)
    assert isinstance(raws["sex"], bool)
    assert isinstance(raws["happy"], bool)