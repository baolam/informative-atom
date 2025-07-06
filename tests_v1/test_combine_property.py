import pytest
from torch import nn, randn
from architecture_v1.utils.list_operator import ReadOnlyList
from architecture_v1.units.CombineProperty import OptionUtilization, BooleanUtilization, ScaleUtilization
from . import *

def test_scale_unit():
    scale = ScaleUtilization(property_name="cuteness_level", phi_dim=128, m_dim=4, coding_behavior=FakeBehavior("1"))

    x = randn(32, 4, 128)
    y = scale(x)

    assert y.shape[0] == 32
    assert y.shape[1] == 1

    x = randn(1, 4, 128)
    ans = scale.intepret(x)

    assert isinstance(ans, dict)
    assert "cuteness_level" in ans
    assert isinstance(ans["cuteness_level"], float)

def test_boolean_unit():
    boolean = BooleanUtilization(coding_behavior=FakeBehavior("1"), property_name="lovely", phi_dim=128, m_dim=4)

    x = randn(32, 4, 128)
    y = boolean(x)

    assert y.shape[0] == 32
    assert y.shape[1] == 1

    x = randn(1, 4, 128)
    ans = boolean.intepret(x)
    
    assert isinstance(ans, dict)
    assert "lovely" in ans
    assert isinstance(ans["lovely"], bool)
    # assert ans["threshold"] == 0.5

def test_option_unit():
    persons = ReadOnlyList(["lâm", "phong", "huy", "vũ"])
    options = OptionUtilization(coding_behavior=FakeBehavior("1"), property_name="persons", options=persons, m_dim=4, phi_dim=128)

    x = randn(32, 4, 128)
    y = options(x)

    assert y.shape[0] == 32
    assert y.shape[1] == 4

    x = randn(1, 4, 128)
    ans = options.intepret(x)

    assert isinstance(ans, dict)
    assert "persons" in ans

    # assert False, ans