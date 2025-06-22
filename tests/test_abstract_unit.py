import pytest
from architecture.units.base import HardUnit, SoftUnit, HybridUnit
from . import *

def test_raises():
    with pytest.raises(TypeError):
        HardUnit()
        SoftUnit()
        HybridUnit()

def test_hard_unit():
    with pytest.raises(TypeError):
        FakeHard(_id="1")
    fake = FakeHard(_id="1", behavior=FakeBehavior(_id="1"), metadata={ "hello" : "world", "aha" : "aha" })
    assert fake.id == "1"
    assert fake.name == "FakeHard"
    assert len(fake.metadata) > 0
    assert fake.as_model_view() == None
    assert fake.as_coding_view() != None

def test_soft_unit():
    with pytest.raises(TypeError):
        FakeSoft(_id="1")
    fake = FakeSoft(_id="1", behavior=FakeNonCodeBehavior(_id="1"), metadata={ "lam" : "cute" })
    assert fake.id == "1"
    assert fake.name == "FakeSoft"
    assert fake.metadata["lam"] == "cute"
    assert fake.as_model_view() != None
    assert fake.as_coding_view() == None


def test_hybrid_unit():
    with pytest.raises(TypeError):
        FakeHybrid(_id="1")
        FakeHybrid(_id="1", coding_behavior=FakeBehavior(_id="1"))
    fake = FakeHybrid(_id="1", coding_behavior=FakeBehavior(_id="1"), non_coding_behavior=FakeNonCodeBehavior(_id="1"))
    assert fake.id == "1"
    assert fake.name == "FakeHybrid"
    assert len(fake.metadata) == 0
    assert fake.as_coding_view() != None
    assert fake.as_model_view() != None