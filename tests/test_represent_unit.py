import pytest
from architecture.units.RepresentUnit import RepresentUnit
from . import *

def test_represent_unit():
    r = RepresentUnit("1", coding_behavior=FakeBehavior("1"), non_coding_behavior=FakeNonCodeBehavior("1"), metadata={})
    assert r.id == "1"

    res = { "haha" : "hello world", "123" : True }
    assert r.represent(res) == res

    r.add_meta("sorry", "work!")
    with pytest.raises(TypeError):
        r.add_meta("sorry", "work!")
    
    r.update_meta("sorry", "haha")
    assert r.metadata["sorry"] == "haha"

    ans = r.pop_meta("sorry")
    assert ans == "haha"

    with pytest.raises(TypeError):
        r.pop_meta("sorry")
    with pytest.raises(TypeError):
        r.update_meta("sorry", "hello world")
    
    assert r.intepret() == None