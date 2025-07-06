import pytest
from architecture_v1.utils.id_management import generate_id
from architecture_v1.relations.manage import ReadOnlyRelation, MutableRelation
from . import *

def build_relation(_from, _to):
    return FakeRelation(_from, _to, generate_id(), metadata={})

def test_relation_readonly():
    raws = [
        build_relation("1", "2"),
        build_relation("2", "5"),
        build_relation("4", "3"),
        build_relation("1", "3"),
        build_relation("3", "5")
    ]
    obj = ReadOnlyRelation(raws)

    assert len(obj) == 5
    assert obj[0].id == raws[0].id
    assert raws[4] in obj
    assert raws[3].id in obj

    with pytest.raises(TypeError):
        obj[4] = build_relation("haha", "haha")

def test_mutable_relation():
    raws = [
        build_relation("1", "2"),
        build_relation("2", "5"),
        build_relation("4", "3"),
        build_relation("1", "3"),
        build_relation("3", "5")
    ]

    obj = MutableRelation(raws)
    assert len(obj) == 5
    assert obj[4] == raws[4]
    assert build_relation("random", "12") not in obj

    del obj[3]
    assert len(obj) == 4
    assert obj[3] == raws[-1]

    with pytest.raises(TypeError):
        MutableRelation([ raws[0], raws[0] ])
    with pytest.raises(TypeError):
        obj[5] = build_relation("huhu", "hihi")
    with pytest.raises(TypeError):
        obj[0] = raws[0]
    with pytest.raises(AttributeError):
        obj.append("error!")
    with pytest.raises(ValueError):
        obj.append(raws[0])

    new_relation = build_relation("huhu", "hihi")
    obj[0] = new_relation
    assert new_relation in obj
    
    relations = [ 
        build_relation("34", "49"),
        build_relation("12", "25"),
        build_relation("12", "37")
    ]
    new_manager = MutableRelation(relations)

    assert len(new_manager) == 3
    for i, relation in enumerate(new_manager):
        assert relations[i] == relation
    deleted = relations[2]
    new_manager.pop(relations[2])
    assert len(new_manager) == 2
    assert deleted not in new_manager

    with pytest.raises(ValueError):
        new_manager.pop(build_relation("random", "1234"))