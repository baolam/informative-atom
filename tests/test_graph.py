import pytest
from architecture.graph.DynamicGraph import DynamicGraph
from architecture.graph.StaticGraph import StaticGraph
from architecture.graph.ForwardGraph import ForwardGraph
from architecture.utils.id_management import generate_id
from architecture.utils.list_operator import ReadOnlyList
from architecture.units.manage import MutableUnit, ReadOnlyUnit
from architecture.relations.manage import ReadOnlyRelation
from architecture.relations.adjacency_list import AdjacencyList
from . import *

def build_unit(_id : str, metadata = {}):
    return FakeHard(_id, FakeBehavior(_id), metadata=metadata)

u1 = build_unit("1")
u2 = build_unit("2")
u3 = build_unit("3")
u4 = build_unit("4")
u5 = build_unit("5")

def build_relation(_from, _to, metadata={}):
    return FakeRelation(_from, _to, generate_id(), metadata=metadata)

r1 = build_relation("1", "2")
r2 = build_relation("1", "3")
r3 = build_relation("3", "4")
r4 = build_relation("3", "5")
r5 = build_relation("1", "5")

graph = DynamicGraph()

def test_dynamic_graph():
    graph.add_unit(u1)
    assert isinstance(graph.units, MutableUnit)
    assert len(graph.units) == 1

    graph.add_unit(u2)
    assert len(graph.units) == 2

    with pytest.raises(ValueError):
        graph.add_relation(r4)
    with pytest.raises(AttributeError):
        graph.add_unit(r1)

    graph.add_unit(u3)
    graph.add_unit(u4)
    graph.add_unit(u5)

    assert len(graph.units) == 5

    graph.add_relation(r1)
    graph.add_relation(r2)
    graph.add_relation(r3)
    graph.add_relation(r4)
    graph.add_relation(r5)

    assert len(graph.relations) == 5

def test_special_case_dp():
    r6 = build_relation("5", "6")
    with pytest.raises(ValueError):
        graph.add_relation(r6)

def test_static_graph():
    static = graph.as_static_graph()
    assert isinstance(static, StaticGraph)
    assert isinstance(static.adjacency_list(), AdjacencyList)
    assert isinstance(static.in_unit("2"), ReadOnlyUnit)
    assert static.in_unit("2")[0].id == "1"
    assert static.in_unit("4")[0].id == "3"
    assert len(static.out_unit("1")) == 3
    assert isinstance(static.in_relations("1"), ReadOnlyRelation)
    assert len(static.in_relations("2")) == 1
    assert static.in_relations("2")[0].id == r1.id
    assert len(static.out_relations("1")) == 3
    assert static.out_relations("1")[0].id == r1.id

def test_forward_graph():
    forward = ForwardGraph(graph.units, graph.relations)
    topo_sort = ["1", "2", "3", "4", "5"]
    assert isinstance(forward.order, ReadOnlyList)
    for i, code in enumerate(forward.order):
        assert code == topo_sort[i]
    assert forward.maxlabel == 2
    assert isinstance(forward._retrieve_unit(1), ReadOnlyList)
    unit_str = ["2", "3"]
    for i, code in enumerate(forward._retrieve_unit(1)):
        assert code == unit_str[i]
    assert list(forward.keytoindex.keys()) == topo_sort