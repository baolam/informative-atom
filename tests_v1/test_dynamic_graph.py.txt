import pytest
from architecture.utils.id_management import generate_id
from architecture.graph.DynamicGraph import DynamicGraph
from . import *

def build_unit(_id : str, metadata = {}):
    return FakeHard(_id, FakeBehavior(_id), metadata=metadata)

def build_relation(_from, _to, metadata={}):
    return FakeRelation(_from, _to, generate_id(), metadata=metadata)

def test_dynamic_graph_v2():
    new_graph = DynamicGraph()

    raw_units = [build_unit("6"), build_unit("7"), build_unit("8"), build_unit("9"), build_unit("10")]
    raw_edges = [build_relation("6", "8"), build_relation("7", "10"), build_relation("6", "10"), build_relation("8", "9"), build_relation("9", "10")]
    new_graph + ("unit", raw_units)
    new_graph + ("relation", raw_edges)

    assert len(new_graph.units) == 5
    assert len(new_graph.relations) == 5
    assert new_graph.units[1] == raw_units[1]
    assert new_graph.relations[3] == raw_edges[3]
