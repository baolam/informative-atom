from .base import BaseGraph
from networkx import DiGraph
from ..units.manage import MutableUnit, ReadOnlyUnit, convert_mutable_to_readonly as unit_converter
from ..relations.manage import MutableRelation, ReadOnlyRelation, convert_mutable_to_readonly as relation_converter

def convert_mutable_readonly(obj : MutableUnit | ReadOnlyUnit | MutableRelation | ReadOnlyRelation) -> ReadOnlyUnit | ReadOnlyRelation:
    if isinstance(obj, MutableUnit):
        return unit_converter(obj)
    if isinstance(obj, MutableRelation):
        return relation_converter(obj)
    return obj


class StaticGraph(BaseGraph):
    """
    Là bản dựng lớp mà các phương thức thêm, xoá không được cho phép.

    Đây là bản dựng cơ sở cho các khai thác ở phía sau.
    """
    def __init__(self, units : MutableUnit | ReadOnlyUnit, relations : MutableRelation | ReadOnlyRelation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._units : ReadOnlyUnit = convert_mutable_readonly(units)
        self._relations : ReadOnlyRelation = convert_mutable_readonly(relations)
        self._adjacency_list = self.adjacency_list()

    def add_unit(self, unit):
        pass

    def del_unit(self, unit):
        pass

    def add_relation(self, relation):
        pass

    def del_relation(self, relation):
        pass

    def in_unit(self, unit_id : str) -> ReadOnlyUnit:
        """
        Danh sách các đơn vị vào của một đỉnh được chỉ định
        """
        units = []

        for relation in self._relations:
            if relation._to == unit_id:
                units.append(self._units.find(relation._from))

        return ReadOnlyUnit(units)

    def out_unit(self, unit_id : str) -> ReadOnlyUnit:
        """
        Danh sách các đơn vị ra của một đỉnh được chỉ định
        """
        ids = self._adjacency_list.out_nodes(unit_id)
        units = [ self._units.find(id) for id in ids ]
        return ReadOnlyUnit(units)

    def in_relations(self, unit_id : str):
        return self._adjacency_list.in_relations(unit_id)
    
    def out_relations(self, unit_id : str):
        return self._adjacency_list.out_relations(unit_id)
    
    def as_digraph_network(self) -> DiGraph:
        graph = DiGraph()

        for unit in self._units:
            graph.add_node(unit.id, **unit.metadata)

        for relation in self._relations:
            graph.add_edge(relation._from, relation._to, **relation.metadata)

        return graph