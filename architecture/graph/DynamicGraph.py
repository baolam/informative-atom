from .base import BaseGraph
from ..units.manage import MutableUnit
from .StaticGraph import StaticGraph

def check_from_to(units : MutableUnit, _from : str, _to : str):
    if not _from in units:
        raise ValueError(f"unit list does not contain {_from}")
    if not _to in units:
        raise ValueError(f"unit list does not contain {_to}")


class DynamicGraph(BaseGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_unit(self, unit):
        self._units.append(unit)

    def del_unit(self, unit):
        self._units.pop(unit)

    def add_relation(self, relation):
        check_from_to(self._units, relation._from, relation._to)
        self._relations.append(relation)

    def del_relation(self, relation):
        check_from_to(self._units, relation._from, relation._to)
        self._relations.pop(relation)

    def as_static_graph(self) -> StaticGraph:
        return StaticGraph(self._units, self._relations)