from networkx import topological_sort
from .StaticGraph import StaticGraph
from ..utils.list_operator import ReadOnlyList


class ForwardGraph(StaticGraph):
    """
    Là đồ thị lan truyền dùng cho việc dựng và quy hoạch các Layer.
    Bản chất là đồ thị tĩnh, tức không thể thay đổi được các cạnh và đỉnh.
    """
    def __init__(self, units, relations, *args, **kwargs):
        super().__init__(units, relations, *args, **kwargs)
        self.__order = list(topological_sort(self.as_digraph_network()))
        self.__mapkeytoindex = { v : i for i, v in enumerate(self.__order) }
        
        self.__labels = [ 0 for _ in range(len(self.__order)) ]
        self.__calc_label()

    @property
    def order(self):
        return ReadOnlyList(self.__order)
    
    @property
    def keytoindex(self):
        return self.__mapkeytoindex
    
    @property
    def maxlabel(self):
        return self.__labels[-1]

    def _retrieve_unit(self, label : int) -> ReadOnlyList:
        """
        Truy xuất ra các node_id dựa trên nhãn được cấp
        """
        unit_ids = []

        for i, _label in enumerate(self.__labels):
            if label == _label:
                unit_ids.append( self.__order[i] )

        return ReadOnlyList(unit_ids)

    def __calc_label(self):
        """
        Tiến hành gán nhãn để phân lớp cho các đỉnh lan truyền
        """
        for i, node_id in enumerate(self.__order):
            in_relation = self.in_unit(node_id)
            if len(in_relation) > 0:
                self.__labels[i] = self.__labels[self.__mapkeytoindex[in_relation[0].id]] + 1

    # def as_sequence(self) -> SequenceLayer:
    #     raise NotImplementedError("as_sequence method must be implemented!")