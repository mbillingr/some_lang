import abc


Node = int


class Reachability(abc.ABC):
    @abc.abstractmethod
    def add_node(self) -> Node:
        pass

    @abc.abstractmethod
    def add_edge(self, lhs: Node, rhs: Node) -> list[(Node, Node)]:
        pass


class NaiveReachability(Reachability):
    def __init__(self):
        self.upsets: list[set[Node]] = []
        self.downsets: list[set[Node]] = []

    def add_node(self) -> Node:
        i = len(self.upsets)
        self.upsets.append({i})
        self.downsets.append({i})

    def add_edge(self, lhs: Node, rhs: Node) -> list[(Node, Node)]:
        if rhs in self.downsets[lhs]:
            return []

        lhs_set = sorted(self.upsets[lhs])
        rhs_set = sorted(self.downsets[rhs])
        out = []

        for lhs2 in lhs_set:
            for rhs2 in rhs_set:
                if rhs2 not in self.downsets[lhs2]:
                    self.downsets[lhs2].add(rhs2)
                    self.upsets[rhs2].add(lhs2)
                    out.append((lhs2, rhs2))

        return out
