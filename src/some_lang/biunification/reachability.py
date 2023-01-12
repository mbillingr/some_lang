import abc


Node = int


class Reachability:
    def __init__(self):
        self.upsets: list[set[Node]] = []
        self.downsets: list[set[Node]] = []

    def add_node(self) -> Node:
        i = len(self.upsets)
        self.upsets.append({i})
        self.downsets.append({i})
        return i

    def add_edge(self, lhs: Node, rhs: Node) -> list[(Node, Node)]:
        if rhs in self.downsets[lhs]:
            return []

        out = []
        for lhs2 in self.upsets[lhs]:
            for rhs2 in self.downsets[rhs]:
                if rhs2 not in self.downsets[lhs2]:
                    self.downsets[lhs2].add(rhs2)
                    self.upsets[rhs2].add(lhs2)
                    out.append((lhs2, rhs2))

        return out
