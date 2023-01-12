import abc


Node = int


class Reachability:
    def __init__(self) -> None:
        self.upsets: list[set[Node]] = []
        self.downsets: list[set[Node]] = []

    def add_node(self) -> Node:
        i = len(self.upsets)
        self.upsets.append(set())
        self.downsets.append(set())
        return i

    def add_edge(self, lhs: Node, rhs: Node) -> list[tuple[Node, Node]]:
        if rhs in self.downsets[lhs]:
            return []

        self.downsets[lhs].add(rhs)
        self.upsets[rhs].add(lhs)
        out = [(lhs, rhs)]

        for lhs2 in self.upsets[lhs]:
            out += self.add_edge(lhs2, rhs)

        for rhs2 in self.downsets[rhs]:
            out += self.add_edge(lhs, rhs2)

        return out
