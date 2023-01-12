from __future__ import annotations
import abc
import typing

from some_lang.biunification.reachability import Reachability, Node


class VTypeHead(abc.ABC):
    pass


class UTypeHead(abc.ABC):
    @abc.abstractmethod
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        pass


TypeNode: typing.TypeAlias = typing.Literal["Var"] | VTypeHead | UTypeHead
Value = typing.NewType("Value", Node)
Use = typing.NewType("Use", Node)


class TypeCheckerCore:
    def __init__(self) -> None:
        self.r = Reachability()
        self.types: list[TypeNode] = []

    def new_val(self, val_type: VTypeHead) -> Value:
        i = self.r.add_node()
        assert i == len(self.types)
        self.types.append(val_type)
        return Value(i)

    def new_use(self, constraint: UTypeHead) -> Use:
        i = self.r.add_node()
        assert i == len(self.types)
        self.types.append(constraint)
        return Use(i)

    def var(self) -> typing.Tuple[Value, Use]:
        i = self.r.add_node()
        assert i == len(self.types)
        self.types.append("Var")
        return Value(i), Use(i)

    def flow(self, lhs: Value, rhs: Use):
        pending_edges = [(lhs, rhs)]
        while pending_edges:
            lhs, rhs = pending_edges.pop()
            type_pairs_to_check = [(Value(l), Use(r)) for l, r in self.r.add_edge(lhs, rhs)]

            while type_pairs_to_check:
                lhs, rhs = type_pairs_to_check.pop()
                lhs_head = self.types[lhs]
                rhs_head = self.types[rhs]
                if isinstance(lhs_head, VTypeHead) and isinstance(rhs_head, UTypeHead):
                    pending_edges += check_heads(lhs_head, rhs_head)


def check_heads(lhs: VTypeHead, rhs: UTypeHead) -> list[tuple[Value, Use]]:
    return rhs.check(lhs)
