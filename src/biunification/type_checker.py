from __future__ import annotations
import abc
import dataclasses
import typing

from biunification.reachability import Reachability, Node


class VTypeHead(abc.ABC):
    def component_types(self) -> typing.Iterator[tuple[str, int]]:
        # the default impl assumes dataclasses
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)

    def substitute(self, t_old, t_new) -> VTypeHead:
        return self

    def check(self, rhs: UTypeHead) -> list[tuple[Value, Use]]:
        """Override on types that can't be checked by usage"""
        return []


class UTypeHead(abc.ABC):
    @abc.abstractmethod
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        pass

    def component_types(self) -> typing.Iterator[tuple[str, int]]:
        # the default impl assumes dataclasses
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)

    def substitute(self, t_old, t_new) -> UTypeHead:
        return self


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
            type_pairs_to_check = [
                (Value(l), Use(r)) for l, r in self.r.add_edge(lhs, rhs)
            ]

            while type_pairs_to_check:
                lhs, rhs = type_pairs_to_check.pop()
                lhs_head = self.types[lhs]
                rhs_head = self.types[rhs]
                if isinstance(lhs_head, VTypeHead):
                    pending_edges += lhs_head.check(rhs_head)
                    if isinstance(rhs_head, UTypeHead):
                        pending_edges += check_heads(lhs_head, rhs_head)

    def __str__(self):
        out = []
        for i, t in enumerate(self.types):
            if t == "Var":
                t = f"t{i}"
            out.append(f"{'t'+str(i):>4}  {t}")
            for j in self.r.downsets[i]:
                out.append(f"{'t'+str(i):>4}  {t} -> t{j}")
        return "\n".join(out)

    def reify(self, t: int):
        return f"{self.types[t]}{t}"

    def extract(self, t: int) -> TypeCheckerCore:
        """Extract the subgraph relevant for one specific type."""
        out = TypeCheckerCore()
        self.copy_type(t, out, {})
        return out

    def copy_type(self, t: int, dst: TypeCheckerCore, copied: dict[int, int]) -> int:
        try:
            return copied[t]
        except KeyError:
            pass

        type_head = self.types[t]

        if type_head == "Var":
            t_, _ = dst.var()
        else:
            components = {
                k: self.copy_type(s, dst, copied)
                for k, s in type_head.component_types()
            }
            dst_type_head = type_head.__class__(**components)
            match type_head:
                case VTypeHead():
                    t_ = dst.new_val(dst_type_head)
                case UTypeHead():
                    t_ = dst.new_use(dst_type_head)
                case other:
                    raise RuntimeError("Should not reach", other)

        copied[t] = t_

        for v in self.r.upsets[t]:
            v_ = self.copy_type(v, dst, copied)
            dst.flow(Value(v_), Use(t_))

        for u in self.r.downsets[t]:
            u_ = self.copy_type(u, dst, copied)
            dst.flow(Value(t_), Use(u_))

        return t_

    def all_upvtypes(self, t: int, seen: typing.Optional[set[int]] = None) -> set[int]:
        seen = seen or set()
        if t in seen:
            return set()
        seen.add(t)

        vts = set()

        if isinstance(self.types[t], VTypeHead):
            vts.add(t)

        for v in self.r.upsets[t]:
            vts |= self.all_upvtypes(v, seen)

        return vts

    def collapse_cycles(self):
        while True:
            cycle = self._find_cycle()
            if not cycle:
                return

            if not all(self.types[t] == "Var" for t in cycle):
                raise NotImplementedError(
                    "Non-var part of cycle -- don't know what to do..."
                )

            t0 = min(cycle)  # keep lowest type nr
            cycle = [t for t in cycle if t != t0]

            for t in cycle:
                self.types[t] = "erased"
                self.r.remove_node(t)
                self.types = [
                    th if isinstance(th, str) else th.substitute(t, t0)
                    for th in self.types
                ]

    def _find_cycle(self):
        not_part_of_cycle = [False] * len(self.types)
        visited = [False] * len(self.types)

        cycle = []

        class CycleDetected(Exception):
            pass

        def dfs(t):
            nonlocal cycle
            if not_part_of_cycle[t]:
                return
            if visited[t]:
                raise CycleDetected(cycle)
            visited[t] = True
            for u in self.r.downsets[t]:
                try:
                    dfs(u)
                except CycleDetected:
                    if len(cycle) < 2 or cycle[0] != cycle[-1]:
                        cycle.append(u)
                    raise
            not_part_of_cycle[t] = True

        try:
            for t in range(len(self.types)):
                dfs(t)
        except CycleDetected:
            if len(cycle) < 2 or cycle[0] != cycle[-1]:
                cycle.append(t)
            return cycle
        return None


def check_heads(lhs: VTypeHead, rhs: UTypeHead) -> list[tuple[Value, Use]]:
    return rhs.check(lhs)
