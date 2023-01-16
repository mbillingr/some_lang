from __future__ import annotations
import abc
import dataclasses
import typing

from some_lang.biunification.reachability import Reachability, Node


class VTypeHead(abc.ABC):
    def component_types(self) -> typing.Iterator[tuple[str, int]]:
        # the default impl assumes dataclasses
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)


class UTypeHead(abc.ABC):
    @abc.abstractmethod
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        pass

    def component_types(self) -> typing.Iterator[tuple[str, int]]:
        # the default impl assumes dataclasses
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)


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
                if isinstance(lhs_head, VTypeHead) and isinstance(rhs_head, UTypeHead):
                    pending_edges += check_heads(lhs_head, rhs_head)

    def recheck(self):
        for lhs, edges in enumerate(self.r.downsets):
            for rhs in edges:
                lhs_head = self.types[lhs]
                rhs_head = self.types[rhs]
                if lhs_head != "Var" and rhs_head != "Var":
                    check_heads(lhs_head, rhs_head)

    def specialize(self):
        """Try to derive the most specific types possible"""
        while True:
            vars = (t for t, ty in enumerate(self.types) if ty == "Var")
            # variables where exactly one type flows into - these are trivial to specialize
            simple_use_vars = [
                t
                for t in vars
                if len(self.r.upsets[t]) == 1 and self.types[next(iter(self.r.upsets[t]))] != "Var"
            ]
            print(simple_use_vars)
            if not simple_use_vars:
                return

            for t in simple_use_vars:
                v = next(iter(self.r.upsets[t]))  # todo: .pop() instead?
                u = self.new_use(self.types[v].get_use())
                self.substitute(t, v, u)

    def substitute(self, t: int, vhead: Value, uhead: Use):
        for rhs in self.r.downsets[t]:
            self.flow(vhead, Use(rhs))
        for lhs in self.r.upsets[t]:
            self.flow(Value(lhs), uhead)

        while self.r.downsets[t]:
            rhs = next(iter(self.r.downsets[t]))
            self.r.rm_edge(t, rhs)
        while self.r.upsets[t]:
            lhs = next(iter(self.r.upsets[t]))
            self.r.rm_edge(lhs, t)

        self.types = [
            (ty if ty == "Var" else ty.substitute(t, vhead, uhead)) for ty in self.types
        ]

    def __str__(self):
        out = []
        for i, t in enumerate(self.types):
            if t == "Var":
                t = f"t{i}"
            out.append(f"{'t'+str(i):>4}  {t}")
            for j in self.r.downsets[i]:
                out.append(f"{'t'+str(i):>4}  {t} <: t{j}")
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
                case VTypeHead() as vth:
                    t_ = dst.new_val(dst_type_head)
                case UTypeHead() as uth:
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


def check_heads(lhs: VTypeHead, rhs: UTypeHead) -> list[tuple[Value, Use]]:
    return rhs.check(lhs)
