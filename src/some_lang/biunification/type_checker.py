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

    @abc.abstractmethod
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        pass


class UTypeHead(abc.ABC):
    @abc.abstractmethod
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        pass

    def component_types(self) -> typing.Iterator[tuple[str, int]]:
        # the default impl assumes dataclasses
        for field in dataclasses.fields(self):
            yield field.name, getattr(self, field.name)

    @abc.abstractmethod
    def reify(self, engine: TypeCheckerCore) -> typing.Any:
        pass


TypeNode: typing.TypeAlias = typing.Literal["Var"] | VTypeHead | UTypeHead
Value = typing.NewType("Value", Node)
Use = typing.NewType("Use", Node)


class TypeCheckerCore:
    def __init__(self) -> None:
        self.r = Reachability()
        self.types: list[TypeNode] = []

    def get_type(self, t: int) -> TypeNode:
        return self.types[t]

    def reify(self, t: int) -> typing.Any:
        ty = self.get_type(t)
        if ty == "Var":
            return f"var{t}"
        return ty.reify(self)

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

    def __str__(self):
        out = []
        for i, t in enumerate(self.types):
            if t == "Var":
                t = f"t{i}"
            out.append(f"{'t'+str(i):>4}  {t}")
            for j in self.r.downsets[i]:
                out.append(f"{'t'+str(i):>4}  {t} <: t{j}")
        return "\n".join(out)

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

    def reify_all(self, tvar_constructor=object, types=None):
        types = types or [None] * len(self.types)

        def recurse(t):
            if types[t] is not None:
                return types[t]

            if self.types[t] == "Var":
                inflows = [recurse(i) for i in self.r.upsets[t]]
                if inflows:
                    result = make_union(*inflows)
                else:
                    outflows = set(recurse(i) for i in self.r.downsets[t])
                    if not outflows:
                        result = tvar_constructor()
                    else:
                        print(inflows)
                        print(outflows)
                        raise NotImplementedError()
            else:
                result = self.reify(t)
            types[t] = result
            return result

        for i, t in enumerate(self.types):
            recurse(i)

        return types


def check_heads(lhs: VTypeHead, rhs: UTypeHead) -> list[tuple[Value, Use]]:
    return rhs.check(lhs)


def make_union(*types):
    match types:
        case ():
            raise NotImplementedError()
        case (t,):
            return t
        case _:
            u = Union(set(types))
            if len(u.types) == 1:
                u = u.types.pop()
            return u


class Union:
    def __init__(self, types):
        self.types = set()
        for t in types:
            self.add(t)

    def add(self, t):
        types_out = set()
        for tu in self.types:
            if t.is_supertype(tu):
                types_out.add(t)
            elif tu.is_supertype(t):
                return
            else:
                types_out.add(tu)
        types_out.add(t)
        self.types = types_out

    def __repr__(self):
        return f"Union{self.types}"
