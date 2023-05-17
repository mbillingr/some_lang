from __future__ import annotations
import abc
import dataclasses
from typing import Optional, Any

from eopl_explicit_refs.abstract_syntax import Symbol
from eopl_explicit_refs.vtable_manager import VtableManager, VtableIndex


class Type(abc.ABC):
    def __init__(self):
        self.implemented_interfaces: set[InterfaceType] = set()

    def declare_impl(self, ift: InterfaceType):
        self.implemented_interfaces.add(ift)

    def implements(self, ift: InterfaceType) -> bool:
        return ift in self.implemented_interfaces

    def find_method(self, name: str) -> Optional[Any]:
        return None


class TypeVar(Type):
    __match_args__ = ("name", "constraint")

    def __init__(self, name: str, constraints):
        self.name = name
        self.constraints = constraints
        self.type = None

    def is_fresh(self) -> bool:
        return self.type is None

    def set_type(self, ty: Type):
        if isinstance(ty, TypeVar):
            if ty is self:
                return
            raise TypeError(self, ty)
        if self.type is not None and self.type != ty:
            raise TypeError(self.type, ty)
        self.type = ty

    def find_method(self, name: str) -> Optional[Any]:
        for c in self.constraints:
            m = c.find_method(name)
            if m is not None:
                return m

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        return NotImplemented

    def __hash__(self):
        return id(self)


class TypeSchema(Type):
    def __init__(self, ty: Type):
        self.ty = ty
        self.instantiations = []

    def instantiate(self):
        tvars = {}

        def handle_tvar(t: TypeVar, tvars):
            try:
                return tvars[t]
            except KeyError:
                fresh_var = TypeVar(t.name, t.constraints)
                tvars[t] = fresh_var
                return fresh_var

        ty = self._substitute(self.ty, tvars, handle_tvar=handle_tvar)

        n = len(self.instantiations)
        self.instantiations.append((ty, tvars))
        return ty, n

    def concrete_instantiations(self):
        def substitution(t: TypeVar, subs):
            if subs[t].type is None:
                raise TypeError(f"{t} was never substituted")
            return subs[t].type

        for ty, tvars in self.instantiations:
            yield self._substitute(self.ty, tvars, handle_tvar=substitution)

    @staticmethod
    def _substitute(t: Type, subs, handle_tvar):
        match t:
            case TypeVar():
                return handle_tvar(t, subs)
            case AnyType() | NullType() | BoolType() | IntType():
                return t
            case NamedType():
                return t
            case InterfaceType():
                return t
            # case RecordType(fields):
            #    return RecordType({f: recur(t) for f, t in fields.items()})
            case FuncType(arg, ret):
                return FuncType(
                    TypeSchema._substitute(arg, subs, handle_tvar), TypeSchema._substitute(ret, subs, handle_tvar)
                )
            case _:
                raise NotImplementedError(f"{type(t).__name__}({t})")


class NamedType(Type):
    __match_args__ = ("name", "type")

    def __init__(self, fqn: str, ty: Type):
        super().__init__()
        self.name = fqn
        self.type = ty
        self._methods = {}

    def set_type(self, ty: Type):
        assert self.type is None
        self.type = ty

    def find_method(self, name: str) -> Optional[Any]:
        return self._methods.get(name)

    def add_method(self, name: str, fqn: str, signature: Type):
        self._methods[name] = (fqn, signature)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        return NotImplemented

    def __hash__(self):
        return id(self)


@dataclasses.dataclass(frozen=True)
class AnyType(Type):
    def __str__(self):
        return "_"


@dataclasses.dataclass(frozen=True)
class NullType(Type):
    def __str__(self):
        return "()"


@dataclasses.dataclass(frozen=True)
class BoolType(Type):
    def __str__(self):
        return "Bool"


@dataclasses.dataclass(frozen=True)
class IntType(Type):
    def __str__(self):
        return "Int"


@dataclasses.dataclass(frozen=True)
class BoxType(Type):
    item_t: Type

    def __str__(self):
        return f"@{self.item_t}"


@dataclasses.dataclass(frozen=True)
class ListType(Type):
    item_t: Type

    def __str__(self):
        return f"[{self.item_t}]"


@dataclasses.dataclass(frozen=True)
class RecordType(Type):
    fields: dict[Symbol, Type]

    def __str__(self):
        return f"[{', '.join(f'{n}: {t}' for n, t in self.fields.items())}]"

    def __hash__(self):
        return hash(tuple(self.fields.items()))


@dataclasses.dataclass(frozen=True)
class FuncType(Type):
    arg: Type
    ret: Type

    def __str__(self):
        return f"{self.arg}->{self.ret}"


class InterfaceType(Type):
    __match_args__ = ("name", "methods")

    def __init__(self, fqn: str, methods: dict[str, FuncType]):
        super().__init__()
        self.fully_qualified_name = fqn
        self.methods = None
        if methods:
            self.set_methods(methods)

    def set_methods(self, methods):
        assert self.methods is None
        self.methods = methods

    def find_method(self, name: str) -> Optional[tuple[Type, int]]:
        method = self.methods.get(name)
        return method and ("virtual", self.fully_qualified_name, method)

    def __str__(self):
        return f"{self.fully_qualified_name}"

    def __eq__(self, other):
        return self is other or other.implements(self)

    def __hash__(self):
        return id(self)
