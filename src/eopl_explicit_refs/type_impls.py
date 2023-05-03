from __future__ import annotations
import abc
import dataclasses

from eopl_explicit_refs.abstract_syntax import Symbol
from eopl_explicit_refs.vtable_manager import VtableManager, VtableIndex


class Type(abc.ABC):
    def __init__(self):
        self.implemented_interfaces: set[InterfaceType] = set()

    def declare_impl(self, ift: InterfaceType):
        self.implemented_interfaces.add(ift)

    def implements(self, ift: InterfaceType) -> bool:
        return ift in self.implemented_interfaces


class NamedType(Type):
    __match_args__ = ("name", "type")

    def __init__(self, name: str, ty: Type):
        super().__init__()
        self.name = name
        self.type = ty

    def set_type(self, ty: Type):
        assert self.type is None
        self.type = ty

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


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

    def __init__(self, name: str, methods: dict[str, FuncType], vtm: VtableManager):
        self.name = name
        self.methods = None
        self.vtm = vtm
        self.virtuals: dict[str, VtableIndex] = {}
        if methods:
            self.set_methods(methods)

    def set_methods(self, methods):
        assert self.methods is None
        self.methods = methods
        self.virtuals = self.vtm.assign_virtuals(methods.keys())

    def as_virtual(self, method: str):
        return self.virtuals[method]

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)
