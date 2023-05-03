from __future__ import annotations
import abc
import dataclasses

from eopl_explicit_refs.abstract_syntax import Symbol


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

    def __init__(self, name: str, methods: dict[str, FuncType]):
        self.name = name
        self.methods = methods

    def __str__(self):
        return f"{self.name}"

    def set_methods(self, methods):
        assert self.methods is None
        self.methods = methods

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)
