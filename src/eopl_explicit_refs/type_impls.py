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
        return ("virtual", self.fully_qualified_name, self.methods[name])

    def __str__(self):
        return f"{self.name}"

    def __eq__(self, other):
        return self is other or other.implements(self)

    def __hash__(self):
        return id(self)
