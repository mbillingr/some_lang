from __future__ import annotations
import abc
import contextlib
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

    def unify(self, other: Type):
        if self is other:
            return
        match self, other:
            case TypeVar(), _:
                self._unify(other)
            case _, TypeVar():
                other._unify(self)
            case _, InterfaceType():
                other._unify(self)
            case _:
                self._unify(other)

    def _unify(self, other: Type):
        if self != other:
            raise TypeError()

    def copy_unify(self, other: Type) -> Type:
        if self is other:
            return self
        match self, other:
            case TypeVar(), _:
                return self._copy_unify(other)
            case _, TypeVar():
                return other._copy_unify(self)
            case _, InterfaceType():
                return other._copy_unify(self)
            case _:
                return self._copy_unify(other)

    def _copy_unify(self, other: Type) -> Type:
        if self == other:
            return self
        raise TypeError(self, other)

    def substitute(self, substitution):
        try:
            return substitution[self]
        except KeyError:
            return self._substitute(substitution)

    def _substitute(self, substitution):
        return self


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

    def implements(self, ift: InterfaceType) -> bool:
        return ift in self.constraints

    def _unify(self, other: Type):
        if self is other:
            return
        elif self.is_fresh():
            self.set_type(other)
        else:
            self.type.unify(other)

    def _copy_unify(self, other: Type) -> Type:
        assert self.is_fresh()
        for c in self.constraints:
            if not other.implements(c):
                raise TypeError(f"{other} does not implement {c}")
        return other

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if self is other:
            return True
        if self.is_fresh():
            self.set_type(other)
            return True
        else:
            return self.type == other

    def __hash__(self):
        return id(self)


class TypeSchema(Type):
    def __init__(self, tvars: list[TypeVar], ty: Type):
        self.tvars = tvars
        self.ty = ty
        self.instantiations = []

    def fixed_instantiation(self) -> Type:
        substitution = {}
        for tv in self.tvars:
            tconst = NamedType(tv.name, AnyType())
            tconst.implemented_interfaces = tv.constraints
            substitution[tv] = tconst

        return self.ty.substitute(substitution)

    def unify_instantiation(self, t: Type) -> tuple[Type, int]:
        u = self.ty.copy_unify(t)

        try:
            idx = self.instantiations.index(u)
            return self.instantiations[idx], idx
        except ValueError:
            pass

        idx = len(self.instantiations)
        self.instantiations.append(u)
        return u, idx

    def concrete_instantiations(self):
        for idx, u in enumerate(self.instantiations):
            yield u, idx


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

    def _unify(self, other: Type):
        if self is not other:
            raise TypeError(self, other)

    def _copy_unify(self, other: Type) -> Type:
        if self is not other:
            raise TypeError(self, other)
        return self

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

    def _unify(self, other: Type):
        if not isinstance(other, BoxType):
            raise TypeError()
        else:
            self.item_t.unify(other.item_t)

    def _copy_unify(self, other: Type) -> Type:
        if not isinstance(other, BoxType):
            raise TypeError()
        else:
            return BoxType(self.item_t.copy_unify(other.item_t))

    def _substitute(self, substitution):
        return BoxType(self.item_t.substitute(substitution))

    def __str__(self):
        return f"@{self.item_t}"


@dataclasses.dataclass(frozen=True)
class ListType(Type):
    item_t: Type

    def _unify(self, other: Type):
        if not isinstance(other, ListType):
            raise TypeError(self, other)
        else:
            self.item_t.unify(other.item_t)

    def _copy_unify(self, other: Type) -> Type:
        if not isinstance(other, ListType):
            raise TypeError(self, other)
        else:
            return ListType(self.item_t.copy_unify(other.item_t))

    def _substitute(self, substitution):
        return ListType(self.item_t.substitute(substitution))

    def __str__(self):
        return f"[{self.item_t}]"


@dataclasses.dataclass(frozen=True)
class RecordType(Type):
    fields: dict[Symbol, Type]

    def _unify(self, other: Type):
        if not isinstance(other, RecordType):
            raise TypeError()
        if self.fields.keys() != other.fields.keys():
            raise TypeError()
        for k in self.fields:
            self.fields[k].unify(other.fields[k])

    def _copy_unify(self, other: Type) -> Type:
        if not isinstance(other, RecordType):
            raise TypeError(self, other)
        if self.fields.keys() != other.fields.keys():
            raise TypeError(self, other)
        return RecordType(
            {k: self.fields[k].copy_unify(other.fields[k]) for k in self.fields}
        )

    def _substitute(self, substitution):
        return RecordType(
            {k: t.substitute(substitution) for k, t in self.fields.items()}
        )

    def __str__(self):
        return f"[{', '.join(f'{n}: {t}' for n, t in self.fields.items())}]"

    def __hash__(self):
        return hash(tuple(self.fields.items()))


@dataclasses.dataclass(frozen=True)
class FuncType(Type):
    arg: Type
    ret: Type

    def _unify(self, other: Type):
        if not isinstance(other, FuncType):
            raise TypeError()
        self.arg.unify(other.arg)
        self.ret.unify(other.ret)

    def _copy_unify(self, other: Type) -> Type:
        if not isinstance(other, FuncType):
            raise TypeError(self, other)
        return FuncType(self.arg.copy_unify(other.arg), self.ret.copy_unify(other.ret))

    def _substitute(self, substitution):
        return FuncType(
            self.arg.substitute(substitution), self.ret.substitute(substitution)
        )

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

    def _unify(self, other: Type):
        if not other.implements(self):
            raise TypeError()

    def _copy_unify(self, other: Type) -> Type:
        if not other.implements(self):
            raise TypeError(self, other)
        return other

    def __str__(self):
        return f"{self.fully_qualified_name}"

    def __hash__(self):
        return id(self)
