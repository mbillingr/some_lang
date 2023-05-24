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
    def __init__(self, ty: Type):
        self.ty = ty
        self.instantiations = []
        self.current_instantiation = None

    @contextlib.contextmanager
    def instantiation(self, track=True):
        if self.current_instantiation:
            yield self.current_instantiation
        else:
            try:
                i = self._instantiate(track)
                self.current_instantiation = i
                yield i
            finally:
                self.current_instantiation = None

    def _instantiate(self, track):
        tvars = {}

        def handle_tvar(t: TypeVar, tvars):
            try:
                return tvars[t]
            except KeyError:
                fresh_var = TypeVar(t.name, t.constraints)
                tvars[t] = fresh_var
                return fresh_var

        ty = self._substitute(self.ty, tvars, handle_tvar=handle_tvar)

        if track:
            n = len(self.instantiations)
            self.instantiations.append((ty, tvars))
            return ty, n
        else:
            return ty, None

    def concrete_instantiations(self):
        def substitution(t: TypeVar, subs):
            if subs[t].type is None:
                raise TypeError(f"{t} was never substituted")
            return subs[t].type

        assert self.current_instantiation is None

        for n, (ty, tvars) in enumerate(self.instantiations):
            i = self._substitute(self.ty, tvars, handle_tvar=substitution)
            self.current_instantiation = i, n
            yield self.current_instantiation
            self.current_instantiation = None

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
                    TypeSchema._substitute(arg, subs, handle_tvar),
                    TypeSchema._substitute(ret, subs, handle_tvar),
                )
            case ListType(item):
                return ListType(TypeSchema._substitute(item, subs, handle_tvar))
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

    def _unify(self, other: Type):
        if self is not other:
            raise TypeError(self, other)

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

    def __str__(self):
        return f"@{self.item_t}"


@dataclasses.dataclass(frozen=True)
class ListType(Type):
    item_t: Type

    def _unify(self, other: Type):
        if not isinstance(other, ListType):
            raise TypeError()
        else:
            self.item_t.unify(other.item_t)

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

    def __str__(self):
        return f"{self.fully_qualified_name}"

    def __hash__(self):
        return id(self)
