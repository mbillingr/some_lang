from __future__ import annotations

import abc
import dataclasses
import typing
from typing import Generic, TypeVar, Optional

from biunification.type_checker import VTypeHead, UTypeHead, Value, Use


T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Assoc(abc.ABC, Generic[T]):
    @abc.abstractmethod
    def __getitem__(self, item: str):
        pass

    @abc.abstractmethod
    def items(self) -> typing.Iterator[tuple[str, T]]:
        pass

    @abc.abstractmethod
    def substitute(self, x_old, x_new) -> Assoc[T]:
        pass


@dataclasses.dataclass(frozen=True)
class AssocEmpty(Assoc[T]):
    def __getitem__(self, item: str):
        raise KeyError(item)

    def items(self) -> typing.Iterator[tuple[str, T]]:
        return iter(())

    def substitute(self, x_old, x_new) -> Assoc[T]:
        return self


@dataclasses.dataclass(frozen=True)
class AssocItem(Assoc[T]):
    key: str
    val: T
    next: Assoc[T]

    def __getitem__(self, item: str):
        if self.key == item:
            return self.val
        return self.next[item]

    def items(self) -> typing.Iterator[tuple[str, T]]:
        yield self.key, self.val
        yield from self.next.items()

    def substitute(self, x_old, x_new) -> Assoc[T]:
        return AssocItem(
            self.key,
            x_new if self.val == x_old else self.val,
            self.next.substitute(x_old, x_new),
        )


@dataclasses.dataclass(frozen=True)
class VNever(VTypeHead):
    """A type that cannot be used. Maybe it has no value?"""
    def check(self, _):
        raise TypeError("Unusable type")


@dataclasses.dataclass(frozen=True)
class VBool(VTypeHead):
    pass


@dataclasses.dataclass(frozen=True)
class UBool(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VBool):
            raise TypeError(self, val)
        return []


@dataclasses.dataclass(frozen=True)
class VInt(VTypeHead):
    pass


@dataclasses.dataclass(frozen=True)
class UInt(UTypeHead):
    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VInt):
            raise TypeError(self, val)
        return []


@dataclasses.dataclass(frozen=True)
class VObj(VTypeHead):
    fields: Assoc[Value]

    @staticmethod
    def from_dict(d: dict[str, Value]) -> VObj:
        fields = AssocEmpty()
        for k, v in d.items():
            fields = AssocItem(k, v, fields)
        return VObj(fields)

    def substitute(self, t_old, t_new):
        return VObj(self.fields.substitute(t_old, t_new))


@dataclasses.dataclass(frozen=True)
class UObj(UTypeHead):
    field: str
    use: Use

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VObj):
            raise TypeError(self, val)
        try:
            return [(val.fields[self.field], self.use)]
        except KeyError:
            raise TypeError("Missing Field", self.field) from None

    def substitute(self, t_old, t_new):
        return UObj(self.field, t_new if self.use == t_old else t_new)


@dataclasses.dataclass(frozen=True)
class VCase(VTypeHead):
    tag: str
    typ: Value

    def substitute(self, t_old, t_new):
        return VCase(self.tag, t_new if self.typ == t_old else self.typ)


@dataclasses.dataclass(frozen=True)
class UCase(UTypeHead):
    cases: Assoc[Use]

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VCase):
            raise TypeError(self, val)
        try:
            return [(val.typ, self.cases[val.tag])]
        except KeyError:
            raise TypeError("Unhandled Case", val.tag) from None

    @staticmethod
    def from_dict(d: dict[str, Use]) -> UCase:
        cases = AssocEmpty()
        for k, v in d.items():
            cases = AssocItem(k, v, cases)
        return UCase(cases)

    def substitute(self, t_old, t_new):
        return UCase(self.cases.substitute(t_old, t_new))


@dataclasses.dataclass(frozen=True)
class VFunc(VTypeHead):
    arg: Use
    ret: Value

    def substitute(self, t_old, t_new):
        return VFunc(
            t_new if self.arg == t_old else self.arg,
            t_new if self.ret == t_old else self.ret,
        )


@dataclasses.dataclass(frozen=True)
class UFunc(UTypeHead):
    arg: Value
    ret: Use

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        if not isinstance(val, VFunc):
            raise TypeError(self, val)
        return [(val.ret, self.ret), (self.arg, val.arg)]

    def substitute(self, t_old, t_new):
        return UFunc(
            t_new if self.arg == t_old else self.arg,
            t_new if self.ret == t_old else self.ret,
        )


@dataclasses.dataclass(frozen=True)
class VRef(VTypeHead):
    write: Optional[Use]
    read: Optional[Value]

    def substitute(self, t_old, t_new) -> VTypeHead:
        return VRef(
            t_new if self.write == t_old else self.write,
            t_new if self.read == t_old else self.read,
        )


@dataclasses.dataclass(frozen=True)
class URef(UTypeHead):
    write: Optional[Value]
    read: Optional[Use]

    def check(self, val: VTypeHead) -> list[tuple[Value, Use]]:
        out = []
        if not isinstance(val, VRef):
            raise TypeError(self, val)
        if self.read is not None:
            if val.read is not None:
                out.append((val.read, self.read))
            else:
                raise TypeError("Reference is not readable")
        if self.write is not None:
            if val.write is not None:
                out.append((self.write, val.write))
            else:
                raise TypeError("Reference is not writable")
        return out

    def substitute(self, t_old, t_new) -> UTypeHead:
        return URef(
            t_new if self.write == t_old else self.write,
            t_new if self.read == t_old else self.read,
        )
