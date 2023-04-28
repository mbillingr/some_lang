import abc
import dataclasses
from typing import Optional

from eopl_explicit_refs.abstract_syntax import Symbol


class Type(abc.ABC):
    pass


@dataclasses.dataclass
class BoolType(Type):
    def __str__(self):
        return "Bool"


@dataclasses.dataclass
class IntType(Type):
    def __str__(self):
        return "Int"


@dataclasses.dataclass
class BoxType(Type):
    item_t: Type

    def __str__(self):
        return f"@{self.item_t}"


@dataclasses.dataclass
class ListType(Type):
    item_t: Type

    def __str__(self):
        return f"[{self.item_t}]"


@dataclasses.dataclass
class RecordType(Type):
    name: Optional[Symbol]
    fields: dict[Symbol, Type]

    def __str__(self):
        return f"[{', '.join(f'{n}: {t}' for n, t in self.fields.items())}]"


@dataclasses.dataclass
class FuncType(Type):
    arg: Type
    ret: Type

    def __str__(self):
        return f"{self.arg}->{self.ret}"
