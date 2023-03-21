import abc
import dataclasses
from typing import Any


class Expression(abc.ABC):
    pass


@dataclasses.dataclass
class Program:
    exp: Expression


@dataclasses.dataclass
class Identifier:
    name: str


@dataclasses.dataclass
class Literal(Expression):
    val: Any


@dataclasses.dataclass
class NewRef(Expression):
    val: Expression


@dataclasses.dataclass
class DeRef(Expression):
    ref: Expression


@dataclasses.dataclass
class SetRef(Expression):
    ref: Expression
    val: Expression
