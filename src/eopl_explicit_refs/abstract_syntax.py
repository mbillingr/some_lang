import abc
import dataclasses
from typing import Any


class Expression(abc.ABC):
    pass


class Statement(abc.ABC):
    pass


@dataclasses.dataclass
class Program:
    exp: Expression


class Symbol(str):
    pass


@dataclasses.dataclass
class ExprStmt(Statement):
    expr: Expression


@dataclasses.dataclass
class Assignment(Statement):
    lhs: Expression
    rhs: Expression


@dataclasses.dataclass
class Sequence(Expression):
    pre: Statement
    exp: Expression


@dataclasses.dataclass
class Identifier(Expression):
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


@dataclasses.dataclass
class Let(Expression):
    var: Identifier
    val: Expression
    bdy: Expression
