import abc
import dataclasses
from typing import Any

import typing


class ToplevelItem(abc.ABC):
    pass


class Expression(ToplevelItem):
    pass


@dataclasses.dataclass(frozen=True)
class Literal(Expression):
    val: Any


@dataclasses.dataclass(frozen=True)
class Reference(Expression):
    var: str


@dataclasses.dataclass(frozen=True)
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression


@dataclasses.dataclass(frozen=True)
class Record(Expression):
    fields: list[tuple[str, Expression]]


@dataclasses.dataclass(frozen=True)
class FieldAccess(Expression):
    field: str
    expr: Expression


@dataclasses.dataclass(frozen=True)
class Case(Expression):
    tag: str
    val: Expression


@dataclasses.dataclass(frozen=True)
class MatchArm:
    tag: str
    var: str
    bdy: Expression


@dataclasses.dataclass(frozen=True)
class Match(Expression):
    expr: Expression
    arms: list[MatchArm]


@dataclasses.dataclass(frozen=True)
class Function(Expression):
    var: str
    body: Expression


@dataclasses.dataclass(frozen=True)
class Application(Expression):
    fun: Expression
    arg: Expression


@dataclasses.dataclass(frozen=True)
class Let(Expression):
    var: str
    val: Expression
    body: Expression


@dataclasses.dataclass(frozen=True)
class FuncDef:
    name: str
    fun: Function


@dataclasses.dataclass(frozen=True)
class LetRec(Expression):
    bind: list[FuncDef]
    body: Expression


@dataclasses.dataclass(frozen=True)
class DefineLet(ToplevelItem):
    var: str
    val: Expression


@dataclasses.dataclass(frozen=True)
class DefineLetRec(ToplevelItem):
    bind: list[FuncDef]


@dataclasses.dataclass(frozen=True)
class Script:
    statements: list[ToplevelItem]


TRUE = Literal(True)
FALSE = Literal(False)


def free_vars(expr: Expression) -> typing.Iterator[str]:
    match expr:
        case Literal(_):
            return
        case Reference(var):
            yield var
        case Function(var, body):
            for fv in free_vars(body):
                if fv != var:
                    yield fv
        case Application(fun, arg):
            yield from free_vars(fun)
            yield from free_vars(arg)
        case Conditional(a, b, c):
            yield from free_vars(a)
            yield from free_vars(b)
            yield from free_vars(c)
        case _:
            raise NotImplementedError(expr)
