from __future__ import annotations

import abc
import dataclasses


@dataclasses.dataclass
class Module:
    defs: list[Definition]
    code: list[Statement]


@dataclasses.dataclass
class Definition:
    name: str
    arg: TypeExpression
    res: TypeExpression
    patterns: list[DefinitionPattern]


@dataclasses.dataclass
class DefinitionPattern:
    name: str
    pat: Pattern
    exp: Expression


class TypeExpression(abc.ABC):
    pass


@dataclasses.dataclass
class BooleanType(TypeExpression):
    pass


@dataclasses.dataclass
class IntegerType(TypeExpression):
    pass


@dataclasses.dataclass
class FunctionType(TypeExpression):
    arg: TypeExpression
    res: TypeExpression


class Pattern(abc.ABC):
    pass


@dataclasses.dataclass
class IntegerPattern(Pattern):
    val: int


@dataclasses.dataclass
class BindingPattern(Pattern):
    var: str


class Statement(abc.ABC):
    pass


@dataclasses.dataclass
class PrintStatement(Statement):
    exp: Expression


class Expression(abc.ABC):
    pass


@dataclasses.dataclass
class Boolean(Expression):
    val: int


@dataclasses.dataclass
class Integer(Expression):
    val: int


@dataclasses.dataclass
class Reference(Expression):
    var: str


@dataclasses.dataclass
class Lambda(Expression):
    var: str
    bdy: Expression


@dataclasses.dataclass
class Application(Expression):
    rator: Expression
    rand: Expression
