import abc
import dataclasses
from typing import Any


class AstNode(abc.ABC):
    pass


class Expression(AstNode):
    pass


class Statement(AstNode):
    pass


class Pattern(AstNode):
    pass


@dataclasses.dataclass
class Program(AstNode):
    exp: Expression


class Symbol(str, AstNode):
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
class Let(Expression):
    var: Identifier
    val: Expression
    bdy: Expression


@dataclasses.dataclass
class MatchArm(AstNode):
    pat: Pattern
    bdy: Expression


@dataclasses.dataclass
class Function(Expression):
    patterns: list[MatchArm]


@dataclasses.dataclass
class Application(Expression):
    fun: Expression
    arg: Expression


@dataclasses.dataclass
class BindingPattern(Pattern):
    name: Symbol


@dataclasses.dataclass
class LiteralPattern(Pattern):
    value: Any