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
class NopStatement(Statement):
    pass


@dataclasses.dataclass
class ExprStmt(Statement):
    expr: Expression


@dataclasses.dataclass
class Assignment(Statement):
    lhs: Expression
    rhs: Expression


@dataclasses.dataclass
class IfStatement(Statement):
    condition: Expression
    consequence: Statement
    alternative: Statement


@dataclasses.dataclass
class BlockStatement(Statement):
    fst: Statement
    snd: Statement


@dataclasses.dataclass
class BlockExpression(Expression):
    pre: Statement
    exp: Expression


@dataclasses.dataclass
class Identifier(Expression):
    name: str


@dataclasses.dataclass
class Literal(Expression):
    val: Any


@dataclasses.dataclass
class BinOp(Expression):
    lhs: Expression
    rhs: Expression
    op: str


@dataclasses.dataclass
class NewRef(Expression):
    val: Expression


@dataclasses.dataclass
class DeRef(Expression):
    ref: Expression


@dataclasses.dataclass
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression


@dataclasses.dataclass
class Let(Expression):
    var: Identifier
    val: Expression
    bdy: Expression


@dataclasses.dataclass
class EmptyList(Expression):
    pass


@dataclasses.dataclass
class ListCons(Expression):
    car: Expression
    cdr: Expression


@dataclasses.dataclass
class MatchArm(AstNode):
    pats: list[Pattern]
    body: Expression


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


@dataclasses.dataclass
class ListConsPattern(Pattern):
    car: Pattern
    cdr: Pattern


def stmt_to_expr(stmt: Statement) -> Expression:
    match stmt:
        case ExprStmt(x):
            return x
        case IfStatement(a, b, c):
            return Conditional(a, stmt_to_expr(b), stmt_to_expr(c))
        case _:
            raise TypeError(f"Can't convert {type(stmt).__name__} to expression")
