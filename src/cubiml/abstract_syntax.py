import abc
import dataclasses
from typing import Any

import typing


class AstNode(abc.ABC):
    pass


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


Op = typing.Literal["+", "-", "*", "/", "<", "<=", ">=", ">", "==", "!=", "::"]
OpType = typing.Literal["any", "int", "bool"]


@dataclasses.dataclass(frozen=True)
class BinOp(Expression):
    lval: Expression
    rval: Expression
    opty: tuple[OpType, OpType, OpType]
    rtor: Op


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
class MatchArm(AstNode):
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
class FuncDef(AstNode):
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
class Script(AstNode):
    statements: list[ToplevelItem]


TRUE = Literal(True)
FALSE = Literal(False)


def free_vars(expr: Expression) -> typing.Iterator[str]:
    match expr:
        case Literal(_):
            return
        case Reference(var):
            yield var
        case BinOp(a, b, _, _):
            yield from free_vars(a)
            yield from free_vars(b)
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
        case Record(fields):
            yield from (free_vars(f[1]) for f in fields)
        case FieldAccess(_, rec):
            yield from free_vars(rec)
        case _:
            raise NotImplementedError(expr)


def visit(expr: Expression, visitor):
    try:
        _visit(expr, visitor)
    except StopIteration:
        pass


def _visit(node: AstNode, visitor):
    visitor(node)
    match node:
        case Script(statements):
            for stmt in statements:
                _visit(stmt, visitor)
        case DefineLet(_, expr):
            _visit(expr, visitor)
        case DefineLetRec(defs):
            for d in defs:
                _visit(d, visitor)
        case Literal(_):
            return
        case Reference(_):
            return
        case Function(_, body):
            _visit(body, visitor)
        case Application(fun, arg):
            _visit(fun, visitor)
            _visit(arg, visitor)
        case Conditional(a, b, c):
            _visit(a, visitor)
            _visit(b, visitor)
            _visit(c, visitor)
        case Record(fields):
            for f in fields:
                _visit(f[1], visitor)
        case FieldAccess(_, rec):
            _visit(rec, visitor)
        case _:
            raise NotImplementedError(node)


class FunctionFreeVars:
    def __init__(self, node: AstNode):
        self.vars = {}
        visit(node, self)

    def __call__(self, node: AstNode):
        match node:
            case Function(var, body):
                fvs = free_vars(node)
                self.vars[id(node)] = fvs
            case _:
                pass
