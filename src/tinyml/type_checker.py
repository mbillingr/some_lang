import abc
import dataclasses
from typing import Any, TypeAlias

from tinyml import abstract_syntax as ast
from tinyml.bindings import Bindings


class Type(abc.ABC):
    def __str__(self):
        return f"{self.__class__.__name__}"


TEnv: TypeAlias = Bindings[Type]


@dataclasses.dataclass
class Bool(Type):
    pass


@dataclasses.dataclass
class Int(Type):
    pass


@dataclasses.dataclass
class FunctionType(Type):
    targ: Type
    tret: Type

    def __str__(self):
        return f"({self.targ} -> {self.tret})"


def check(ty: Type, expr: ast.Expression, tenv: TEnv):
    match ty, expr:
        case FunctionType(targ, tret), ast.Function(var, body):
            with tenv.child_scope() as tenv_:
                tenv.insert(var, targ)
                check(tret, body, tenv_)
        case _, _:
            texp = infer(expr, tenv)
            if texp != ty:
                raise TypeError(texp, ty)


def infer(expr: ast.Expression, tenv: TEnv) -> Type:
    match expr:
        case ast.Annotation(exp, txp):
            typ = eval_type(txp, tenv)
            check(typ, exp, tenv)
            return typ
        case ast.Literal(bool()):
            return Bool()
        case ast.Literal(int()):
            return Int()
        case ast.Reference(var):
            return tenv.get(var)
        case ast.Application(ast.Function(var, body), arg):
            # this special case allows inferring the type of direct application
            targ = infer(arg, tenv)
            with tenv.child_scope() as tenv_:
                tenv.insert(var, targ)
                return infer(body, tenv_)
        case ast.Application(fun, arg):
            tfun = infer(fun, tenv)
            assert isinstance(tfun, FunctionType)
            check(tfun.targ, arg, tenv)
            return tfun.tret
        case _:
            raise NotImplementedError(expr)


def eval_type(texp: ast.TypeExpression, tenv: TEnv) -> Type:
    match texp:
        case ast.TypeLiteral("Bool"):
            return Bool()
        case ast.TypeLiteral("Int"):
            return Int()
        case ast.FuncType(lhs, rhs):
            targ = eval_type(lhs, tenv)
            tret = eval_type(rhs, tenv)
            return FunctionType(targ, tret)
        case _:
            raise NotImplementedError(texp)


def empty_tenv() -> TEnv:
    return Bindings()
