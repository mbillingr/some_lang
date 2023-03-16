from __future__ import annotations
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


@dataclasses.dataclass
class TypeFunction(Type):
    env: MetaEnv
    var: str
    bdy: ast.TypeExpression

    def __str__(self):
        return f"(λ ({self.var}) {self.bdy})"


def check(ty: Type, expr: ast.Expression, tenv: TEnv):
    match ty, expr:
        case FunctionType(targ, tret), ast.Function(var, body):
            tenv_ = tenv.extend(var, targ)
            check(tret, body, tenv_)
        case TypeFunction(env, var, bdy), _:
            texp = infer(expr, tenv)
            ty.apply(texp)
        case _, _:
            texp = infer(expr, tenv)
            if texp != ty:
                raise TypeError(texp, ty)


def infer(expr: ast.Expression, tenv: TEnv) -> Type:
    match expr:
        case ast.Annotation(exp, txp):
            typ = eval_type(txp, empty_menv())
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
            tenv_ = tenv.extend(var, targ)
            return infer(body, tenv_)
        case ast.Application(fun, arg):
            tfun = infer(fun, tenv)
            assert isinstance(tfun, FunctionType)
            check(tfun.targ, arg, tenv)
            return tfun.tret
        case _:
            raise NotImplementedError(expr)


MetaEnv: TypeAlias = Bindings[Type]


def eval_type(texp: ast.TypeExpression, menv: MetaEnv) -> Type:
    match texp:
        case ast.TypeLiteral("Bool"):
            return Bool()
        case ast.TypeLiteral("Int"):
            return Int()
        case ast.TypeLiteral(x):
            return menv.get(x)
        case ast.FuncType(lhs, rhs):
            targ = eval_type(lhs, menv)
            tret = eval_type(rhs, menv)
            return FunctionType(targ, tret)
        case ast.TypeFunction(var, _, body):
            return TypeFunction(menv, var, body)
        case ast.TypeApplication(tfun, targ):
            tfun = eval_type(tfun, menv)
            targ = eval_type(targ, menv)
            with tfun.env.child_scope() as menv_:
                menv_.insert(tfun.var, targ)
                return eval_type(tfun.bdy, menv_)
        case _:
            raise NotImplementedError(texp)


def empty_tenv() -> TEnv:
    return Bindings()


def empty_menv() -> MetaEnv:
    return Bindings()
