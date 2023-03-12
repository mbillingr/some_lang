import dataclasses
from typing import Any, Type, TypeAlias

from tinyml import abstract_syntax as ast
from tinyml.bindings import Bindings


TEnv: TypeAlias = Bindings[Type]


@dataclasses.dataclass
class FunctionType:
    targ: Type
    tret: Type


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
        case ast.Literal(x):
            return type(x)
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
        case ast.TypeLiteral("int"):
            return int
        case _:
            raise NotImplementedError(texp)


def empty_tenv() -> TEnv:
    return Bindings()
