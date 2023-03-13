from typing import Any, TypeAlias

from tinyml import abstract_syntax as ast
from tinyml.bindings import Bindings


Env: TypeAlias = Bindings[Any]


def analyze(expr: ast.Expression, env: Env):
    match expr:
        case ast.Annotation(x, _):
            return analyze(x, env)
        case ast.Literal(x):
            return lambda _: x
        case ast.Reference(var):
            idx = env.get(var)
            return lambda store: store[idx]
        case ast.Function(var, body):
            with env.child_scope() as env_:
                env_.insert(var, env_.depth())
                fun = analyze(body, env_)
                return lambda store: lambda arg: fun(store + (arg,))
        case ast.Application(fun, arg):
            f = analyze(fun, env)
            a = analyze(arg, env)
            return lambda store: f(store)(a(store))
        case _:
            raise NotImplementedError(expr)


def empty_env() -> Env:
    return Bindings()
