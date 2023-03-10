from __future__ import annotations
from typing import Callable, Union

from some_lang import ast
from some_lang.env import Env, EmptyEnv

Func = Callable[["Value"], "Value"]
Value = Union[int, Func]


def run_module(mod: ast.Module, env: Env[Value] = EmptyEnv()) -> Env[Value]:
    for defn in mod.defs:

        def func(x, defn=defn, local_env=env):
            for pattern in defn.patterns:
                assert pattern.name == defn.name
                match pattern.pat:
                    case ast.IntegerPattern(val) if val == x:
                        return evaluate(pattern.exp, local_env)
                    case ast.BindingPattern(var):
                        local_env = local_env.extend(var, x)
                        return evaluate(pattern.exp, local_env)

        env = env.extend(defn.name, func)

    for stmt in mod.code:
        match stmt:
            case ast.PrintStatement(expr):
                print(evaluate(expr, env))
    return env


def evaluate(expr: ast.Expression, env: Env[Value]) -> Value:
    match expr:
        case ast.Boolean(val) | ast.Integer(val):
            return val
        case ast.Reference(var):
            match env.apply(var):
                case None:
                    raise NameError(var)
                case v:
                    assert v is not None  # for mypy
                    return v
        case ast.Lambda(var, bdy):
            return make_function(var, bdy, env)
        case ast.Application(rator, rand):
            func = evaluate(rator, env)
            if not callable(func):
                raise TypeError(f"not callable: {func}")
            arg = evaluate(rand, env)
            return func(arg)
        case _:
            raise ValueError(f"invalid expression: {expr}")


def make_function(var: str, body: ast.Expression, captured_env: Env) -> Func:
    def _func(arg: Value) -> Value:
        local_env = captured_env.extend(var, arg)
        return evaluate(body, local_env)

    return _func
