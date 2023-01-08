from __future__ import annotations
from typing import Callable, Union

from some_lang import ast
from some_lang.env import Env

Func = Callable[["Value"], "Value"]
Value = Union[int, Func]


def evaluate(expr: ast.Expr, env: Env) -> Value:
    match expr:
        case ast.Integer(val):
            return val
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


def make_function(var: str, body: ast.Expr, captured_env: Env) -> Func:
    def _func(arg: Value) -> Value:
        local_env = captured_env.extend(var, arg)
        return evaluate(body, local_env)
    return _func
