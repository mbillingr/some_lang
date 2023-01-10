import abc
import dataclasses
from typing import TypeAlias

from some_lang import ast
from some_lang.env import EmptyEnv, Env


Type: TypeAlias = ast.TypeExpression


def check_module(mod: ast.Module):
    env = EmptyEnv()

    for defn in mod.defs:
        func_type = ast.FunctionType(defn.arg, defn.res)

        local_env = env
        for pattern in defn.patterns:
            assert pattern.name == defn.name
            match pattern.pat:
                case ast.IntegerPattern():
                    if defn.arg != ast.IntegerType():
                        raise TypeError(f"Integer pattern cannot match {defn.arg}")
                case ast.BindingPattern(var):
                    local_env = local_env.extend(var, defn.arg)
                    check_expr(defn.res, pattern.exp, local_env)

        env = env.extend(defn.name, func_type)

    for stmt in mod.code:
        check_stmt(stmt, env)


def check_expr(t: Type, expr: ast.Expression, env: Env[Type]):
    if t != infer_expr(expr, env):
        raise TypeError(f"Expected {t}: {expr}")


def infer_expr(expr: ast.Expression, env: Env[Type]) -> Type:
    match expr:
        case ast.Integer():
            return ast.IntegerType()
        case ast.Reference(var):
            return env.apply(var)
        case ast.Application(rator, rand):
            rator_t = infer_expr(rator, env)

            if not isinstance(rator_t, ast.FunctionType):
                raise TypeError(f"Expected Function: {rator}")

            check_expr(rator_t.arg, rand, env)
            return rator_t.res
        case _:
            raise NotImplementedError(expr)


def check_stmt(stmt: ast.Statement, env: Env[Type]):
    match stmt:
        case ast.PrintStatement(exp):
            # print accepts any type; just make sure exp does not contain type errors
            infer_expr(exp, env)
        case _:
            raise NotImplementedError(stmt)
