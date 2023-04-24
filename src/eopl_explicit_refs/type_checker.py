from typing import TypeAlias

from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs.generic_environment import Env, EmptyEnv
from eopl_explicit_refs.types import Type
from eopl_explicit_refs import types as t

TEnv: TypeAlias = Env[Type]


def init_env() -> TEnv:
    return EmptyEnv()


def check_program(pgm: ast.Program) -> ast.Program:
    match pgm:
        case ast.Program(exp):
            prog, _ = infer_expr(exp, init_env())
            return ast.Program(prog)


def check_expr(exp: ast.Expression, typ: Type, env: TEnv) -> ast.Expression:
    match typ, exp:
        case _, ast.Literal(val):
            mapping = {bool: t.BoolType, int: t.IntType}
            if mapping[type(val)] != type(typ):
                raise TypeError(exp, typ)
            return exp
        case t.IntType(), ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), env)
            rhs = check_expr(rhs, t.IntType(), env)
            return ast.BinOp(lhs, rhs, op)
        case other:
            raise NotImplementedError(other)


def infer_expr(exp: ast.Expression, env: TEnv) -> (ast.Expression, Type):
    match exp:
        case ast.Literal(val):
            mapping = {bool: t.BoolType, int: t.IntType}
            return exp, mapping[type(val)]()
        case ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), env)
            rhs = check_expr(rhs, t.IntType(), env)
            return ast.BinOp(lhs, rhs, op), t.IntType
        case _:
            raise NotImplementedError(exp)


def eval_type(tx: ast.Type) -> Type:
    raise NotImplementedError()
