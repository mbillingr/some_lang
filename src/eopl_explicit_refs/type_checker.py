from typing import TypeAlias

from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs.generic_environment import Env, EmptyEnv
from eopl_explicit_refs.type_impls import Type
from eopl_explicit_refs import type_impls as t

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
        case ast.Type(), _:
            raise TypeError("Unevaluated type passed to checker")

        case _, ast.Literal(val):
            mapping = {bool: t.BoolType, int: t.IntType}
            if mapping[type(val)] != type(typ):
                raise TypeError(exp, typ)
            return exp

        case _, ast.Identifier(name):
            actual_t = env.lookup(name)
            if actual_t != typ:
                raise TypeError(f"Expected a {typ} but {name} is a {actual_t}")
            return exp

        case t.IntType(), ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), env)
            rhs = check_expr(rhs, t.IntType(), env)
            return ast.BinOp(lhs, rhs, op)

        case t.ListType(_), ast.EmptyList():
            return exp

        case t.ListType(item_t), ast.BinOp(lhs, rhs, "::" as op):
            lhs = check_expr(lhs, item_t, env)
            rhs = check_expr(rhs, typ, env)
            return ast.BinOp(lhs, rhs, op)

        case t.FuncType(_, _), ast.Function(arms):
            return ast.Function(
                [
                    ast.MatchArm(arm.pats, check_matcharm(arm.pats, arm.body, typ, env))
                    for arm in arms
                ]
            )

        case typ, ast.Application(fun, arg):
            fun, fun_t = infer_expr(fun, env)
            match fun_t:
                case t.FuncType(arg_t, ret_t):
                    pass
                case _:
                    raise TypeError(f"Cannot call {fun_t}")
            if ret_t != typ:
                raise TypeError(f"Function returns {ret_t} but expected {typ}")
            return ast.Application(fun, check_expr(arg, arg_t, env))

        case _:
            actual_t = infer_expr(exp, env)
            if actual_t != typ:
                raise TypeError(actual_t, typ)


def infer_expr(exp: ast.Expression, env: TEnv) -> (ast.Expression, Type):
    match exp:
        case ast.Literal(val):
            mapping = {bool: t.BoolType, int: t.IntType}
            return exp, mapping[type(val)]()

        case ast.Identifier(name):
            return exp, env.lookup(name)

        case ast.EmptyList():
            raise TypeError("can't infer empty list type")

        case ast.TypeAnnotation(tx, expr):
            ty = eval_type(tx)
            expr = check_expr(expr, ty, env)
            return expr, ty

        case ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), env)
            rhs = check_expr(rhs, t.IntType(), env)
            return ast.BinOp(lhs, rhs, op), t.IntType

        case ast.BinOp(lhs, rhs, "::" as op):
            lhs, item_t = infer_expr(lhs, env)
            rhs = check_expr(rhs, t.ListType(item_t), env)
            return ast.BinOp(lhs, rhs, op), t.IntType

        case ast.Function(patterns):
            raise TypeError(
                "can't infer function signatures. Please provide a type hint."
            )

        case ast.Application(fun, arg):
            fun, fun_t = infer_expr(fun, env)
            match fun_t:
                case t.FuncType(arg_t, ret_t):
                    pass
                case _:
                    raise TypeError(f"Cannot call {fun_t}")
            return ast.Application(fun, check_expr(arg, arg_t, env)), ret_t

        case ast.Let(var, val, bdy, None):
            # without type declaration, the let variable can't be used in the val expression
            val, val_t = infer_expr(val, env)
            body, out_t = infer_expr(bdy, env.extend(var, val_t))
            return ast.Let(var, val, body, val_t), out_t

        case ast.Let(var, val, bdy, var_t):
            var_t = eval_type(var_t)
            let_env = env.extend(var, var_t)
            val = check_expr(val, var_t, let_env)
            body, out_t = infer_expr(bdy, let_env)
            return ast.Let(var, val, body, var_t), out_t

        case _:
            raise NotImplementedError(exp)


def check_matcharm(
    patterns: list[ast.Pattern], body: ast.Expression, typ: Type, env: TEnv
) -> ast.Expression:
    match typ, patterns:
        case t.FuncType(arg_t, res_t), [p0, *p_rest]:
            bindings = check_pattern(p0, arg_t, env)
            for name, ty in bindings.items():
                env = env.extend(name, ty)
            return check_matcharm(p_rest, body, res_t, env)
        case _, []:
            return check_expr(body, typ, env)
        case other:
            raise NotImplementedError(other)


def check_pattern(pat: ast.Pattern, typ: Type, env: TEnv) -> dict[str, Type]:
    match typ, pat:
        case _, ast.BindingPattern(name):
            return {name: typ}
        case _, ast.LiteralPattern(val):
            _ = check_expr(ast.Literal(val), typ, env)
            return {}
        case t.ListType(item_t), ast.ListConsPattern(car, cdr):
            return check_pattern(car, item_t, env) | check_pattern(cdr, typ, env)
        case t.ListType(_), ast.EmptyListPattern():
            return {}
        case other:
            raise NotImplementedError(other)


def eval_type(tx: ast.Type) -> Type:
    match tx:
        case ast.IntType:
            return t.IntType()
        case ast.ListType(item_t):
            return t.ListType(eval_type(item_t))
        case ast.FuncType(arg_t, ret_t):
            return t.FuncType(eval_type(arg_t), eval_type(ret_t))
        case _:
            raise NotImplementedError(tx)
