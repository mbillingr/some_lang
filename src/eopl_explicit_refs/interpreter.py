import dataclasses
from typing import Any, Callable

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


def init_env() -> Env:
    return EmptyEnv()


def analyze_program(pgm: ast.Program) -> Any:
    match pgm:
        case ast.Program(exp):
            prog = analyze_expr(exp, init_env())

            def program(store):
                store.clear()
                return prog(store)

            return program


def analyze_stmt(stmt: ast.Statement, env: Env) -> Callable:
    match stmt:
        case ast.ExprStmt(expr):
            # Let's assume expressions have no side effects, so we can just ignore them here
            _ = analyze_expr(expr, env)  # we still check if the expression is valid

            def nop(store):
                pass

            return nop

        case ast.Assignment(lhs, rhs):
            lhs_ = analyze_expr(lhs, env)
            rhs_ = analyze_expr(rhs, env)
            return lambda store: store.setref(lhs_(store), rhs_(store))
        case _:
            raise NotImplementedError(stmt)


def analyze_expr(exp: ast.Expression, env: Env) -> Callable:
    match exp:
        case ast.Literal(val):
            return lambda _: val
        case ast.Identifier(name):
            idx = env.lookup(name)
            return lambda store: store.get(idx)
        case ast.NewRef(val):
            val_ = analyze_expr(val, env)
            return lambda store: store.newref(val_(store))
        case ast.DeRef(ref):
            ref_ = analyze_expr(ref, env)
            return lambda store: store.deref(ref_(store))
        case ast.Sequence(stmt, expr):
            stmt_ = analyze_stmt(stmt, env)
            expr_ = analyze_expr(expr, env)

            def sequence(store):
                stmt_(store)
                return expr_(store)

            return sequence

        case ast.Let(var, val, bdy):
            val_ = analyze_expr(val, env)
            bdy_env = env.extend(var)
            bdy_ = analyze_expr(bdy, bdy_env)

            def let(store):
                store.push(val_(store))
                result = bdy_(store)
                store.pop()
                return result

            return let

        case ast.Function(var, bdy):
            bdy_env = env.extend(var)
            bdy_ = analyze_expr(bdy, bdy_env)

            def the_function(store):
                return Closure(store, bdy_)

            return the_function

        case ast.Application(fun, arg):
            fun_ = analyze_expr(fun, env)
            arg_ = analyze_expr(arg, env)
            return lambda store: fun_(store).apply(arg_(store), store)
        case _:
            raise NotImplementedError(exp)


class Closure:
    def __init__(self, store, body):
        self.body = body
        self.saved_stack = store.stack

    def apply(self, arg, store):
        preserved_stack = store.stack
        store.stack = self.saved_stack
        try:
            store.push(arg)
            return self.body(store)
        finally:
            store.stack = preserved_stack