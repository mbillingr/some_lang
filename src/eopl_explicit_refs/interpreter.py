import dataclasses
from typing import Any, Callable

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


UNDEFINED = object()


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
        case ast.BinOp(lhs, rhs, op):
            lhs_ = analyze_expr(lhs, env)
            rhs_ = analyze_expr(rhs, env)
            match op:
                case "+":
                    return lambda store: lhs_(store) + rhs_(store)
                case "-":
                    return lambda store: lhs_(store) - rhs_(store)
                case "*":
                    return lambda store: lhs_(store) * rhs_(store)
                case "/":
                    return lambda store: lhs_(store) / rhs_(store)
                case _:
                    raise NotImplementedError(op)
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
            let_env = env.extend(var)
            val_ = analyze_expr(val, let_env)
            bdy_ = analyze_expr(bdy, let_env)

            def let(store):
                store.push(UNDEFINED)
                store.set(0, val_(store))
                result = bdy_(store)
                store.pop()
                return result

            return let

        case ast.Function(arms):
            match_bodies = []
            for arm in arms:
                matcher, bound = analyze_pattern(arm.pat)
                bdy_env = env.extend(*bound)
                bdy_ = analyze_expr(arm.bdy, bdy_env)
                match_bodies.append((matcher, bdy_))

            def the_function(store):
                return Closure(store, match_bodies)

            return the_function

        case ast.Application(fun, arg):
            fun_ = analyze_expr(fun, env)
            arg_ = analyze_expr(arg, env)
            return lambda store: fun_(store).apply(arg_(store), store)
        case _:
            raise NotImplementedError(exp)


def analyze_pattern(pat: ast.Pattern) -> tuple[Callable, list[ast.Symbol]]:
    match pat:
        case ast.BindingPattern(name):
            return lambda val: (val,), [name]
        case ast.LiteralPattern(value):

            def literal_matcher(val):
                if value == val:
                    return (value,)
                raise MatcherError(value, val)

            return literal_matcher, []
        case _:
            raise NotImplementedError(pat)


class MatcherError(Exception):
    pass


class Closure:
    def __init__(self, store, match_bodies):
        self.match_bodies = match_bodies
        self.saved_stack = store.stack

    def apply(self, arg, store):
        preserved_stack = store.stack
        store.stack = self.saved_stack
        for matcher, body in self.match_bodies:
            try:
                bindings = matcher(arg)
            except MatcherError:
                continue
            try:
                store.push(*bindings)
                return body(store)
            finally:
                store.stack = preserved_stack
        raise MatcherError("no pattern matched")
