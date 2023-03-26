from typing import Any

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs.store import PythonStore as Store
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


def analyze_expr(exp: ast.Expression, env: Env) -> Any:
    match exp:
        case ast.Literal(val):
            return lambda _: val
        case ast.NewRef(val):
            val_ = analyze_expr(val, env)
            return lambda store: store.newref(val_(store))
        case ast.DeRef(ref):
            ref_ = analyze_expr(ref, env)
            return lambda store: store.deref(ref_(store))
        case ast.SetRef(ref, val):
            ref_ = analyze_expr(ref, env)
            val_ = analyze_expr(val, env)

            def set_ref(store):
                store.setref(ref_(store), val_(store))
                return Nothing()

            return set_ref
        case _:
            raise NotImplementedError(exp)


class Nothing:
    """No value - this is "returned" by statements and such."""

    def __str__(self):
        return "<nothing>"
