from typing import Any

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs.store import initialize_store, newref, deref, setref
from eopl_explicit_refs import abstract_syntax as ast


def init_env() -> Env:
    return EmptyEnv()


def value_of_program(pgm: ast.Program) -> Any:
    initialize_store()
    match pgm:
        case ast.Program(exp):
            return value_of(exp, init_env())


def value_of(exp: ast.Expression, env: Env) -> Any:
    match exp:
        case ast.Literal(val):
            return val
        case ast.NewRef(val):
            return newref(value_of(val, env))
        case ast.DeRef(ref):
            return deref(value_of(ref, env))
        case ast.SetRef(ref, val):
            ref_ = value_of(ref, env)
            val_ = value_of(val, env)
            setref(ref_, val_)
            return val_
        case _:
            raise NotImplementedError(exp)
