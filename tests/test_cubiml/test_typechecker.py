import pytest

from cubiml import type_checker, ast, type_heads


def test_literal():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[type_checker.check_expr(ast.Literal(False), env, engine)]
        == type_heads.VBool()
    )


def test_variable():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()
    env.insert("x", type_checker.Value(123))

    assert type_checker.check_expr(ast.Reference("x"), env, engine) == 123

    with pytest.raises(type_checker.UnboundError):
        type_checker.check_expr(ast.Reference("y"), env, engine)
