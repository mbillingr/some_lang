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


def test_records():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert engine.types[
        type_checker.check_expr(ast.Record([]), env, engine)
    ] == type_heads.VObj({})

    assert engine.types[
        type_checker.check_expr(ast.Record([("x", ast.TRUE)]), env, engine)
    ] == type_heads.VObj({"x": type_checker.Value(1)})

    with pytest.raises(type_checker.RepeatedFieldNameError):
        type_checker.check_expr(
            ast.Record([("x", ast.TRUE), ("x", ast.TRUE)]), env, engine
        )


def test_cases():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert engine.types[
        type_checker.check_expr(ast.Case("Foo", ast.FALSE), env, engine)
    ] == type_heads.VCase("Foo", type_checker.Value(0))


def test_conditional():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.Conditional(ast.TRUE, ast.Record([]), ast.Record([])), env, engine
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError):
        type_checker.check_expr(
            ast.Conditional(ast.Record([]), ast.Record([]), ast.Record([])), env, engine
        )


def test_field_access():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.FieldAccess("x", ast.Record([("x", ast.TRUE)])), env, engine
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError):
        type_checker.check_expr(
            ast.FieldAccess("x", ast.Record([("y", ast.TRUE)])), env, engine
        )


def test_match():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.Match(
                    ast.Case("A", ast.TRUE),
                    [ast.MatchArm("A", "x", ast.Reference("x"))],
                ),
                env,
                engine,
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError, match="Unhandled Case"):
        type_checker.check_expr(ast.Match(ast.Case("A", ast.TRUE), []), env, engine)

    with pytest.raises(type_checker.RepeatedCaseError):
        type_checker.check_expr(
            ast.Match(
                ast.Case("A", ast.TRUE),
                [
                    ast.MatchArm("A", "x", ast.Reference("x")),
                    ast.MatchArm("A", "x", ast.Reference("x")),
                ],
            ),
            env,
            engine,
        )

    with pytest.raises(TypeError, match="Bool"):
        type_checker.check_expr(ast.Match(ast.TRUE, []), env, engine)
