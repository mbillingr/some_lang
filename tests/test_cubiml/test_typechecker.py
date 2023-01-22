import pytest

from cubiml import type_checker, abstract_syntax as ast, type_heads, parser


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
    ] == type_heads.VObj.from_dict({})

    assert engine.types[
        type_checker.check_expr(ast.Record([("x", ast.TRUE)]), env, engine)
    ] == type_heads.VObj.from_dict({"x": type_checker.Value(1)})

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


def test_funcdef():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert engine.types[
        type_checker.check_expr(ast.Function("x", ast.TRUE), env, engine)
    ] == type_heads.VFunc(type_checker.Use(0), type_checker.Value(1))


def test_funcall():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.Application(ast.Function("x", ast.TRUE), ast.FALSE), env, engine
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError, match="Bool"):
        type_checker.check_expr(ast.Application(ast.TRUE, ast.FALSE), env, engine)


def test_let():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.Let("x", ast.TRUE, ast.Reference("x")), env, engine
            )
        ]
        == type_heads.VBool()
    )


def test_letrec():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    assert (
        engine.types[
            type_checker.check_expr(
                ast.LetRec(
                    [
                        ast.FuncDef(
                            "foo",
                            ast.Function(
                                "x",
                                ast.Application(
                                    ast.Reference("foo"), ast.Reference("x")
                                ),
                            ),
                        )
                    ],
                    ast.Application(ast.Reference("foo"), ast.TRUE),
                ),
                env,
                engine,
            )
        ]
        == "Var"
    )


def test_toplevel_expr():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    type_checker.check_toplevel(ast.FALSE, env, engine)  # should not raise

    with pytest.raises(TypeError):
        type_checker.check_toplevel(ast.Application(ast.FALSE, ast.FALSE), env, engine)


def test_toplevel_let():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    type_checker.check_toplevel(ast.DefineLet("x", ast.TRUE), env, engine)

    assert (
        engine.types[type_checker.check_expr(ast.Reference("x"), env, engine)]
        == type_heads.VBool()
    )


def test_toplevel_letrec():
    engine = type_checker.TypeCheckerCore()
    env = type_checker.Bindings()

    type_checker.check_toplevel(
        ast.DefineLetRec([ast.FuncDef("foo", ast.Function("x", ast.TRUE))]), env, engine
    )

    assert (
        engine.types[
            type_checker.check_expr(
                ast.Application(ast.Reference("foo"), ast.Reference("foo")), env, engine
            )
        ]
        == "Var"
    )


def test_script():
    src = """
let not = fun b -> if b then false else true;

let rec even = fun n -> 
        match n with
            | `Z x -> true
            | `S k -> odd k
    and odd = fun n ->
        match n with
            | `Z x -> false
            | `S k -> even k;

if not (odd (`S `S `Z {})) then `Ok {} else `Fail {}
"""
    script = parser.parse_script(src)
    type_checker.TypeChecker().check_script(script)


def test_script_rollback():
    tc = type_checker.TypeChecker()
    tc.check_script(parser.parse_script("let x = true"))

    try:
        tc.check_script(parser.parse_script("let y = true; let z = true true"))
    except Exception:
        pass

    tc.check_script(parser.parse_script("x"))  # should pass

    with pytest.raises(type_checker.UnboundError):
        tc.check_script(parser.parse_script("y"))
