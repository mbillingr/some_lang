import pytest

from cubiml import type_checker, abstract_syntax as ast, type_heads, parser


def test_literal():
    ctx = type_checker.Context.default()
    assert (
        ctx.engine.types[type_checker.check_expr(ast.Literal(False), ctx)]
        == type_heads.VBool()
    )


def test_variable():
    ctx = type_checker.Context.default()
    ctx.bindings.insert("x", type_checker.Value(123))

    assert type_checker.check_expr(ast.Reference("x"), ctx) == 123

    with pytest.raises(type_checker.UnboundError):
        type_checker.check_expr(ast.Reference("y"), ctx)


def test_records():
    ctx = type_checker.Context.default()

    assert ctx.engine.types[
        type_checker.check_expr(ast.Record([]), ctx)
    ] == type_heads.VObj.from_dict({})

    assert ctx.engine.types[
        type_checker.check_expr(ast.Record([("x", ast.TRUE)]), ctx)
    ] == type_heads.VObj.from_dict({"x": type_checker.Value(1)})

    with pytest.raises(type_checker.RepeatedFieldNameError):
        type_checker.check_expr(ast.Record([("x", ast.TRUE), ("x", ast.TRUE)]), ctx)


def test_cases():
    ctx = type_checker.Context.default()

    assert ctx.engine.types[
        type_checker.check_expr(ast.Case("Foo", ast.FALSE), ctx)
    ] == type_heads.VCase("Foo", type_checker.Value(0))


def test_conditional():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
            type_checker.check_expr(
                ast.Conditional(ast.TRUE, ast.Record([]), ast.Record([])), ctx
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError):
        type_checker.check_expr(
            ast.Conditional(ast.Record([]), ast.Record([]), ast.Record([])), ctx
        )


def test_field_access():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
            type_checker.check_expr(
                ast.FieldAccess("x", ast.Record([("x", ast.TRUE)])), ctx
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError):
        type_checker.check_expr(
            ast.FieldAccess("x", ast.Record([("y", ast.TRUE)])), ctx
        )


def test_match():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
            type_checker.check_expr(
                ast.Match(
                    ast.Case("A", ast.TRUE),
                    [ast.MatchArm("A", "x", ast.Reference("x"))],
                ),
                ctx,
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError, match="Unhandled Case"):
        type_checker.check_expr(ast.Match(ast.Case("A", ast.TRUE), []), ctx)

    with pytest.raises(type_checker.RepeatedCaseError):
        type_checker.check_expr(
            ast.Match(
                ast.Case("A", ast.TRUE),
                [
                    ast.MatchArm("A", "x", ast.Reference("x")),
                    ast.MatchArm("A", "x", ast.Reference("x")),
                ],
            ),
            ctx,
        )

    with pytest.raises(TypeError, match="Bool"):
        type_checker.check_expr(ast.Match(ast.TRUE, []), ctx)


def test_funcdef():
    ctx = type_checker.Context.default()

    assert ctx.engine.types[
        type_checker.check_expr(ast.Function("x", ast.TRUE), ctx)
    ] == type_heads.VFunc(type_checker.Use(0), type_checker.Value(1))


def test_funcall():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
            type_checker.check_expr(
                ast.Application(ast.Function("x", ast.TRUE), ast.FALSE), ctx
            )
        ]
        == "Var"
    )

    with pytest.raises(TypeError, match="Bool"):
        type_checker.check_expr(ast.Application(ast.TRUE, ast.FALSE), ctx)


def test_let():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
            type_checker.check_expr(ast.Let("x", ast.TRUE, ast.Reference("x")), ctx)
        ]
        == type_heads.VBool()
    )


def test_letrec():
    ctx = type_checker.Context.default()

    assert (
        ctx.engine.types[
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
                ctx,
            )
        ]
        == "Var"
    )


def test_toplevel_expr():
    ctx = type_checker.Context.default()

    type_checker.check_toplevel(ast.FALSE, ctx)  # should not raise

    with pytest.raises(TypeError):
        type_checker.check_toplevel(ast.Application(ast.FALSE, ast.FALSE), ctx)


def test_toplevel_let():
    ctx = type_checker.Context.default()

    type_checker.check_toplevel(ast.DefineLet("x", ast.TRUE), ctx)

    assert (
        ctx.engine.types[type_checker.check_expr(ast.Reference("x"), ctx)]
        == type_heads.VBool()
    )


def test_toplevel_letrec():
    ctx = type_checker.Context.default()

    type_checker.check_toplevel(
        ast.DefineLetRec([ast.FuncDef("foo", ast.Function("x", ast.TRUE))]), ctx
    )

    assert (
        ctx.engine.types[
            type_checker.check_expr(
                ast.Application(ast.Reference("foo"), ast.Reference("foo")), ctx
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
