from cubiml import ast, interpreter, parser


def test_simple_exprs():
    assert interpreter.evaluate(ast.TRUE, {}) is True
    assert interpreter.evaluate(ast.FALSE, {}) is False
    assert interpreter.evaluate(ast.Reference("x"), {"x": 1}) == 1


def test_conditional():
    assert (
        interpreter.evaluate(
            ast.Conditional(ast.TRUE, ast.Reference("x"), ast.Reference("y")),
            {"x": 1, "y": 2},
        )
        == 1
    )
    assert (
        interpreter.evaluate(
            ast.Conditional(ast.FALSE, ast.Reference("x"), ast.Reference("y")),
            {"x": 1, "y": 2},
        )
        == 2
    )


def test_record():
    assert interpreter.evaluate(
        ast.Record({"a": ast.TRUE, "b": ast.Record({})}), {}
    ) == {
        "a": True,
        "b": {},
    }

    assert (
        interpreter.evaluate(ast.FieldAccess("a", ast.Reference("r")), {"r": {"a": 1}})
        == 1
    )

    assert (
        interpreter.evaluate(
            ast.FieldAccess(
                "b", ast.Record({"a": ast.FALSE, "b": ast.TRUE, "c": ast.FALSE})
            ),
            {},
        )
        == True
    )


def test_case_match():
    assert interpreter.evaluate(ast.Case("Foo", ast.TRUE), {}) == ("Foo", True)
    assert (
        interpreter.evaluate(
            ast.Match(
                ast.Case("Bar", ast.TRUE),
                [
                    ast.MatchArm("Foo", "x", ast.TRUE),
                    ast.MatchArm("Bar", "x", ast.FALSE),
                ],
            ),
            {},
        )
        is False
    )

    assert (
        interpreter.evaluate(
            ast.Match(
                ast.Case("Bar", ast.TRUE),
                [
                    ast.MatchArm("Foo", "x", ast.FALSE),
                    ast.MatchArm("Bar", "x", ast.Reference("x")),
                ],
            ),
            {},
        )
        is True
    )


def test_function():
    assert (
        interpreter.evaluate(
            ast.Application(ast.Function("x", ast.Reference("x")), ast.TRUE), {}
        )
        is True
    )
    assert (
        interpreter.evaluate(
            ast.Application(ast.Function("x", ast.Reference("y")), ast.TRUE), {"y": 0}
        )
        == 0
    )


def test_let():
    assert (
        interpreter.evaluate(ast.Let("x", ast.TRUE, ast.Reference("x")), {"x": 0})
        is True
    )


def test_letrec():
    expr = ast.LetRec(
        [
            ast.FuncDef(
                "foo",
                ast.Function(
                    "x",
                    ast.Conditional(
                        ast.Reference("x"),
                        ast.Application(ast.Reference("foo"), ast.FALSE),
                        ast.Record({}),
                    ),
                ),
            )
        ],
        ast.Application(ast.Reference("foo"), ast.TRUE),
    )
    assert interpreter.evaluate(expr, {}) == {}


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

even (`S `S `Z false)
"""
    exp = parser.parse_script(src)
    assert interpreter.run_script(exp, {}) is True
