import pytest

from cubiml import parser, ast


def test_parse_valid_identifiers():
    assert parser.ident.parse_string("x")[0] == "x"
    assert parser.ident.parse_string("abc")[0] == "abc"
    assert parser.ident.parse_string("foo_bar")[0] == "foo_bar"
    assert parser.ident.parse_string("ab34")[0] == "ab34"


def test_parse_boolean_literal():
    assert parser.expr.parse_string("true")[0] == ast.Literal(True)
    assert parser.expr.parse_string("false")[0] == ast.Literal(False)


def test_parse_variable_reference():
    assert parser.expr.parse_string("foo")[0] == ast.Reference("foo")


def test_parse_conditional():
    src = "if true then false else true"
    expect = ast.Conditional(ast.TRUE, ast.FALSE, ast.TRUE)
    assert parser.expr.parse_string(src)[0] == expect


def test_parse_records():
    assert parser.expr.parse_string("{}")[0] == ast.Record([])
    assert parser.expr.parse_string("{a=true}")[0] == ast.Record([("a", ast.TRUE)])
    assert parser.expr.parse_string("{a=true; b=false}")[0] == ast.Record(
        [("a", ast.TRUE), ("b", ast.FALSE)]
    )
    assert parser.expr.parse_string("{a=true; b=false; foo={}}")[0] == ast.Record(
        [("a", ast.TRUE), ("b", ast.FALSE), ("foo", ast.Record([]))]
    )


def test_parse_field_access():
    assert parser.expr.parse_string("{}.abc")[0] == ast.FieldAccess(
        "abc", ast.Record([])
    )
    assert parser.expr.parse_string("{}.x.y")[0] == ast.FieldAccess(
        "y", ast.FieldAccess("x", ast.Record([]))
    )


def test_parse_paren_exp():
    src = "(if true then {} else {}).x"
    assert parser.expr.parse_string(src)[0] == ast.FieldAccess(
        "x", ast.Conditional(ast.TRUE, ast.Record([]), ast.Record([]))
    )


def test_parse_function():
    assert parser.expr.parse_string("fun x -> x")[0] == ast.Function(
        "x", ast.Reference("x")
    )


def test_parse_application():
    assert parser.expr.parse_string("a b")[0] == ast.Application(
        ast.Reference("a"), ast.Reference("b")
    )
    assert parser.expr.parse_string("a b c")[0] == ast.Application(
        ast.Application(ast.Reference("a"), ast.Reference("b")), ast.Reference("c")
    )
    assert (
        parser.expr.parse_string("a b c d")[0]
        == parser.expr.parse_string("(((a b) c) d)")[0]
    )


def test_parse_let():
    assert parser.expr.parse_string("let x = y in x")[0] == ast.Let(
        "x", ast.Reference("y"), ast.Reference("x")
    )


def test_parse_letrec():
    assert parser.expr.parse_string("let rec hang = fun x -> hang x in hang")[
        0
    ] == ast.LetRec(
        [
            ast.FuncDef(
                "hang",
                ast.Function(
                    "x", ast.Application(ast.Reference("hang"), ast.Reference("x"))
                ),
            )
        ],
        ast.Reference("hang"),
    )

    assert parser.expr.parse_string(
        "let rec a = fun x -> b x and b = fun y -> a y in a"
    )[0] == ast.LetRec(
        [
            ast.FuncDef(
                "a",
                ast.Function(
                    "x", ast.Application(ast.Reference("b"), ast.Reference("x"))
                ),
            ),
            ast.FuncDef(
                "b",
                ast.Function(
                    "y", ast.Application(ast.Reference("a"), ast.Reference("y"))
                ),
            ),
        ],
        ast.Reference("a"),
    )


def test_parse_case():
    assert parser.expr.parse_string("`Foo true")[0] == ast.Case("Foo", ast.TRUE)


def test_parse_match():
    assert parser.expr.parse_string("match x with | `A y -> y | `B z -> z")[
        0
    ] == ast.Match(
        ast.Reference("x"),
        [
            ast.MatchArm("A", "y", ast.Reference("y")),
            ast.MatchArm("B", "z", ast.Reference("z")),
        ],
    )


def test_parse_toplevel():
    assert parser.script.parse_string("let x = y;")[0] == ast.Script(
        [ast.DefineLet("x", ast.Reference("y"))]
    )
    assert parser.script.parse_string("let rec x = fun x -> x;")[0] == ast.Script(
        [ast.DefineLetRec([ast.FuncDef("x", ast.Function("x", ast.Reference("x")))])]
    )
    assert parser.script.parse_string("let rec x = fun x -> x and y = fun y -> y;")[
        0
    ] == ast.Script(
        [
            ast.DefineLetRec(
                [
                    ast.FuncDef("x", ast.Function("x", ast.Reference("x"))),
                    ast.FuncDef("y", ast.Function("y", ast.Reference("y"))),
                ]
            )
        ]
    )


def test_get_expr_location():
    src = " if a then a else a"
    exp = parser.expr.parse_string(src)[0]

    assert parser.get_loc(exp) == 0
    assert parser.get_loc(exp.condition) == 3
    assert parser.get_loc(exp.consequence) == 10
    assert parser.get_loc(exp.alternative) == 17
