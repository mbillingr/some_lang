import pytest

from cubiml import parser, ast


def test_parse_valid_identifiers():
    assert parser.ident.parse_string("x")[0] == "x"
    assert parser.ident.parse_string("abc")[0] == "abc"
    assert parser.ident.parse_string("foo_bar")[0] == "foo_bar"
    assert parser.ident.parse_string("ab34")[0] == "ab34"


def test_parse_boolean_literal():
    assert parser.expr.parse_string("true")[0] == ast.Boolean(True)
    assert parser.expr.parse_string("false")[0] == ast.Boolean(False)


def test_parse_variable_reference():
    assert parser.expr.parse_string("foo")[0] == ast.Reference("foo")


def test_parse_conditional():
    src = "if true then false else true"
    expect = ast.Conditional(ast.TRUE, ast.FALSE, ast.TRUE)
    assert parser.expr.parse_string(src)[0] == expect


def test_parse_records():
    assert parser.expr.parse_string("{}")[0] == ast.Record({})
    assert parser.expr.parse_string("{a=true}")[0] == ast.Record({"a": ast.TRUE})
    assert parser.expr.parse_string("{a=true; b=false}")[0] == ast.Record(
        {"a": ast.TRUE, "b": ast.FALSE}
    )
    assert parser.expr.parse_string("{a=true; b=false; foo={}}")[0] == ast.Record(
        {"a": ast.TRUE, "b": ast.FALSE, "foo": ast.Record({})}
    )


def test_parse_field_access():
    assert parser.expr.parse_string("{}.abc")[0] == ast.FieldAccess(
        "abc", ast.Record({})
    )
    assert parser.expr.parse_string("{}.x.y")[0] == ast.FieldAccess(
        "y", ast.FieldAccess("x", ast.Record({}))
    )


def test_parse_paren_exp():
    src = "(if true then {} else {}).x"
    assert parser.expr.parse_string(src)[0] == ast.FieldAccess(
        "x", ast.Conditional(ast.TRUE, ast.Record({}), ast.Record({}))
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


def test_get_expr_location():
    src = " if a then a else a"
    exp = parser.expr.parse_string(src)[0]

    assert parser.get_loc(exp) == 0
    assert parser.get_loc(exp.condition) == 3
    assert parser.get_loc(exp.consequence) == 10
    assert parser.get_loc(exp.alternative) == 17
