import pytest

import cubiml.tokenizer
from cubiml import tokenizer, scanner, abstract_syntax as ast, parser2


def test_parse_expr_atom():
    assert parse_expr("0") == ast.Literal(0)
    assert parse_expr("true") == ast.Literal(True)
    assert parse_expr("false") == ast.Literal(False)
    assert parse_expr("foo") == ast.Reference("foo")


def test_parse_expr_infix():
    assert parse_expr("0 + 0") == binop("+", ast.Literal(0), ast.Literal(0))


def test_parse_expr_binding_power():
    assert parse_expr("1 + 2 * 3 + 4") == binop(
        "+",
        binop("+", ast.Literal(1), binop("*", ast.Literal(2), ast.Literal(3))),
        ast.Literal(4),
    )


def test_parse_expr_right_associative():
    assert parse_expr("2 ** 3 ** 4") == binop(
        "**", ast.Literal(2), binop("**", ast.Literal(3), ast.Literal(4))
    )


def test_parse_expr_prefix_operator():
    assert parse_expr("~1") == ast.UnaryOp(ast.Literal(1), ("bool", "bool"), "~")


def test_parse_expr_postfix_operator():
    assert parse_expr("1!") == ast.UnaryOp(ast.Literal(1), ("int", "int"), "!")


def test_parse_expr_parens():
    assert parse_expr("(((1) * (2 + 3)))") == binop(
        "*", ast.Literal(1), binop("+", ast.Literal(2), ast.Literal(3))
    )


def test_parse_expr_call():
    assert parse_expr("foo(0)") == ast.Application(ast.Reference("foo"), ast.Literal(0))


def test_parse_expr_ternary():
    assert parse_expr("a if x else b if y else c") == ast.Conditional(
        ast.Reference("x"),
        ast.Reference("a"),
        ast.Conditional(ast.Reference("y"), ast.Reference("b"), ast.Reference("c")),
    )


def test_parse_regular_if():
    src = """
    if x:
       y
    else:
       n
    """
    assert parse_expr(src) == ast.Conditional(
        ast.Reference("x"), ast.Reference("y"), ast.Reference("n")
    )


@pytest.mark.parametrize("w1", ["", "  ", "    "])
@pytest.mark.parametrize("w2", [" ", "\n", "\n  ", "\n    "])
@pytest.mark.parametrize("w3", [" ", "\n", "\n  ", "\n    "])
def test_parse_expr_indented(w1, w2, w3):
    src = f"{w1}1{w2}+{w3}2"
    assert parse_expr(src) == binop("+", ast.Literal(1), ast.Literal(2))


def test_parse_expr_indented_in_block():
    src = f"""
    if true:
       1 +
         2
    else:
       3
    """
    assert parse_expr(src) == ast.Conditional(
        ast.Literal(True), binop("+", ast.Literal(1), ast.Literal(2)), ast.Literal(3)
    )


def test_invalid_dedent():
    src = f"""
    if true:
        1
      else:
        2
    """
    with pytest.raises(tokenizer.LayoutError):
        parse_expr(src)


def test_parse_expr_incomplete():
    with pytest.raises(cubiml.tokenizer.UnexpectedEnd):
        parse_expr("0 +")

    # with pytest.raises(parser2.UnexpectedToken):
    #    print(parse_expr("0 0"))


def parse_expr(src):
    token_stream = tokenizer.default_tokenizer(src)
    return parser2.parse_expr(token_stream)


def token(txt: str, t: tokenizer.TokenKind) -> tokenizer.Token:
    return txt, t, scanner.Span("", 0, 0)


def binop(op, lhs, rhs, ty=("int", "int", "int")):
    return ast.BinOp(lhs, rhs, ty, op)
