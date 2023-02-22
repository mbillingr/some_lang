import pytest

from cubiml import tokenizer, scanner, abstract_syntax as ast, parser2


def test_parse_expr_atom():
    assert parse_expr("0") == ast.Literal(0)


def test_parse_expr_infix():
    assert parse_expr("0 + 0") == ast.BinOp(
        ast.Literal(0), ast.Literal(0), ("int", "int", "int"), "+"
    )


def test_parse_expr_binding_power():
    assert parse_expr("1 + 2 * 3 + 4") == ast.BinOp(
        ast.BinOp(
            ast.Literal(1),
            ast.BinOp(
                ast.Literal(2),
                ast.Literal(3),
                ("int", "int", "int"),
                "*",
            ),
            ("int", "int", "int"),
            "+",
        ),
        ast.Literal(4),
        ("int", "int", "int"),
        "+",
    )


def test_parse_expr_right_associative():
    assert parse_expr("2 ** 3 ** 4") == ast.BinOp(
        ast.Literal(2),
        ast.BinOp(ast.Literal(3), ast.Literal(4), ("int", "int", "int"), "**"),
        ("int", "int", "int"),
        "**",
    )


def test_parse_expr_prefix_operator():
    assert parse_expr("~1") == ast.UnaryOp(ast.Literal(1), ("bool", "bool"), "~")


def test_parse_expr_incomplete():
    with pytest.raises(parser2.UnexpectedEnd):
        parse_expr("0 +")

    with pytest.raises(parser2.UnexpectedToken):
        parse_expr("0 0")


def parse_expr(src):
    token_stream = tokenizer.default_tokenizer(src)
    return parser2.parse_expr(token_stream)


def token(txt: str, t: tokenizer.TokenKind) -> tokenizer.Token:
    return txt, t, scanner.Span("", 0, 0)
