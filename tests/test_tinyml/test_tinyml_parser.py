import pytest

from tinyml import tokenizer, abstract_syntax as ast, parser
from parsing.scanner_generator import Span


def test_parse_expr_atom():
    assert parse_expr("0") == ast.Literal(0)
    assert parse_expr("true") == ast.Literal(True)
    assert parse_expr("false") == ast.Literal(False)
    assert parse_expr("foo") == ast.Reference("foo")


def test_parse_annotation():
    assert parse_expr("the Int 0") == ast.Annotation(
        ast.Literal(0), ast.TypeLiteral("Int")
    )
    assert parse_expr("the Bool true") == ast.Annotation(
        ast.Literal(True), ast.TypeLiteral("Bool")
    )

    assert parse_expr("the int -> int -> int 0") == ast.Annotation(
        ast.Literal(0),
        ast.FuncType(
            ast.TypeLiteral("int"),
            ast.FuncType(ast.TypeLiteral("int"), ast.TypeLiteral("int")),
        ),
    )


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
    assert parse_expr("foo 0") == ast.Application(ast.Reference("foo"), ast.Literal(0))
    assert parse_expr("f 1 2") == ast.Application(
        ast.Application(ast.Reference("f"), ast.Literal(1)), ast.Literal(2)
    )
    assert parse_expr("f (g 0)") == ast.Application(
        ast.Reference("f"), ast.Application(ast.Reference("g"), ast.Literal(0))
    )


def test_parse_expr_ternary():
    assert parse_expr("if x then a else b") == ast.Conditional(
        ast.Reference("x"), ast.Reference("a"), ast.Reference("b")
    )
    assert parse_expr("if x then a else if y then b else c") == ast.Conditional(
        ast.Reference("x"),
        ast.Reference("a"),
        ast.Conditional(ast.Reference("y"), ast.Reference("b"), ast.Reference("c")),
    )


def test_parse_lambda():
    assert parse_expr("fn x => x") == ast.Function("x", ast.Reference("x"))


def test_parse_lambda_with_lower_precedence_than_application():
    assert parse_expr("fn f => f 0") == ast.Function(
        "f", ast.Application(ast.Reference("f"), ast.Literal(0))
    )
    assert parse_expr("(fn x => x) 0") == ast.Application(
        ast.Function("x", ast.Reference("x")), ast.Literal(0)
    )


def test_let_expression():
    src = "let x = 0 in x"
    assert parse_expr(src) == ast.Let("x", ast.Literal(0), ast.Reference("x"))


def test_bind_lambda():
    src = "let inc = fn x => x + 1 in inc 41"
    assert parse_expr(src) == ast.Let(
        "inc",
        ast.Function("x", binop("+", ast.Reference("x"), ast.Literal(1))),
        ast.Application(ast.Reference("inc"), ast.Literal(41)),
    )


def test_parse_expr_incomplete():
    with pytest.raises(tokenizer.UnexpectedEnd):
        parse_expr("0 +")


def test_parse_toplevel_expr():
    assert parse_top("0") == ast.Script([ast.Literal(0)])


def test_parse_toplevel_let():
    assert parse_top("let x = 0") == ast.Script([ast.DefineLet("x", ast.Literal(0))])


def test_parse_toplevel_function():
    assert parse_top("func foo x = 0") == ast.Script(
        [ast.DefineLetRec([ast.FuncDef("foo", ast.Function("x", ast.Literal(0)))])]
    )


def test_parse_toplevel_functions_are_mutually_recursive():
    assert parse_top("func foo x = 0; func bar y = 1") == ast.Script(
        [
            ast.DefineLetRec(
                [
                    ast.FuncDef("foo", ast.Function("x", ast.Literal(0))),
                    ast.FuncDef("bar", ast.Function("y", ast.Literal(1))),
                ]
            )
        ]
    )


def parse_expr(src):
    token_stream = tokenizer.default_tokenizer(src, implicit_block=False)
    return parser.parse_expr(token_stream)


def parse_top(src):
    token_stream = tokenizer.default_tokenizer(src, implicit_block=True)
    return parser.parse_toplevel(token_stream)


def token(txt: str, t: tokenizer.TokenKind) -> tokenizer.Token:
    return txt, t, Span("", 0, 0)


def binop(op, lhs, rhs, ty=("int", "int", "int")):
    return ast.BinOp(lhs, rhs, ty, op)
