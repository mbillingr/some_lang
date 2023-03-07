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


def test_parse_whitespace_before_eof():
    src = """
do:
    0
   
    """
    assert parse_expr(src) == ast.Literal(0)


def test_parse_lambda():
    assert parse_expr("lambda x = x") == ast.Function("x", ast.Reference("x"))


def test_parse_lambda_with_lower_precedence_than_application():
    assert parse_expr("lambda f = f 0") == ast.Function(
        "f", ast.Application(ast.Reference("f"), ast.Literal(0))
    )
    assert parse_expr("(lambda x = x) 0") == ast.Application(
        ast.Function("x", ast.Reference("x")), ast.Literal(0)
    )


def test_parse_let_in_block():
    src = """
do:
    let x = 0
    x
    """
    assert parse_expr(src) == ast.Let("x", ast.Literal(0), ast.Reference("x"))


def test_let_expression():
    src = """
let x = 0:
    x
    """
    assert parse_expr(src) == ast.Let("x", ast.Literal(0), ast.Reference("x"))


def test_bind_lambda():
    src = """
let inc = lambda x = x + 1:
    inc 41
    """
    assert parse_expr(src) == ast.Let(
        "inc",
        ast.Function("x", binop("+", ast.Reference("x"), ast.Literal(1))),
        ast.Application(ast.Reference("inc"), ast.Literal(41)),
    )


@pytest.mark.parametrize("w2", [" ", "\n", "\n  ", "\n    "])
@pytest.mark.parametrize("w3", [" ", "\n", "\n  ", "\n    "])
def test_parse_expr_indented(w2, w3):
    src = f"1{w2}+{w3}2"
    assert parse_expr(src) == binop("+", ast.Literal(1), ast.Literal(2))


def test_parse_expr_indented_in_block():
    src = f"""
do:
  1 +
    2
    """
    assert parse_expr(src) == binop("+", ast.Literal(1), ast.Literal(2)), ast.Literal(3)


def test_parse_expr_multiple_dedent():
    src = f"""
do:
  1
  do:
    do:
      2
    do:
      3
  4
    """
    assert parse_expr(src) == ast.Sequence(
        ast.Literal(1),
        ast.Sequence(ast.Sequence(ast.Literal(2), ast.Literal(3)), ast.Literal(4)),
    )


def test_parse_expr_incomplete():
    with pytest.raises(cubiml.tokenizer.UnexpectedEnd):
        parse_expr("0 +")

    # with pytest.raises(parser2.UnexpectedToken):
    #    print(parse_expr("0 0"))


def test_parse_toplevel_expr():
    assert parse_top("0") == ast.Script([ast.Literal(0)])


def test_parse_toplevel_let():
    assert parse_top("let x = 0") == ast.Script([ast.DefineLet("x", ast.Literal(0))])


def test_parse_toplevel_function():
    assert parse_top("func foo x = 0") == ast.Script(
        [ast.DefineLetRec([ast.FuncDef("foo", ast.Function("x", ast.Literal(0)))])]
    )


def test_parse_toplevel_functions_are_mutually_recursive():
    assert parse_top("func foo x = 0\nfunc bar y = 1") == ast.Script(
        [
            ast.DefineLetRec(
                [
                    ast.FuncDef("foo", ast.Function("x", ast.Literal(0))),
                    ast.FuncDef("bar", ast.Function("y", ast.Literal(1))),
                ]
            )
        ]
    )


def test_parse_nested_toplevel_blocks():
    assert parse_top(":\n  let x = 1\nlet y = 0") == ast.Script(
        [ast.DefineLet("x", ast.Literal(1)), ast.DefineLet("y", ast.Literal(0))]
    )


def parse_expr(src):
    token_stream = tokenizer.default_tokenizer(src, implicit_block=False)
    return parser2.parse_expr(token_stream)


def parse_top(src):
    token_stream = tokenizer.default_tokenizer(src, implicit_block=True)
    return parser2.parse_toplevel(token_stream)


def token(txt: str, t: tokenizer.TokenKind) -> tokenizer.Token:
    return txt, t, scanner.Span("", 0, 0)


def binop(op, lhs, rhs, ty=("int", "int", "int")):
    return ast.BinOp(lhs, rhs, ty, op)
