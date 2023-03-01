from cubiml import abstract_syntax as ast
from cubiml.tokenizer import (
    TokenStream,
    TokenKind,
    Span,
    UnexpectedEnd,
    UnexpectedToken,
)

infix_binding_power = {
    "if": (4, 3),
    "+": (5, 6),
    "-": (5, 6),
    "*": (7, 8),
    "/": (7, 8),
    "**": (14, 13),
}

prefix_binding_power = {
    "lambda": (None, 1),
    "~": (None, 9),
}

postfix_binding_power = {
    "!": (11, None),
    "(": (15, None),  # function call
}

op_types = {
    "+": ("int", "int", "int"),
    "-": ("int", "int", "int"),
    "*": ("int", "int", "int"),
    "/": ("int", "int", "int"),
    "~": ("bool", "bool"),
    "**": ("int", "int", "int"),
    "!": ("int", "int"),
}


def parse_toplevel(ts: TokenStream) -> ast.Script:
    expr = parse_expr(ts)

    extra_token = ts.get_next()
    if extra_token != ts.EOF:
        raise UnexpectedToken(extra_token)

    return ast.Script([expr])


def parse_block_expr(ts) -> ast.Expression:
    expect_token(ts, TokenKind.BEGIN_BLOCK)
    expr = parse_expr(ts)
    while True:
        match ts.peek():
            case ts.EOF:
                raise UnexpectedEnd()
            case _, TokenKind.END_BLOCK, _:
                ts.get_next()
                return expr
            case _:
                expect_token(ts, TokenKind.SEP_BLOCK)
                expr = parse_expr(ts)


def parse_expr(ts: TokenStream, min_bp: int = 0) -> ast.Expression:
    match ts.peek():
        case op, _, _ if op in prefix_binding_power:
            rbp, _ = prefix_binding_power[op]
            lhs = parse_prefix_operator(rbp, ts)
        case _:
            lhs = parse_atom(ts)

    while True:
        match ts.peek():
            case op, _, _ if op in postfix_binding_power:
                lbp, _ = postfix_binding_power[op]
                if lbp < min_bp:
                    break
                lhs = parse_postfix_operator(lhs, ts)
            case op, _, _ if op in infix_binding_power:
                lbp, rbp = infix_binding_power[op]
                if lbp < min_bp:
                    break
                lhs = parse_infix_operator(lhs, rbp, ts)
            case _:
                break

    return lhs


def parse_atom(ts: TokenStream):
    match ts.get_next():
        case ts.EOF:
            raise UnexpectedEnd()
        case val, TokenKind.LITERAL_BOOL | TokenKind.LITERAL_INT, span:
            return spanned(span, ast.Literal(val))
        case name, TokenKind.IDENTIFIER, span:
            return spanned(span, ast.Reference(name))
        case "(", _, span:
            inner = parse_expr(ts)
            _, _, sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), inner)
        case "if", _, span:
            cond = parse_expr(ts)
            lhs = parse_block_expr(ts)
            optional_token(ts, TokenKind.SEP_BLOCK)
            expect_tokens(ts, "else")
            rhs = parse_block_expr(ts)
            return spanned(span.merge(get_span(rhs)), ast.Conditional(cond, lhs, rhs))
        case token:
            raise UnexpectedToken(token)


def parse_prefix_operator(rbp, ts):
    match ts.get_next():
        case op, TokenKind.OPERATOR, span if op in prefix_binding_power:
            rhs = parse_expr(ts, rbp)
            return spanned(
                make_operator_span(span, get_span(rhs)),
                ast.UnaryOp(rhs, op_types[op], op),
            )
        case "lambda", _, span:
            expect_token(ts, "(")
            var, vspan = parse_identifier(ts)
            expect_token(ts, ")")
            body = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(body)), ast.Function(var, body))
        case token:
            raise UnexpectedToken(token)


def parse_infix_operator(lhs, rbp, ts):
    match ts.get_next():
        case "if", _, _:
            cond = parse_expr(ts)
            expect_token(ts, "else")
            rhs = parse_expr(ts, rbp)
            return spanned(
                get_span(lhs).merge(get_span(rhs)), ast.Conditional(cond, lhs, rhs)
            )
        case op, TokenKind.OPERATOR, span:
            rhs = parse_expr(ts, rbp)

            return spanned(
                make_operator_span(span, get_span(lhs), get_span(rhs)),
                ast.BinOp(lhs, rhs, op_types[op], op),
            )
        case token:
            raise UnexpectedToken(token)


def parse_postfix_operator(lhs, ts):
    match ts.get_next():
        case "(", _, span:
            arg = parse_expr(ts)
            _, _, sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), ast.Application(lhs, arg))
        case op, TokenKind.OPERATOR, span:
            return spanned(
                make_operator_span(span, get_span(lhs)),
                ast.UnaryOp(lhs, op_types[op], op),
            )
        case token:
            raise UnexpectedToken(token)


def parse_identifier(ts):
    tok, _, span = expect_token(ts, TokenKind.IDENTIFIER)
    return tok, span


def expect_tokens(ts, *expect):
    return [expect_token(ts, ex) for ex in expect]


def expect_token(ts, expect):
    match ts.get_next():
        case ts.EOF:
            raise UnexpectedEnd()
        case tok, kind, span:
            if isinstance(expect, TokenKind):
                if kind != expect:
                    raise UnexpectedToken((kind, tok, span))
            elif tok != expect:
                raise UnexpectedToken((kind, tok, span))

            return tok, kind, span


def optional_token(ts, expect):
    match ts.peek():
        case tok, kind, span:
            if isinstance(expect, TokenKind):
                if kind == expect:
                    ts.get_next()
            elif tok == expect:
                ts.get_next()

            return span
        case _:
            pass


def make_operator_span(rator: Span, *rands: Span):
    return None


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x) -> Span:
    return Span("", 0, 0)
