from cubiml import abstract_syntax as ast
from cubiml.tokenizer import TokenStream, TokenKind, Span, UnexpectedEnd, UnexpectedToken

infix_binding_power = {
    "if": (2, 1),
    "+": (3, 4),
    "-": (3, 4),
    "*": (5, 6),
    "/": (5, 6),
    "**": (12, 11),
}

prefix_binding_power = {
    "~": (None, 7),
}

postfix_binding_power = {
    "!": (9, None),
    "(": (13, None),  # function call
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

    extra_token = next(ts)
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
                expr = parse_expr(ts)


def parse_expr(ts: TokenStream, min_bp: int = 0) -> ast.Expression:
    lhs = parse_atom(ts)

    while True:
        match ts.peek():
            case ts.EOF:
                break
            case op, _, _:
                pass

        if op in postfix_binding_power:
            lbp, _ = postfix_binding_power[op]
            if lbp < min_bp:
                break
            lhs = parse_postfix_operator(lhs, ts)
        elif op in infix_binding_power:
            lbp, rbp = infix_binding_power[op]
            if lbp < min_bp:
                break
            lhs = parse_infix_operator(lhs, rbp, ts)
        else:
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
            sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), inner)
        # prefix operator
        case op, TokenKind.OPERATOR, span:
            _, rbp = prefix_binding_power[op]
            rhs = parse_expr(ts, rbp)
            return spanned(
                make_operator_span(span, get_span(rhs)),
                ast.UnaryOp(rhs, op_types[op], op),
            )
        case "if", _, span:
            cond = parse_expr(ts)
            lhs = parse_block_expr(ts)
            expect_tokens(ts, "else")
            rhs = parse_block_expr(ts)
            return spanned(span.merge(get_span(rhs)), ast.Conditional(cond, lhs, rhs))
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
            sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), ast.Application(lhs, arg))
        case op, TokenKind.OPERATOR, span:
            return spanned(
                make_operator_span(span, get_span(lhs)),
                ast.UnaryOp(lhs, op_types[op], op),
            )
        case token:
            raise UnexpectedToken(token)


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

            return span


def make_operator_span(rator: Span, *rands: Span):
    return None


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x) -> Span:
    return Span("", 0, 0)
