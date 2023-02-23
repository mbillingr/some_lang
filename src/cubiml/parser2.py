from cubiml import abstract_syntax as ast
from cubiml.tokenizer import PeekableTokenStream, TokenKind, Span


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


class ParseError(Exception):
    pass


class UnexpectedEnd(ParseError):
    pass


class UnexpectedToken(ParseError):
    pass


def parse_expr(ts: PeekableTokenStream, min_bp: int = 0) -> ast.Expression:
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


def parse_atom(ts):
    match next(ts):
        case ts.EOF:
            raise UnexpectedEnd()
        case val, TokenKind.LITERAL_INT, span:
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
            expect_tokens(ts, ":", TokenKind.INDENT)
            lhs = parse_expr(ts)
            expect_tokens(ts, TokenKind.DEDENT, "else", ":", TokenKind.INDENT)
            rhs = parse_expr(ts)
            expect_tokens(ts, TokenKind.DEDENT)
            return spanned(span.merge(get_span(rhs)), ast.Conditional(cond, lhs, rhs))
        case tok, kind, _:
            raise UnexpectedToken(kind, tok)


def parse_infix_operator(lhs, rbp, ts):
    match next(ts):
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
        case tok, kind, _:
            raise UnexpectedToken(kind, tok)


def parse_postfix_operator(lhs, ts):
    match next(ts):
        case "(", _, span:
            arg = parse_expr(ts)
            sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), ast.Application(lhs, arg))
        case op, TokenKind.OPERATOR, span:
            return spanned(
                make_operator_span(span, get_span(lhs)),
                ast.UnaryOp(lhs, op_types[op], op),
            )
        case tok, kind, _:
            raise UnexpectedToken(kind, tok)


def expect_tokens(ts, *expect):
    return [expect_token(ts, ex) for ex in expect]


def expect_token(ts, expect):
    tok, kind, span = next(ts)

    if isinstance(expect, TokenKind):
        if kind != expect:
            raise UnexpectedToken(tok, kind)
    elif tok != expect:
        raise UnexpectedToken(tok, kind)

    return span


def make_operator_span(rator: Span, *rands: Span):
    return None


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x) -> Span:
    return Span("", 0, 0)
