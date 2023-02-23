from cubiml import abstract_syntax as ast
from cubiml.tokenizer import PeekableTokenStream, TokenKind, Span


infix_binding_power = {
    "+": (1, 2),
    "-": (1, 2),
    "*": (3, 4),
    "/": (3, 4),
    "**": (10, 9),
}

prefix_binding_power = {
    "~": (None, 5),
}

postfix_binding_power = {
    "!": (7, None),
    "(": (11, None),  # function call
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
        try:
            token = ts.peek()
        except StopIteration:
            break

        match token:
            case ts.EOF:
                break
            case op, TokenKind.OPERATOR, _:
                pass
            case _, tok, _:
                raise UnexpectedToken(tok)

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
            sp2 = expect_token(")", ts)
            return spanned(span.merge(sp2), inner)
        # prefix operator
        case op, TokenKind.OPERATOR, span:
            _, rbp = prefix_binding_power[op]
            rhs = parse_expr(ts, rbp)
            return spanned(
                make_operator_span(span, get_span(rhs)),
                ast.UnaryOp(rhs, op_types[op], op),
            )
        case _, expect, _:
            raise NotImplementedError(expect)


def parse_infix_operator(lhs, rbp, ts):
    op, _, span = next(ts)

    rhs = parse_expr(ts, rbp)

    return spanned(
        make_operator_span(span, get_span(lhs), get_span(rhs)),
        ast.BinOp(lhs, rhs, op_types[op], op),
    )


def parse_postfix_operator(lhs, ts):
    match next(ts):
        case "(", _, span:
            arg = parse_expr(ts)
            sp2 = expect_token(")", ts)
            return spanned(span.merge(sp2), ast.Application(lhs, arg))
        case op, TokenKind.OPERATOR, span:
            return spanned(
                make_operator_span(span, get_span(lhs)),
                ast.UnaryOp(lhs, op_types[op], op),
            )
        case _, tok, _:
            raise NotImplementedError(tok)


def expect_token(expect, ts):
    s, _, span = next(ts)
    if s != expect:
        raise UnexpectedToken(s)
    return span


def make_operator_span(rator: Span, *rands: Span):
    return None


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x):
    return None
