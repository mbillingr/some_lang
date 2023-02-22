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


def transform_errors(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except StopIteration:
            raise UnexpectedEnd()

    return wrapped


def parse_expr(ts: PeekableTokenStream, min_bp: int = 0) -> ast.Expression:
    lhs = parse_atom(ts)

    while True:
        try:
            token = ts.peek()
        except StopIteration:
            break

        match token:
            case op, TokenKind.OPERATOR, span:
                pass
            case _, tok, _:
                raise UnexpectedToken(tok)

        if op in postfix_binding_power:
            lbp, _ = postfix_binding_power[op]
            if lbp < min_bp:
                break
            next(ts)
            lhs = spanned([span, get_span(lhs)], ast.UnaryOp(lhs, op_types[op], op))
        elif op in infix_binding_power:
            lbp, rbp = infix_binding_power[op]
            if lbp < min_bp:
                break

            next(ts)
            rhs = parse_expr(ts, rbp)

            lhs = spanned(
                [get_span(lhs), span, get_span(rhs)],
                ast.BinOp(lhs, rhs, op_types[op], op),
            )
        else:
            break

    return lhs


@transform_errors
def parse_atom(ts):
    match next(ts):
        case val, TokenKind.LITERAL_INT, span:
            return spanned(span, ast.Literal(val))
        case "(", _, span:
            lhs = parse_expr(ts)
            s, _, sp2 = next(ts)
            if s != ")":
                raise UnexpectedToken(s)
            return spanned(Span(span.src, span.start, sp2.end), lhs)
        case op, TokenKind.OPERATOR, span:
            _, rbp = prefix_binding_power[op]
            rhs = parse_expr(ts, rbp)
            return spanned([span, get_span(rhs)], ast.UnaryOp(rhs, op_types[op], op))
        case _, tok, _:
            raise NotImplementedError(tok)


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x):
    return None
