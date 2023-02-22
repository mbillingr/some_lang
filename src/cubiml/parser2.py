from cubiml import abstract_syntax as ast
from cubiml.tokenizer import PeekableTokenStream, TokenKind


infix_binding_power = {
    "+": (1, 2),
    "-": (1, 2),
    "*": (3, 4),
    "/": (3, 4),
}

op_types = {
    "+": ("int", "int", "int"),
    "-": ("int", "int", "int"),
    "*": ("int", "int", "int"),
    "/": ("int", "int", "int"),
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

        lbp, rbp = infix_binding_power[op]
        if lbp < min_bp:
            break

        next(ts)
        rhs = parse_expr(ts, rbp)

        lhs = spanned(
            [get_span(lhs), span, get_span(rhs)], ast.BinOp(lhs, rhs, op_types[op], op)
        )

    return lhs


@transform_errors
def parse_atom(ts):
    match next(ts):
        case val, TokenKind.LITERAL_INT, span:
            lhs = spanned(span, ast.Literal(val))
        case _, tok, _:
            raise NotImplementedError(tok)
    return lhs


def spanned(span, x):
    # used later, to add span info to ast nodes
    return x


def get_span(x):
    return None
