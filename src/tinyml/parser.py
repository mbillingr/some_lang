from tinyml import abstract_syntax as ast
from tinyml.tokenizer import (
    Token,
    TokenStream,
    TokenKind,
    Span,
    UnexpectedEnd,
    UnexpectedToken,
)

infix_binding_power = {
    "!=": (3, 4),
    "==": (3, 4),
    "<": (3, 4),
    ">": (3, 4),
    "<=": (3, 4),
    ">=": (3, 4),
    "+": (5, 6),
    "-": (5, 6),
    "*": (7, 8),
    "/": (7, 8),
    "**": (14, 13),
    "< apply >": (15, 16),  # function call
}

prefix_binding_power = {
    "fn": (None, 1),
    "if": (None, 3),
    "~": (None, 9),
}

postfix_binding_power = {
    "!": (11, None),
}

op_types = {
    "+": ("int", "int", "int"),
    "-": ("int", "int", "int"),
    "*": ("int", "int", "int"),
    "/": ("int", "int", "int"),
    "~": ("bool", "bool"),
    "**": ("int", "int", "int"),
    "!": ("int", "int"),
    "<": ("int", "int", "bool"),
    ">": ("int", "int", "bool"),
    "<=": ("int", "int", "bool"),
    ">=": ("int", "int", "bool"),
    "==": ("any", "any", "bool"),
    "!=": ("any", "any", "bool"),
}


def parse_toplevel(ts: TokenStream) -> ast.Script:
    lets = []
    funcs = []
    exprs = []
    nesting_depth = 1
    while nesting_depth > 0:
        match ts.peek():
            case "let", _, span:
                ts.get_next()
                var = parse_identifier(ts)
                expect_token(ts, "=")
                val = parse_expr(ts)
                let = spanned(span.merge(get_span(val)), ast.DefineLet(var, val))
                lets.append(let)
            case "func", _, span:
                ts.get_next()
                name = parse_identifier(ts)
                var = parse_identifier(ts)
                expect_token(ts, "=")
                body = parse_expr(ts)
                func = spanned(
                    span.merge(get_span(body)),
                    ast.FuncDef(name, ast.Function(var, body)),
                )
                funcs.append(func)
            case _:
                exprs.append(parse_expr(ts))

        match ts.peek():
            case _, TokenKind.TOPLEVEL_SEPARATOR, _:
                ts.get_next()
            case _:
                break

    funcs = funcs and [ast.DefineLetRec(funcs)]

    return ast.Script(funcs + lets + exprs)


def parse_expr(ts: TokenStream, min_bp: int = 0) -> ast.Expression:
    match ts.peek():
        case op, _, _ if op in prefix_binding_power:
            _, rbp = prefix_binding_power[op]
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
            case ts.EOF:
                break
            case t, k, _ if is_delimiter(t, k):
                break
            case _:
                # we treat application as an "invisible" infix operator
                lbp, rbp = infix_binding_power["< apply >"]
                if lbp < min_bp:
                    break
                arg = parse_expr(ts, rbp)
                lhs = spanned(
                    get_span(lhs).merge(get_span(arg)), ast.Application(lhs, arg)
                )

    return lhs


def is_delimiter(t, k) -> bool:
    match t, k:
        case "true" | "false", _:
            return False
        case _, TokenKind.RPAREN | TokenKind.KEYWORD | TokenKind.TOPLEVEL_SEPARATOR:
            return True
        case _:
            return False


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
        case "the", _, span:
            typ = parse_type(ts)
            exp = parse_expr(ts)
            return spanned(span.merge(get_span(exp)), ast.Annotation(exp, typ))
        case "let", _, span:
            var = parse_identifier(ts)
            expect_token(ts, "=")
            val = parse_expr(ts)
            expect_token(ts, "in")
            body = parse_expr(ts)
            return spanned(span.merge(get_span(body)), ast.Let(var, val, body))
        case "{", _, span:
            _, _, sp2 = expect_token(ts, "}")
            return spanned(span.merge(sp2), ast.Record([]))
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
        case "fn", _, span:
            var = parse_identifier(ts)
            expect_token(ts, "=>")
            body = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(body)), ast.Function(var, body))
        case "if", _, span:
            cond = parse_expr(ts)
            expect_token(ts, "then")
            lhs = parse_expr(ts)
            expect_token(ts, "else")
            rhs = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(rhs)), ast.Conditional(cond, lhs, rhs))
        case token:
            raise UnexpectedToken(token)


def parse_infix_operator(lhs, rbp, ts):
    match ts.get_next():
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


def parse_type(ts):
    typ = parse_atomic_type(ts)

    match ts.peek():
        case "->", _, _:
            ts.get_next()
            rhs = parse_type(ts)
            return spanned(get_span(typ).merge(get_span(rhs)), ast.FuncType(typ, rhs))
        case _:
            return typ


def parse_atomic_type(ts):
    match ts.get_next():
        case txt, TokenKind.IDENTIFIER, span:
            return spanned(span, ast.TypeLiteral(txt))
        case "(", _, span:
            inner = parse_type(ts)
            expect_token(ts, ")")
            return inner
        case "lambda", _, span:
            expect_token(ts, "(")
            var = parse_identifier(ts)
            expect_token(ts, ")")
            body = parse_type(ts)
            return spanned(
                span.merge(get_span(body)), ast.TypeFunction(var, frozenset(), body)
            )
        case "apply", _, span:
            tfun = parse_type(ts)
            targ = parse_type(ts)
            return spanned(
                span.merge(get_span(targ)), ast.TypeApplication(tfun, targ)
            )
        case token:
            raise NotImplementedError(token)


def parse_identifier(ts) -> ast.Identifier:
    tok, _, span = expect_token(ts, TokenKind.IDENTIFIER)
    return spanned(span, ast.Identifier(tok))


def expect_tokens(ts, *expect):
    return [expect_token(ts, ex) for ex in expect]


def expect_token(ts, expect):
    match ts.get_next():
        case ts.EOF:
            raise UnexpectedEnd()
        case tok, kind, span:
            if isinstance(expect, TokenKind):
                if kind != expect:
                    raise UnexpectedToken((tok, kind, span))
            elif tok != expect:
                raise UnexpectedToken((tok, kind, span))

            return tok, kind, span


def try_token(ts, expect) -> Token | bool:
    match ts.peek(), expect:
        case (tok, kind, span), TokenKind() if kind == expect:
            ts.get_next()
            return tok, kind, span
        case (tok, kind, span), _ if tok == expect:
            ts.get_next()
            return tok, kind, span
        case _:
            return False


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
