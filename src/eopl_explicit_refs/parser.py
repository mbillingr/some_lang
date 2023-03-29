from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs.tokenizer import (
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
    "deref": (None, 3),
    "newref": (None, 3),
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


def parse_program(ts: TokenStream) -> ast.Program:
    return ast.Program(parse_expr(ts))


def parse_statement(ts: TokenStream) -> ast.Statement:
    match ts.peek():
        case "if", _, span:
            ts.get_next()
            cond = parse_expr(ts)
            expect_token(ts, "then")
            lhs = parse_statements(ts)
            expect_token(ts, "else")
            rhs = parse_statements(ts)
            return spanned(span.merge(get_span(rhs)), ast.IfStatement(cond, lhs, rhs))
        case "set", _, span:
            ts.get_next()
            lhs = parse_expr(ts)
            expect_token(ts, "=")
            rhs = parse_expr(ts)
            return spanned(span.merge(get_span(rhs)), ast.Assignment(lhs, rhs))
        case _:
            expr = parse_expr(ts)
            return spanned(get_span(expr), ast.ExprStmt(expr))


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
        case "=" | ";", _:
            return True
        case _, TokenKind.RPAREN | TokenKind.KEYWORD:
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
            return spanned(span, ast.Identifier(name))
        case "(", _, span:
            inner = parse_expr(ts)
            _, _, sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), inner)
        case "begin", _, span:
            expr = parse_sequence(ts)
            return spanned(span.merge(get_span(expr)), expr)
        case "let", _, span:
            var = parse_symbol(ts)
            expect_token(ts, "=")
            val = parse_expr(ts)
            expect_token(ts, "in")
            body = parse_sequence(ts)
            return spanned(span.merge(get_span(body)), ast.Let(var, val, body))
        case token:
            raise UnexpectedToken(token)


def parse_sequence(ts):
    stmts = []
    while True:
        stmts.append(parse_statement(ts))
        match ts.peek():
            case ";", _, _:
                pass
            case ts.EOF:
                break
            case _:
                break
        ts.get_next()
    expr = ast.stmt_to_expr(
        stmts.pop()
    )  # last statement is expected to be an expression
    while stmts:
        s = stmts.pop()
        expr = spanned(get_span(s).merge(get_span(expr)), ast.Sequence(s, expr))
    return expr


def parse_statements(ts):
    stmts = []
    while True:
        stmts.append(parse_statement(ts))
        match ts.peek():
            case ";", _, _:
                pass
            case ts.EOF:
                break
            case _:
                break
        ts.get_next()
    compound_statement = stmts.pop()
    while stmts:
        s = stmts.pop()
        compound_statement = spanned(
            get_span(s).merge(get_span(compound_statement)),
            ast.Statements(s, compound_statement),
        )
    return compound_statement


def parse_prefix_operator(rbp, ts):
    match ts.get_next():
        case op, TokenKind.OPERATOR, span if op in prefix_binding_power:
            rhs = parse_expr(ts, rbp)
            return spanned(
                make_operator_span(span, get_span(rhs)),
                ast.UnaryOp(rhs, op_types[op], op),
            )
        case "fn", _, span:
            arms = parse_match_arms(ts)
            return spanned(span.merge(get_span(arms[-1])), ast.Function(arms))
        case "newref", _, span:
            val = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(val)), ast.NewRef(val))
        case "deref", _, span:
            ref = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(ref)), ast.DeRef(ref))
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
                ast.BinOp(lhs, rhs, op),
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


def parse_match_arms(ts) -> list[ast.MatchArm]:
    arms = [parse_match_arm(ts)]
    while ts.peek()[0] == "|":
        ts.get_next()
        arms.append(parse_match_arm(ts))
    return arms


def parse_match_arm(ts) -> ast.MatchArm:
    pat = parse_pattern(ts)
    expect_token(ts, "=>")
    body = parse_expr(ts)
    return spanned(get_span(pat).merge(get_span(body)), ast.MatchArm(pat, body))


def parse_pattern(ts) -> ast.Pattern:
    match ts.peek():
        case ident, TokenKind.IDENTIFIER, span:
            ts.get_next()
            return spanned(span, ast.BindingPattern(ident))
        case value, TokenKind.LITERAL_BOOL | TokenKind.LITERAL_INT, span:
            ts.get_next()
            return spanned(span, ast.LiteralPattern(value))
        case token:
            raise NotImplementedError(token)


def parse_symbol(ts) -> ast.Symbol:
    tok, _, span = expect_token(ts, TokenKind.IDENTIFIER)
    return spanned(span, ast.Symbol(tok))


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
