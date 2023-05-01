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
    "::": (6, 5),
    "+": (7, 8),
    "-": (7, 8),
    "*": (9, 10),
    "/": (9, 10),
    "**": (16, 15),
    "< apply >": (17, 18),  # function call
}

prefix_binding_power = {
    "fn": (None, 1),
    "if": (None, 1),
    "newref": (None, 5),
    "~": (None, 11),
    "the": (None, 19),
    "deref": (None, 97),
}

postfix_binding_power = {
    "!": (13, None),
    ".": (99, None),
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
    records = []
    while True:
        match ts.peek():
            case "struct", _, _:
                records.append(parse_struct_decl(ts))
            case _:
                return ast.Program(parse_expr(ts), records)


def parse_statement(ts: TokenStream) -> ast.Statement:
    match ts.peek():
        case "if", _, span:
            ts.get_next()
            cond = parse_expr(ts)
            expect_token(ts, "then")
            lhs = parse_statement(ts)
            if try_token(ts, "else"):
                rhs = parse_statement(ts)
            else:
                rhs = ast.NopStatement()
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


def parse_expr(ts: TokenStream, min_bp: int = 0, invisible_application=True) -> ast.Expression:
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
            case _ if invisible_application:
                # we treat application as an "invisible" infix operator
                lbp, rbp = infix_binding_power["< apply >"]
                if lbp < min_bp:
                    break
                arg = parse_expr(ts, rbp)
                lhs = spanned(get_span(lhs).merge(get_span(arg)), ast.Application(lhs, arg))
            case _:
                break

    return lhs


def is_delimiter(t, k) -> bool:
    match t, k:
        case "()" | "true" | "false", _:
            return False
        case "=" | ";" | ",", _:
            return True
        case _, TokenKind.RPAREN | TokenKind.KEYWORD:
            return True
        case _:
            return False


def parse_atom(ts: TokenStream):
    match ts.get_next():
        case ts.EOF:
            raise UnexpectedEnd()
        case "()", _, span:
            return spanned(span, ast.Literal(None))
        case val, TokenKind.LITERAL_BOOL | TokenKind.LITERAL_INT, span:
            return spanned(span, ast.Literal(val))
        case name, TokenKind.IDENTIFIER, span:
            return spanned(span, ast.Identifier(name))
        case "(", _, span:
            inner = parse_expr(ts)
            _, _, sp2 = expect_token(ts, ")")
            return spanned(span.merge(sp2), inner)
        case "{", _, span:
            expr = parse_block_expression(ts, skip_opening=True)
            return spanned(span.merge(get_span(expr)), expr)
        case "[", _, span:
            match ts.peek(1):
                case "=", _, _:
                    expr = parse_record_expression(ts, skip_opening=True)
                    return spanned(span.merge(get_span(expr)), expr)
                case other:
                    expr = parse_list_expression(ts, skip_opening=True)
                    return spanned(span.merge(get_span(expr)), expr)
        case "let", _, span:
            var = parse_symbol(ts)
            if try_token(ts, ":"):
                typ = parse_type(ts)
            else:
                typ = None
            expect_token(ts, "=")
            val = parse_expr(ts)
            expect_token(ts, "in")
            body = parse_expr(ts)
            return spanned(span.merge(get_span(body)), ast.Let(var, val, body, var_t=typ))
        case token:
            raise UnexpectedToken(token)


def parse_block_expression(ts, skip_opening: bool):
    if not skip_opening:
        expect_token(ts, "{")

    stmts = []
    while True:
        stmts.append(parse_statement(ts))
        match ts.get_next():
            case "}", _, span:
                break
            case ";", _, _:
                pass
            case tok:
                raise UnexpectedToken(tok)
    # last statement is expected to be an expression
    last = stmts.pop()
    expr = ast.stmt_to_expr(last)
    expr_span = get_span(last)
    while stmts:
        s = stmts.pop()
        expr_span = get_span(s).merge(expr_span)
        expr = spanned(expr_span, ast.BlockExpression(s, expr))
    return spanned(expr_span.merge(span), expr)


def parse_block_statement(ts, skip_opening: bool):
    if not skip_opening:
        expect_token(ts, "{")

    stmts = []
    while True:
        stmts.append(parse_statement(ts))
        match ts.get_next():
            case "}", _, _:
                break
            case ";", _, _:
                pass
            case tok:
                raise UnexpectedToken(tok)
    compound_statement = stmts.pop()
    while stmts:
        s = stmts.pop()
        compound_statement = spanned(
            get_span(s).merge(get_span(compound_statement)),
            ast.BlockStatement(s, compound_statement),
        )
    return compound_statement


def parse_list_expression(ts, skip_opening: bool):
    span0 = None
    if not skip_opening:
        _, _, span0 = expect_token(ts, "[")

    match ts.peek():
        case "]", _, span:
            ts.get_next()
            result = spanned(span, ast.EmptyList())
        case _:
            first = parse_expr(ts)
            # Support trailing comma by making it optional everywhere. An omitted
            # comma between two elements would be correctly parsed as application.
            try_token(ts, ",")
            rest = parse_list_expression(ts, skip_opening=True)
            result = spanned(get_span(first).merge(get_span(rest)), ast.BinOp(first, rest, "::"))

    if span0 is not None:
        return spanned(span0.merge(get_span(result)), result)
    else:
        return result


def parse_record_expression(ts, skip_opening: bool):
    span0 = None
    if not skip_opening:
        _, _, span0 = expect_token(ts, "[")

    fields = {}
    while True:
        field_name = parse_symbol(ts)
        expect_token(ts, "=")
        field_value = parse_expr(ts)
        fields[field_name] = field_value
        if try_token(ts, ","):
            continue
        else:
            break

    _, _, span1 = expect_token(ts, "]")

    result = ast.RecordExpr(fields)

    if span0 is not None:
        return spanned(span0.merge(span1), result)
    else:
        return result


def parse_prefix_operator(rbp, ts):
    match ts.get_next():
        case op, TokenKind.OPERATOR, span if op in prefix_binding_power:
            rhs = parse_expr(ts, rbp)
            return spanned(
                make_operator_span(span, get_span(rhs)),
                ast.UnaryOp(rhs, op_types[op], op),
            )
        case "the", _, span:
            ty = parse_type(ts)
            expr = parse_expr(ts, rbp)
            return spanned(span.merge(get_span(expr)), ast.TypeAnnotation(ty, expr))
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
        case ".", _, _:
            fieldname = parse_symbol(ts)
            return spanned(get_span(lhs).merge(get_span(fieldname)), ast.GetField(lhs, fieldname))
        case op, TokenKind.OPERATOR, span:
            return spanned(
                make_operator_span(span, get_span(lhs)),
                ast.UnaryOp(lhs, op_types[op], op),
            )
        case token:
            raise UnexpectedToken(token)


def parse_match_arms(ts) -> list[ast.MatchArm]:
    arms = [parse_match_arm(ts)]
    while True:
        match ts.peek():
            case "|", _, _:
                ts.get_next()
            case _:
                break
        arms.append(parse_match_arm(ts))
    return arms


def parse_match_arm(ts) -> ast.MatchArm:
    patterns = []
    while True:
        pat = parse_pattern(ts)
        patterns.append(pat)
        if try_token(ts, "=>"):
            break
    body = parse_expr(ts)
    return spanned(get_span(patterns[0]).merge(get_span(body)), ast.MatchArm(patterns, body))


def parse_pattern(ts) -> ast.Pattern:
    lhs = parse_atomic_pattern(ts)
    match ts.peek():
        case "::", _, span:
            ts.get_next()
            rhs = parse_pattern(ts)
            return spanned(get_span(lhs).merge(get_span(rhs)), ast.ListConsPattern(lhs, rhs))
        case _:
            return lhs


def parse_atomic_pattern(ts) -> ast.Pattern:
    match ts.peek():
        case "()", _, span:
            ts.get_next()
            return spanned(span, ast.LiteralPattern(None))
        case ident, TokenKind.IDENTIFIER, span:
            ts.get_next()
            if try_token(ts, ":"):
                _ = parse_type(ts)
            return spanned(span, ast.BindingPattern(ident))
        case value, TokenKind.LITERAL_BOOL | TokenKind.LITERAL_INT, span:
            ts.get_next()
            return spanned(span, ast.LiteralPattern(value))
        case "[", _, span:
            pat = parse_list_pattern(ts)
            return spanned(span.merge(get_span(pat)), pat)
        case token:
            raise NotImplementedError(token)


def parse_list_pattern(ts) -> ast.Pattern:
    _, _, span0 = expect_token(ts, "[")

    match ts.peek():
        case "]", _, span:
            ts.get_next()
            return spanned(span0.merge(span), ast.EmptyListPattern())
        case _:
            raise NotImplementedError()


def parse_type(ts) -> ast.Type:
    ty = parse_atomic_type(ts)
    while try_token(ts, "->"):
        ret = parse_type(ts)
        ty = spanned(get_span(ty).merge(get_span(ret)), ast.FuncType(ty, ret))
    return ty


def parse_atomic_type(ts) -> ast.Type:
    match ts.get_next():
        case "()", _, span:
            return spanned(span, ast.NullType)
        case "Bool", _, span:
            return spanned(span, ast.BoolType)
        case "Int", _, span:
            return spanned(span, ast.IntType)
        case "[", _, span:
            match ts.peek(0), ts.peek(1):
                case (_, TokenKind.IDENTIFIER, _), (":", _, _):
                    fields = parse_record_fields(ts)
                    expect_token(ts, "]")
                    return spanned(span, ast.RecordType(fields))
                case _:
                    item_t = parse_type(ts)
                    expect_token(ts, "]")
                    return spanned(span, ast.ListType(item_t))
        case name, TokenKind.IDENTIFIER, span:
            return spanned(span, ast.TypeRef(name))
        case other:
            raise UnexpectedToken(other)


def parse_struct_decl(ts) -> ast.RecordDecl:
    _, _, span0 = expect_token(ts, "struct")
    name = parse_symbol(ts)
    expect_token(ts, "[")
    fields = parse_record_fields(ts)
    expect_token(ts, "]")
    return ast.RecordDecl(name, fields)


def parse_record_fields(ts) -> dict[ast.Symbol, ast.Type]:
    fields = {}
    while True:
        if try_token(ts, "]", consume=False):
            break
        field_name = parse_symbol(ts)
        expect_token(ts, ":")
        field_type = parse_type(ts)
        fields[field_name] = field_type
        if try_token(ts, ","):
            continue
        else:
            break
    return fields


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


def try_token(ts, expect, consume=True) -> Token | bool:
    match ts.peek(), expect:
        case (tok, kind, span), TokenKind() if kind == expect:
            if consume:
                ts.get_next()
            return tok, kind, span
        case (tok, kind, span), _ if tok == expect:
            if consume:
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
