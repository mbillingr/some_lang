import functools
import string

import pyparsing as pp

from cubiml import ast


def parse_script(src: str) -> ast.Script:
    return script.parse_string(src, True)[0]


def parse_expr(src: str) -> ast.Expression:
    return expr.parse_string(src, True)[0]


### Grammar

keyword = (
    pp.Literal("if")
    | "then"
    | "else"
    | "match"
    | "with"
    | "fun"
    | "let"
    | "rec"
    | "and"
    | "in"
)
_ident_init_chars = string.ascii_lowercase + "_"
_ident_body_chars = _ident_init_chars + pp.nums + string.ascii_uppercase
ident = ~keyword + pp.Word(_ident_init_chars, _ident_body_chars)

tag = pp.Word("`", pp.alphas + pp.nums).add_parse_action(lambda t: t[0][1:])


expr = pp.Forward()
simple_expr = pp.Forward()
call_expr = pp.Forward()

boolean = pp.MatchFirst(["false", "true"]).set_parse_action(
    lambda t: ast.Literal(t[0] == "true")
)

varref = ident.copy().set_parse_action(lambda t: ast.Reference(t[0]))

conditional = ("if" + expr + "then" + expr + "else" + expr).set_parse_action(
    lambda t: ast.Conditional(t[1], t[3], t[5])
)

record = (
    pp.Suppress("{")
    + pp.delimited_list(pp.Group(ident + pp.Suppress("=") + expr), ";")[..., 1]
    + pp.Suppress("}")
).add_parse_action(lambda tok: ast.Record(list(map(tuple, tok))))

field_access = (simple_expr + pp.OneOrMore("." + ident)).add_parse_action(
    lambda t: functools.reduce(lambda x, f: ast.FieldAccess(f, x), t[2::2], t[0])
)

case = (tag + expr).add_parse_action(lambda t: ast.Case(t[0], t[1]))

match_arm = (pp.Literal("|") + tag + ident + "->" + expr).add_parse_action(
    lambda t: ast.MatchArm(t[1], t[2], t[4])
)

match = ("match" + expr + "with" + pp.OneOrMore(match_arm)).add_parse_action(
    lambda t: ast.Match(t[1], list(t[3:]))
)

function = ("fun" + ident + "->" + expr).add_parse_action(
    lambda t: ast.Function(t[1], t[3])
)

let = ("let" + ident + "=" + expr + "in" + expr).add_parse_action(
    lambda t: ast.Let(t[1], t[3], t[5])
)

funcdef = (ident + "=" + function).add_parse_action(lambda t: ast.FuncDef(t[0], t[2]))
funcdefs = pp.Group(pp.delimited_list(funcdef, "and"))

letrec = (pp.Literal("let") + "rec" + funcdefs + "in" + expr).add_parse_action(
    lambda t: ast.LetRec(list(t[2]), t[4])
)

simple_expr <<= record | boolean | varref | (pp.Suppress("(") + expr + pp.Suppress(")"))

call_expr <<= pp.OneOrMore(simple_expr).add_parse_action(
    lambda t: functools.reduce(ast.Application, t[1:], t[0])
)

expr <<= conditional | function | letrec | let | match | case | field_access | call_expr


deflet = ("let" + ident + "=" + expr).add_parse_action(
    lambda t: ast.DefineLet(t[1], t[3])
)

defletrec = (pp.Literal("let") + "rec" + funcdefs).add_parse_action(
    lambda t: ast.DefineLetRec(list(t[2]))
)

top_item = defletrec | deflet | expr

script = pp.delimited_list(top_item, ";").add_parse_action(
    lambda t: ast.Script(list(t))
)


### source location tracking

_LOCS = {}


def get_loc(obj):
    return _LOCS[id(obj)]


def record_location(l, t):
    _LOCS[id(t[0])] = l
    return t[0]


expr.set_parse_action(record_location)

###
