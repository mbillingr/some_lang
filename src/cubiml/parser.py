import functools
import string
import weakref

import pyparsing as pp

from cubiml import ast

keyword = pp.Literal("if") | "then" | "else" | "fun" | "let"
_ident_init_chars = string.ascii_lowercase + "_"
_ident_body_chars = _ident_init_chars + pp.nums + string.ascii_uppercase
ident = ~keyword + pp.Word(_ident_init_chars, _ident_body_chars)


expr = pp.Forward()
simple_expr = pp.Forward()
call_expr = pp.Forward()

boolean = pp.MatchFirst(["false", "true"]).set_parse_action(
    lambda t: ast.Boolean(t[0] == "true")
)

varref = ident.copy().set_parse_action(lambda t: ast.Reference(t[0]))

conditional = ("if" + expr + "then" + expr + "else" + expr).set_parse_action(
    lambda t: ast.Conditional(t[1], t[3], t[5])
)

record = (
    pp.Suppress("{")
    + pp.delimited_list(pp.Group(ident + pp.Suppress("=") + expr), ";")[..., 1]
    + pp.Suppress("}")
).add_parse_action(lambda tok: ast.Record(dict(map(tuple, tok))))

field_access = (simple_expr + pp.OneOrMore("." + ident)).add_parse_action(
    lambda t: functools.reduce(lambda x, f: ast.FieldAccess(f, x), t[2::2], t[0])
)

function = ("fun" + ident + "->" + expr).add_parse_action(
    lambda t: ast.Function(t[1], t[3])
)

simple_expr <<= record | boolean | varref | (pp.Suppress("(") + expr + pp.Suppress(")"))

call_expr <<= pp.OneOrMore(simple_expr).add_parse_action(
    lambda t: functools.reduce(ast.Application, t[1:], t[0])
)

expr <<= conditional | function | field_access | call_expr


### source location tracking

_LOCS = {}


def get_loc(obj):
    return _LOCS[id(obj)]


def record_location(l, t):
    _LOCS[id(t[0])] = l
    return t[0]


expr.set_parse_action(record_location)

###
