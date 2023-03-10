import functools
import string

import pyparsing as pp
from pyparsing.exceptions import ParseException

from cubiml import abstract_syntax as ast


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
    | "proc"
    | "let"
    | "rec"
    | "and"
    | "in"
    | "ref"
    | "do"
    | "end"
    | "return"
)
_ident_init_chars = string.ascii_lowercase + "_"
_ident_body_chars = _ident_init_chars + pp.nums + string.ascii_uppercase
ident = ~keyword + pp.Word(_ident_init_chars, _ident_body_chars)

tag = pp.Word("`", pp.alphas + pp.nums).add_parse_action(lambda t: t[0][1:])


expr = pp.Forward()
simple_expr = pp.Forward()

boolean = pp.MatchFirst(["false", "true"]).set_parse_action(
    lambda t: ast.Literal(t[0] == "true")
)
integer = pp.Combine(pp.Opt(pp.one_of("+ -")) + pp.Word(pp.nums)).set_parse_action(
    lambda t: ast.Literal(int(t[0]))
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


case = (tag + expr).add_parse_action(lambda t: ast.Case(t[0], t[1]))

match_arm = (pp.Literal("|") + tag + ident + "->" + expr).add_parse_action(
    lambda t: ast.MatchArm(t[1], t[2], t[4])
)

match = ("match" + expr + "with" + pp.OneOrMore(match_arm)).add_parse_action(
    lambda t: ast.Match(t[1], list(t[3:]))
)

block = ("do" + pp.Group(pp.delimited_list(expr, ";")) + "end").add_parse_action(
    lambda t: [t[1]]
)

procedure = ("proc" + ident + "->" + block).add_parse_action(
    lambda t: ast.Procedure(t[1], t[3])
)

function = ("fun" + ident + "->" + expr).add_parse_action(
    lambda t: ast.Function(t[1], t[3])
)

let = ("let" + ident + "=" + expr + "in" + expr).add_parse_action(
    lambda t: ast.Let(t[1], t[3], t[5])
)

funcdef = (ident + "=" + function | procedure).add_parse_action(
    lambda t: ast.FuncDef(t[0], t[2])
)
funcdefs = pp.Group(pp.delimited_list(funcdef, "and"))

letrec = (pp.Literal("let") + "rec" + funcdefs + "in" + expr).add_parse_action(
    lambda t: ast.LetRec(list(t[2]), t[4])
)


def left_associative(t, mkast, ty):
    items = t[::-1]
    expr = items.pop()
    while items:
        op = items.pop()
        rhs = items.pop()
        expr = mkast(expr, rhs, (ty, ty, ty), op)
    return expr


field_access = (simple_expr + pp.ZeroOrMore("." + ident)).add_parse_action(
    lambda t: functools.reduce(lambda x, f: ast.FieldAccess(f, x), t[2::2], t[0])
)

ref_get = ("!" + field_access).add_parse_action(lambda t: ast.RefGet(t[1]))
refget_expr = ref_get | field_access

ref_set = (refget_expr + ":=" + expr).add_parse_action(lambda t: ast.RefSet(t[0], t[2]))

newref = ("ref" + expr).add_parse_action(lambda t: ast.NewRef(t[1]))
newref_expr = newref | refget_expr

call_expr = pp.OneOrMore(newref_expr).add_parse_action(
    lambda t: functools.reduce(ast.Application, t[1:], t[0])
)

mult_expr = (call_expr + pp.ZeroOrMore(pp.one_of("* /") + call_expr)).add_parse_action(
    lambda t: left_associative(t, ast.BinOp, "int")
)

add_expr = (mult_expr + pp.ZeroOrMore(pp.one_of("+ -") + mult_expr)).add_parse_action(
    lambda t: left_associative(t, ast.BinOp, "int")
)

cmp_expr = (add_expr + pp.Opt(pp.one_of("< <= >= >") + add_expr)).add_parse_action(
    lambda t: ast.BinOp(t[0], t[2], ("int", "int", "bool"), t[1])
    if len(t) > 1
    else t[0]
)

eq_expr = (cmp_expr + pp.Opt(pp.one_of("== !=") + cmp_expr)).add_parse_action(
    lambda t: ast.BinOp(t[0], t[2], ("any", "any", "bool"), t[1])
    if len(t) > 1
    else t[0]
)

simple_expr <<= (
    record | boolean | integer | varref | (pp.Suppress("(") + expr + pp.Suppress(")"))
)

expr <<= (
    conditional | function | procedure | letrec | let | match | case | ref_set | eq_expr
)


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
