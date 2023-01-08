from __future__ import annotations

import functools

from some_lang import ast, lexer, parsing
from some_lang.lexer import Symbol, Int
from some_lang.parsing import (
    Parser,
    LazyParser,
    MapParseResult,
    Exact,
    ensure_parser,
    parse_alternatives,
    parse_sequence,
)


def parse_program(src: str) -> ast.Expr:
    return compose(
        lexer.tokenize,
        lexer.skip_all_whitespace,
        list,
        expr_parser().parse,
        parsing.final_result,
    )(src)


@ensure_parser.register
def _(obj: str) -> Parser:
    return Exact(Symbol(obj))


def expr_parser():
    return parse_alternatives(
        func_parser(),
        apply_parser(),
        MapParseResult(Int, lambda tok: ast.Integer(tok.value)),
        MapParseResult(Symbol, lambda tok: ast.Reference(tok.value)),
    )


def func_parser():
    return parse_sequence(
        "(", "lambda", "(", Symbol, ")", LazyParser(expr_parser), ")"
    ).map(lambda x: ast.Lambda(x[3], x[5]))


def apply_parser():
    return parse_sequence(
        "(", LazyParser(expr_parser), LazyParser(expr_parser), ")"
    ).map(lambda x: ast.Application(x[1], x[2]))


def compose(*funcs):
    return lambda arg: functools.reduce(lambda x, f: f(x), funcs, arg)
