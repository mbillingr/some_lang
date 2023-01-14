from __future__ import annotations

import functools

from some_lang import ast, lexer, parsing
from some_lang.lexer import Symbol, Int, Indent, Dedent, Bool
from some_lang.parsing import (
    Parser,
    LazyParser,
    MapParseResult,
    Exact,
    ensure_parser,
    parse_alternatives,
    parse_sequence,
    parse_repeated,
    Fail,
    parse_one_or_more,
)


def parse_module(src: str) -> ast.Module:
    return compose(
        lexer.tokenize,
        lexer.skip_all_whitespace,
        list,
        module_parser().parse,
        parsing.final_result,
    )(src)


def inspect(x):
    print(x)
    return x


def parse_program(src: str) -> ast.Expression:
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


def module_parser():
    def build_module(res):
        defs, stmts = [], []
        print(res)
        for obj in res:
            match obj:
                case ast.Definition():
                    defs.append(obj)
                case ast.Statement():
                    stmts.append(obj)
        return ast.Module(defs, stmts)

    return parse_repeated(parse_alternatives(func_parser(), stmt_parser())).map(
        build_module
    )


def func_parser():
    return parse_sequence(
        "def",
        MapParseResult(Symbol, lambda tok: tok.value),
        "(",
        type_parser(),
        ")",
        "->",
        type_parser(),
        ":",
        Indent,
        def_body_parser(),
        Dedent,
    ).map(lambda x: ast.Definition(x[1], x[3], x[6], x[9]))


def def_body_parser():
    return parse_one_or_more(pattern_def_parser())


def pattern_def_parser():
    return parse_sequence(
        MapParseResult(Symbol, lambda tok: tok.value),
        "(",
        parse_alternatives(
            MapParseResult(Int, lambda tok: ast.IntegerPattern(tok.value)),
            MapParseResult(Symbol, lambda tok: ast.BindingPattern(tok.value)),
        ),
        ")",
        "=",
        expr_parser(),
    ).map(lambda r: ast.DefinitionPattern(r[0], r[2], r[5]))


def type_parser():
    return MapParseResult(Symbol("Int"), lambda tok: ast.IntegerType())


def stmt_parser():
    return parse_sequence("print", expr_parser()).map(
        lambda x: ast.PrintStatement(x[1])
    )


def expr_parser():
    return parse_alternatives(
        lambda_parser(),
        apply_parser(),
        MapParseResult(Bool, lambda tok: ast.Boolean(tok.value)),
        MapParseResult(Int, lambda tok: ast.Integer(tok.value)),
        MapParseResult(Symbol, lambda tok: ast.Reference(tok.value)),
    )


def lambda_parser():
    return parse_sequence(
        "(", "lambda", "(", Symbol, ")", LazyParser(expr_parser), ")"
    ).map(lambda x: ast.Lambda(x[3].value, x[5]))


def apply_parser():
    return parse_sequence(
        "(", LazyParser(expr_parser), LazyParser(expr_parser), ")"
    ).map(lambda x: ast.Application(x[1], x[2]))


def compose(*funcs):
    return lambda arg: functools.reduce(lambda x, f: f(x), funcs, arg)
