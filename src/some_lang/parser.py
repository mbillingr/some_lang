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
    parse_delimited_nonempty_list,
)


def parse_module(src: str) -> ast.Module:
    return lex_and_parse(module_parser())(src)


def parse_expr(src: str) -> ast.Expression:
    return lex_and_parse(expr_parser())(src)


def parse_type(src: str) -> ast.Expression:
    return lex_and_parse(type_parser())(src)


def lex_and_parse(parser):
    return compose(
        lexer.tokenize,
        lexer.skip_all_whitespace,
        list,
        parser.parse,
        parsing.final_result,
    )


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

    return parse_repeated(
        parse_alternatives(func_parser(), stmt_parser()),
        name="Module",
    ).map(build_module)


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
        name="Function",
    ).map(lambda x: ast.Definition(x[1], x[3], x[6], x[9]))


def def_body_parser():
    return parse_one_or_more(pattern_def_parser(), name="Body")


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
        name="Pattern",
    ).map(lambda r: ast.DefinitionPattern(r[0], r[2], r[5]))


def type_parser():
    return parse_delimited_nonempty_list(
        atomic_type_parser(), Symbol("->"), name="Type"
    ).map(lambda r: fold(ast.FunctionType, r[1], r[0]))


def atomic_type_parser():
    return parse_alternatives(
        MapParseResult(Symbol("?"), lambda tok: ast.UnknownType()),
        MapParseResult(Symbol("Bool"), lambda tok: ast.BooleanType()),
        MapParseResult(Symbol("Int"), lambda tok: ast.IntegerType()),
        name="AtomicType",
    )


def stmt_parser():
    return parse_sequence("print", expr_parser(), name="Statement").map(
        lambda x: ast.PrintStatement(x[1])
    )


def expr_parser():
    return parse_alternatives(
        lambda_parser(),
        apply_parser(),
        MapParseResult(Bool, lambda tok: ast.Boolean(tok.value)),
        MapParseResult(Int, lambda tok: ast.Integer(tok.value)),
        MapParseResult(Symbol, lambda tok: ast.Reference(tok.value)),
        name="Expression",
    )


def lambda_parser():
    return parse_sequence(
        "(", "lambda", "(", Symbol, ")", LazyParser(expr_parser), ")", name="Lambda"
    ).map(lambda x: ast.Lambda(x[3].value, x[5]))


def apply_parser():
    return parse_sequence(
        "(", LazyParser(expr_parser), LazyParser(expr_parser), ")", name="Apply"
    ).map(lambda x: ast.Application(x[1], x[2]))


def compose(*funcs):
    return lambda arg: functools.reduce(lambda x, f: f(x), funcs, arg)


def fold(f, seq, init):
    if not seq:
        return init
    return f(init, fold(f, seq[1:], seq[0]))
