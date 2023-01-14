from __future__ import annotations

import abc
import dataclasses
from functools import singledispatch
from typing import Optional, Any, Callable


class Token(abc.ABC):
    pass


@dataclasses.dataclass
class ParseResult:
    value: Any
    rest: list[Token]


@dataclasses.dataclass
class ParseErr(Exception):
    actual: list[Token]
    expected: list[Any]


class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self, tokens: list[Token]) -> ParseResult:
        pass

    def map(self, func: Callable) -> Parser:
        return MapParseResult(self, func)

    def filter(self, func: Callable) -> Parser:
        return FilterParseResult(self, func)


@singledispatch
def ensure_parser(obj: Any) -> Parser:
    raise TypeError(f"Not a valid parser : {obj}")


@ensure_parser.register
def _(obj: Parser) -> Parser:
    return obj


@ensure_parser.register
def _(obj: type) -> Parser:
    return Categoric(obj)


@ensure_parser.register
def _(obj: Token) -> Parser:
    return Exact(obj)


class LazyParser(Parser):
    def __init__(self, maker: Callable):
        self.maker = maker

    def parse(self, tokens: list[Token]) -> ParseResult:
        return self.maker().parse(tokens)


class MapParseResult(Parser):
    def __init__(self, parser: Any, func: Callable):
        self.func = func
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> ParseResult:
        match self.parser.parse(tokens):
            case ParseResult(r, rest):
                return ParseResult(self.func(r), rest)


class FilterParseResult(Parser):
    def __init__(self, parser: Parser, func: Callable):
        self.func = func
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> ParseResult:
        match self.parser.parse(tokens):
            case ParseResult(r, rest):
                if self.func(r):
                    return ParseResult(r, rest)
                else:
                    raise ParseErr(tokens, [])


class SequenceAll(Parser):
    def __init__(self, *parsers, name: str):
        self.name = name
        self.parsers = [ensure_parser(p) for p in parsers]

    def parse(self, tokens: list[Token]) -> ParseResult:
        results: list[Any] = []
        rest = tokens
        for p in self.parsers:
            match p.parse(rest):
                case ParseResult(x, r):
                    results.append(x)
                    rest = r
        return ParseResult(results, rest)


class Alternative(Parser):
    def __init__(self, a: Parser, b: Parser, name: str):
        self.name = name
        self.a = ensure_parser(a)
        self.b = ensure_parser(b)

    def parse(self, tokens: list[Token]) -> ParseResult:
        try:
            match self.a.parse(tokens):
                case ParseResult() as ok:
                    return ok
        except ParseErr as e:
            expected_a = e.expected

        try:
            match self.b.parse(tokens):
                case ParseResult() as ok:
                    return ok
        except ParseErr as e:
            expected_b = e.expected

        raise ParseErr(tokens, expected_a + expected_b)


class ZeroOrMore(Parser):
    def __init__(self, parser, name:str):
        self.name = name
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> ParseResult:
        results: list[Any] = []
        rest = tokens
        while True:
            try:
                match self.parser.parse(rest):
                    case ParseResult(x, r):
                        results.append(x)
                        rest = r
            except ParseErr:
                return ParseResult(results, rest)


class Exact(Parser):
    def __init__(self, tok: Token):
        self.tok = tok

    def parse(self, tokens: list[Token]) -> ParseResult:
        match tokens:
            case [fst, *rst] if fst == self.tok:
                return ParseResult(fst, rst)
        raise ParseErr(tokens, [self.tok])


class Succeed(Parser):
    def __init__(self):
        pass

    def parse(self, tokens: list[Token]) -> ParseResult:
        return ParseResult(None, tokens)


class Fail(Parser):
    def __init__(self):
        pass

    def parse(self, tokens: list[Token]) -> ParseResult:
        raise ParseErr(tokens, ["n/a"])


class Categoric(Parser):
    def __init__(self, typ: type):
        self.typ = typ

    def parse(self, tokens: list[Token]) -> ParseResult:
        match tokens:
            case [fst, *rst] if isinstance(fst, self.typ):
                return ParseResult(fst, rst)
        raise ParseErr(tokens, [self.typ])


def parse_sequence(*args, name: str):
    return SequenceAll(*args, name=name)


def parse_alternatives(a, *args, name=None):
    parser = a
    for b in args:
        parser = Alternative(parser, b, name=name)
    return parser


def parse_optional(parser, name:str):
    return Alternative(parser, Succeed(), name=name)


def parse_repeated(parser, name:str):
    return ZeroOrMore(parser, name=name)


def parse_one_or_more(parser, name: str):
    return ZeroOrMore(parser, name=name).filter(lambda r: len(r) >= 1)


def parse_delimited_nonempty_list(item, delimiter, name=None):
    return parse_sequence(
        item,
        parse_repeated(
            parse_sequence(delimiter, item, name=name).map(lambda x: x[1]), name=name
        ),
        name=name,
    )


def final_result(res: ParseResult):
    match res:
        case ParseResult(r, rest):
            if rest:
                raise ValueError("Unexpected tokens", rest)
            return r
        case ParseErr(actual, expected):
            raise ValueError(f"Expected: {expected}\nActual: {actual}")
