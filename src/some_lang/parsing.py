from __future__ import annotations

import abc
from functools import singledispatch
from typing import Optional, Any, Callable


class Token(abc.ABC):
    pass


class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
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

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        return self.maker().parse(tokens)


class MapParseResult(Parser):
    def __init__(self, parser: Any, func: Callable):
        self.func = func
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        match self.parser.parse(tokens):
            case r, rest:
                return self.func(r), rest
            case err:
                return err


class FilterParseResult(Parser):
    def __init__(self, parser: Parser, func: Callable):
        self.func = func
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        match self.parser.parse(tokens):
            case r, rest:
                if self.func(r):
                    return r, rest
                else:
                    return None
            case err:
                return err


class SequenceAll(Parser):
    def __init__(self, *parsers):
        self.parsers = [ensure_parser(p) for p in parsers]

    def parse(self, tokens: list[Token]) -> Optional[tuple[list[Any], list[Token]]]:
        results: list[Any] = []
        rest = tokens
        for p in self.parsers:
            match p.parse(rest):
                case None:
                    return None
                case x, r:
                    results.append(x)
                    rest = r
        return results, rest


class Alternative(Parser):
    def __init__(self, a: Parser, b: Parser):
        self.a = ensure_parser(a)
        self.b = ensure_parser(b)

    def parse(self, tokens: list[Token]) -> Optional[tuple[list[Any], list[Token]]]:
        return self.a.parse(tokens) or self.b.parse(tokens)


class ZeroOrMore(Parser):
    def __init__(self, parser):
        self.parser = ensure_parser(parser)

    def parse(self, tokens: list[Token]) -> Optional[tuple[list[Any], list[Token]]]:
        results: list[Any] = []
        rest = tokens
        while True:
            match self.parser.parse(rest):
                case None:
                    return results, rest
                case x, r:
                    results.append(x)
                    rest = r


class Exact(Parser):
    def __init__(self, tok: Token):
        self.tok = tok

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        match tokens:
            case [fst, *rst] if fst == self.tok:
                return fst, rst
        return None


class Succeed(Parser):
    def __init__(self):
        pass

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        return None, tokens


class Fail(Parser):
    def __init__(self):
        pass

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        return None


class Categoric(Parser):
    def __init__(self, typ: type):
        self.typ = typ

    def parse(self, tokens: list[Token]) -> Optional[tuple[Any, list[Token]]]:
        match tokens:
            case [fst, *rst] if isinstance(fst, self.typ):
                return fst, rst
        return None


def parse_sequence(*args):
    return SequenceAll(*args)


def parse_alternatives(a, *args):
    parser = a
    for b in args:
        parser = Alternative(parser, b)
    return parser


def parse_optional(parser):
    return Alternative(parser, Succeed())


def parse_repeated(parser):
    return ZeroOrMore(parser)


def parse_one_or_more(parser):
    return ZeroOrMore(parser).filter(lambda r: len(r) >= 1)


def final_result(tup):
    res, rest = tup
    if rest:
        raise ValueError("Unexpected tokens", rest)
    return res
