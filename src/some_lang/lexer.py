import dataclasses
import re
from typing import Iterator, Callable

from some_lang.parsing import Token

DELIMITERS = re.compile(r"(\s+|[()[])")


def skip_all_whitespace(tokens: Iterator[Token]) -> Iterator[Token]:
    return make_token_skipper(Whitespace, Indent)(tokens)


def make_token_skipper(*to_skip: type) -> Callable:
    def skipper(tokens: Iterator[Token]) -> Iterator[Token]:
        return filter(lambda t: not isinstance(t, to_skip), tokens)

    return skipper


@dataclasses.dataclass
class Int(Token):
    value: int


@dataclasses.dataclass
class Symbol(Token):
    value: str


@dataclasses.dataclass
class Whitespace(Token):
    pass


@dataclasses.dataclass
class Indent(Token):
    amount: int


def tokenize(src) -> Iterator[Token]:
    return map(match_token, filter(lambda x: x, DELIMITERS.split(src)))


def match_token(tok: str) -> Token:
    try:
        return Int(int(tok))
    except ValueError:
        pass

    if tok.isspace():
        match tok.rsplit("\n", 1):
            case [_]:
                return Whitespace()
            case [_, s]:
                return Indent(len(s))
            case x:
                print(x)

    match tok:
        case _:
            return Symbol(tok)
