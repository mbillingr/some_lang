import dataclasses
import re
from typing import Iterator, Callable

from some_lang.parsing import Token

DELIMITERS = re.compile(r"(\s+|[:()[])")


def skip_all_whitespace(tokens: Iterator[Token]) -> Iterator[Token]:
    return make_token_skipper(Whitespace)(tokens)


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
    pass


@dataclasses.dataclass
class Dedent(Token):
    pass


def tokenize(src) -> Iterator[Token]:
    tm = TokenMatcher()
    yield from map(tm.match_token, filter(lambda x: x, DELIMITERS.split(src)))
    while tm.dedent():
        yield Dedent()


class TokenMatcher:
    def __init__(self):
        self.indent_levels = [0]

    def match_token(self, tok: str) -> Token:
        try:
            return Int(int(tok))
        except ValueError:
            pass

        if tok.isspace():
            match tok.rsplit("\n", 1):
                case [_]:
                    return Whitespace()
                case [_, s]:
                    ls = len(s)
                    if ls == self.indent_levels[-1]:
                        return Whitespace()
                    elif ls > self.indent_levels[-1]:
                        self.indent(ls)
                        return Indent()
                    elif ls < self.indent_levels[-1]:
                        self.dedent()
                        return Dedent()
                case x:
                    print(x)

        match tok:
            case _:
                return Symbol(tok)

    def indent(self, level):
        self.indent_levels.append(level)

    def dedent(self) -> bool:
        if len(self.indent_levels) == 1:
            return False
        self.indent_levels.pop()
        return True
