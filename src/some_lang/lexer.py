import dataclasses
import re
from typing import Iterator, Callable

from some_lang import parsing
from some_lang.parsing import Token

DELIMITERS = re.compile(r"(\s+|[:()[])")


def skip_all_whitespace(tokens: Iterator[Token]) -> Iterator[Token]:
    return make_token_skipper(Whitespace)(tokens)


def make_token_skipper(*to_skip: type) -> Callable:
    def skipper(tokens: Iterator[Token]) -> Iterator[Token]:
        return filter(lambda t: not isinstance(t, to_skip), tokens)

    return skipper


@dataclasses.dataclass
class Bool(Token):
    value: bool


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


@parsing.ensure_parser.register
def _(obj: str) -> parsing.Parser:
    return parsing.Exact(Symbol(obj))


def tokenize(src) -> Iterator[Token]:
    tm = TokenMatcher()
    for tokens in map(tm.match_token, filter(lambda x: x, DELIMITERS.split(src))):
        yield from tokens
    while tm.dedent():
        yield Dedent()


class TokenMatcher:
    def __init__(self):
        self.indent_levels = [0]

    def match_token(self, tok: str) -> list[Token]:
        match tok:
            case "true":
                return [Bool(True)]
            case "false":
                return [Bool(False)]

        try:
            return [Int(int(tok))]
        except ValueError:
            pass

        if tok.isspace():
            match tok.rsplit("\n", 1):
                case [_]:
                    return [Whitespace()]
                case [_, s]:
                    ls = len(s)
                    if ls == self.indent_levels[-1]:
                        return [Whitespace()]

                    if ls > self.indent_levels[-1]:
                        self.indent(ls)
                        return [Indent()]
                    out: list[Token] = []
                    while ls < self.indent_levels[-1]:
                        self.dedent()
                        out.append(Dedent())
                    if ls != self.indent_levels[-1]:
                        raise IndentationError()
                    return out
                case x:
                    print(x)

        match tok:
            case _:
                return [Symbol(tok)]

    def indent(self, level):
        self.indent_levels.append(level)

    def dedent(self) -> bool:
        if len(self.indent_levels) == 1:
            return False
        self.indent_levels.pop()
        return True
