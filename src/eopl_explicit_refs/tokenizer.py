import collections
import os
import time
from enum import Enum
from pathlib import Path
from typing import TypeAlias, Any, Iterable, Iterator

from parsing.scanner_generator import (
    ScannerGenerator,
    Repeat,
    OneOf,
    Alternative,
    Opt,
    Span,
    Scanner,
)

KEYWORDS = (
    ";",
    "|",
    "class",
    "deref",
    "else",
    "false",
    "if",
    "in",
    "let",
    "newref",
    "set",
    "then",
    "true",
)
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
IDENT_SYMS = "+-*/,<>@$~&%=!?^\\'\""
OP_SYMBOLS = IDENT_SYMS + ".:"
LPARENS = "([{"
RPARENS = ")]}"


class TokenKind(str, Enum):
    WHITESPACE = "whitespace"
    KEYWORD = "keyword"
    IDENTIFIER = "identifier"
    OPERATOR = "operator"
    LPAREN = "lparen"
    RPAREN = "rparen"
    LITERAL_BOOL = "literal-bool"
    LITERAL_INT = "literal-int"
    COMMENT = "comment"


Token: TypeAlias = tuple[Any, TokenKind, Span]


class TokenStream:
    EOF = object()

    def __init__(self, ts: Iterable[Token]):
        self.ts = iter(ts)
        self.buffer = collections.deque()

    def peek(self, n=0):
        try:
            while n >= len(self.buffer):
                self.buffer.append(next(self.ts))
            return self.buffer[n]
        except StopIteration:
            return self.EOF

    def get_next(self):
        try:
            return next(self)
        except StopIteration:
            return self.EOF

    def insert(self, token: Token):
        self.buffer.appendleft(token)

    def append(self, token: Token):
        self.buffer.append(token)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.popleft()
        return next(self.ts)


def default_tokenizer(src: str) -> TokenStream:
    token_stream = scanner.tokenize(src)
    # token_stream = inspect(token_stream)
    token_stream = remove_all_whitespace(token_stream)
    token_stream = transform_literals(token_stream)
    token_stream = TokenStream(token_stream)
    return token_stream


def inspect(ts):
    ts = list(ts)
    for t in ts:
        print(t)
    yield from ts


# scanner postprocessors


def transform_literals(token_stream: Iterable[Token]) -> Iterator[Token]:
    for token in token_stream:
        match token:
            case val, TokenKind.LITERAL_INT as tok, span:
                yield int(val), tok, span
            case "true", TokenKind.KEYWORD, span:
                yield True, TokenKind.LITERAL_BOOL, span
            case "false", TokenKind.KEYWORD, span:
                yield False, TokenKind.LITERAL_BOOL, span
            case _:
                yield token


def remove_all_whitespace(token_stream: Iterable[Token]) -> Iterator[Token]:
    for token in token_stream:
        match token:
            case _, TokenKind.WHITESPACE, _:
                pass
            case _:
                yield token


# scanner implementation


def whitespace(optional=False):
    return Repeat(OneOf(" \t\n"), accept_empty=optional)


def ident_first():
    return OneOf(LETTERS.lower() + LETTERS.upper() + "_")


def ident_rest():
    return ident_first() | OneOf(IDENT_SYMS)


def num():
    return Repeat(OneOf(DIGITS), accept_empty=False)


scg = (
    ScannerGenerator()
    .set_token_priority(TokenKind.KEYWORD, TokenKind.IDENTIFIER)
    .add_rule(TokenKind.WHITESPACE, whitespace())
    .add_rule(TokenKind.KEYWORD, Alternative(*KEYWORDS))
    .add_rule(TokenKind.IDENTIFIER, ident_first() + Repeat(ident_rest()))
    .add_rule(TokenKind.OPERATOR, Repeat(OneOf(OP_SYMBOLS), accept_empty=False))
    .add_rule(TokenKind.LPAREN, OneOf(LPARENS))
    .add_rule(TokenKind.RPAREN, OneOf(RPARENS))
    .add_rule(TokenKind.LITERAL_INT, Opt(OneOf("+-")) + num())
)
# this rule needs to know the complete alphabet to implement an "any char" like regex
scg.add_rule(TokenKind.COMMENT, "#" + Repeat(OneOf(scg.alphabet - {"\n"})) + Opt("\n"))

if os.getenv("REBUILD_SCANNER"):
    print("building scanner...")
    start = time.time()
    scanner = scg.build()
    print(f"... {time.time() - start:.3f}s")
    scanner.store(Path(__file__).parent / "scanner.json")
else:
    scanner = Scanner.load(TokenKind, Path(__file__).parent / "scanner.json")


class ParseError(Exception):
    pass


class UnexpectedEnd(ParseError):
    pass


class UnexpectedToken(ParseError):
    def __str__(self):
        tok, kind, span = self.args[0]
        return f"Unexpected {kind.name} '{tok}'\n" + span.show_line()
