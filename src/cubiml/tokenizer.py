import time
from enum import Enum
from typing import TypeAlias, Any, Iterable, Iterator

from cubiml.scanner import ScannerGenerator, Repeat, OneOf, Alternative, Opt, Span

KEYWORDS = ("if", "else")
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
IDENT_SYMS = "+-*/,<>;@$~&%=!?^\\|'\""
OP_SYMBOLS = IDENT_SYMS + ".:"


class TokenKind(Enum):
    WHITESPACE = 0
    NEWLINE = 1
    KEYWORD = 2
    IDENTIFIER = 3
    OPERATOR = 4
    LITERAL = 5
    COMMENT = 6


Token: TypeAlias = tuple[Any, TokenKind, Span]

# scanner postprocessors


def ignore_whitespace(token_stream: Iterable[Token]) -> Iterator[Token]:
    for token in token_stream:
        match token:
            case _, TokenKind.WHITESPACE | TokenKind.NEWLINE, _:
                pass
            case _:
                yield token


# scanner implementation


def whitespace(optional=False):
    return Repeat(OneOf(" \t"), accept_empty=optional)


def ident_first():
    return OneOf(LETTERS.lower() + LETTERS.upper() + "_")


def ident_rest():
    return ident_first() | OneOf(IDENT_SYMS)


def num():
    return Repeat(OneOf(DIGITS), accept_empty=False)


scg = (
    ScannerGenerator()
    .set_token_priority(TokenKind.KEYWORD, TokenKind.IDENTIFIER)
    .add_rule(TokenKind.NEWLINE, "\n" + Repeat(whitespace(optional=True) + "\n"))
    .add_rule(TokenKind.WHITESPACE, whitespace())
    .add_rule(TokenKind.KEYWORD, Alternative(*KEYWORDS))
    .add_rule(TokenKind.IDENTIFIER, ident_first() + Repeat(ident_rest()))
    .add_rule(TokenKind.OPERATOR, Repeat(OneOf(OP_SYMBOLS), accept_empty=False))
    .add_rule(TokenKind.LITERAL, Opt(OneOf("+-")) + num())
)
# this rule needs to know the complete alphabet to implement an "any char" like regex
scg.add_rule(TokenKind.COMMENT, "#" + Repeat(OneOf(scg.alphabet - {"\n"})) + Opt("\n"))

start = time.time()
print("building scanner...")
scanner = scg.build()
print(f"... {time.time() - start:.3f}s")
