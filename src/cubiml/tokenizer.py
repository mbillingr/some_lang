import collections
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
    LITERAL_INT = 5
    COMMENT = 6
    INDENT = 7
    DEDENT = 8


Token: TypeAlias = tuple[Any, TokenKind, Span]
TokenStream: TypeAlias = Iterable[Token]


class PeekableTokenStream:
    def __init__(self, ts: TokenStream):
        self.ts = ts
        self.buffer = collections.deque()

    def peek(self, n=0):
        while n >= len(self.buffer):
            self.buffer.append(next(self.ts))
        return self.buffer[n]

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.popleft()
        return next(self.ts)


def default_tokenizer(src: str) -> PeekableTokenStream:
    token_stream = scanner.tokenize(src)
    token_stream = indentify(token_stream)
    token_stream = ignore_whitespace(token_stream)
    token_stream = transform_literals(token_stream)
    token_stream = PeekableTokenStream(token_stream)
    return token_stream


# scanner postprocessors


def transform_literals(token_stream: Iterable[Token]) -> Iterator[Token]:
    for token in token_stream:
        match token:
            case val, TokenKind.LITERAL_INT as tok, span:
                yield int(val), tok, span
            case _:
                yield token


def ignore_whitespace(token_stream: Iterable[Token]) -> Iterator[Token]:
    for token in token_stream:
        match token:
            case _, TokenKind.WHITESPACE | TokenKind.NEWLINE, _:
                pass
            case _:
                yield token


def indentify(token_stream: Iterable[Token]) -> Iterator[Token]:
    token_stream = iter(token_stream)
    current_indent = [""]

    try:
        while True:
            # start of line
            next_token = next(token_stream)
            match next_token:
                case _, TokenKind.NEWLINE, _:
                    continue
                case s, TokenKind.WHITESPACE, span:
                    if s == current_indent[-1]:
                        pass  # no change in indent
                    elif s.startswith(current_indent[-1]):
                        current_indent.append(s)
                        yield s, TokenKind.INDENT, span
                    else:
                        while current_indent[-1] != s:
                            current_indent.pop()
                            if not current_indent:
                                raise IndentationError(span)
                            yield "", TokenKind.DEDENT, Span(
                                span.src, span.end, span.end
                            )
                    next_token = next(token_stream)
                case _, _, span:
                    while len(current_indent) > 1:
                        current_indent.pop()
                        yield "", TokenKind.DEDENT, Span(
                            span.src, span.start, span.start
                        )

            while not _is_newline(next_token):
                yield next_token
                next_token = next(token_stream)

    except StopIteration:
        pass

    span = next_token[2]
    while len(current_indent) > 1:
        current_indent.pop()
        yield "", TokenKind.DEDENT, Span(span.src, span.end, span.end)


def _is_newline(t: Token) -> bool:
    return t[1] == TokenKind.NEWLINE


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
    .add_rule(TokenKind.LITERAL_INT, Opt(OneOf("+-")) + num())
)
# this rule needs to know the complete alphabet to implement an "any char" like regex
scg.add_rule(TokenKind.COMMENT, "#" + Repeat(OneOf(scg.alphabet - {"\n"})) + Opt("\n"))

start = time.time()
print("building scanner...")
scanner = scg.build()
print(f"... {time.time() - start:.3f}s")
