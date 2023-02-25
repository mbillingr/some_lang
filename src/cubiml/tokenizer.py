import collections
import time
from enum import Enum
from typing import TypeAlias, Any, Iterable, Iterator

from cubiml.scanner import ScannerGenerator, Repeat, OneOf, Alternative, Opt, Span

KEYWORDS = ("else", "false", "if", "true")
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
IDENT_SYMS = "+-*/,<>;@$~&%=!?^\\|'\""
OP_SYMBOLS = IDENT_SYMS + ".:"
SYNTAX = ":"
LPARENS = "([{"
RPARENS = ")]}"


class TokenKind(Enum):
    SPECIAL = -1  # not part of teh syntax, used during processing
    WHITESPACE = 0
    NEWLINE = 1
    KEYWORD = 2
    IDENTIFIER = 3
    OPERATOR = 4
    LPAREN = 5
    RPAREN = 6
    LITERAL_BOOL = 7
    LITERAL_INT = 8
    COMMENT = 9
    INDENT = 10
    DEDENT = 11
    SYNTAX = 12


Token: TypeAlias = tuple[Any, TokenKind, Span]
TokenStream: TypeAlias = Iterable[Token]


class PeekableTokenStream:
    EOF = object()

    def __init__(self, ts: TokenStream):
        self.ts = ts
        self.buffer = collections.deque()

    def peek(self, n=0):
        while n >= len(self.buffer):
            self.buffer.append(self._next_from_stream())
        return self.buffer[n]

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.popleft()
        return self._next_from_stream()

    def _next_from_stream(self):
        try:
            return next(self.ts)
        except StopIteration:
            return self.EOF


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
            case "true", TokenKind.KEYWORD, span:
                yield True, TokenKind.LITERAL_BOOL, span
            case "false", TokenKind.KEYWORD, span:
                yield False, TokenKind.LITERAL_BOOL, span
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
    def track_parentheses(token):
        nonlocal parenthesis_level
        match token:
            case _, TokenKind.LPAREN, _:
                parenthesis_level += 1
            case _, TokenKind.RPAREN, _:
                parenthesis_level -= 1
        return token

    parenthesis_level = 0

    token_stream = map(track_parentheses, token_stream)
    current_indent = [""]

    try:
        while True:
            # start of line
            token = next(token_stream)
            match token:
                case _, TokenKind.NEWLINE, _:
                    continue
                case s, TokenKind.WHITESPACE, span if parenthesis_level == 0:
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
                    token = next(token_stream)
                case _, _, span if parenthesis_level == 0:
                    while len(current_indent) > 1:
                        current_indent.pop()
                        yield "", TokenKind.DEDENT, Span(
                            span.src, span.start, span.start
                        )

            # rest of line
            while True:
                if _is_newline(token):
                    if parenthesis_level == 0:
                        break
                else:
                    yield token
                token = next(token_stream)

    except StopIteration:
        pass

    span = token[2]
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
    .set_token_priority(TokenKind.SYNTAX, TokenKind.OPERATOR)
    .set_token_priority(TokenKind.KEYWORD, TokenKind.IDENTIFIER)
    .add_rule(TokenKind.NEWLINE, "\n" + Repeat(whitespace(optional=True) + "\n"))
    .add_rule(TokenKind.WHITESPACE, whitespace())
    .add_rule(TokenKind.KEYWORD, Alternative(*KEYWORDS))
    .add_rule(TokenKind.IDENTIFIER, ident_first() + Repeat(ident_rest()))
    .add_rule(TokenKind.OPERATOR, Repeat(OneOf(OP_SYMBOLS), accept_empty=False))
    .add_rule(TokenKind.LPAREN, OneOf(LPARENS))
    .add_rule(TokenKind.RPAREN, OneOf(RPARENS))
    .add_rule(TokenKind.SYNTAX, OneOf(SYNTAX))
    .add_rule(TokenKind.LITERAL_INT, Opt(OneOf("+-")) + num())
)
# this rule needs to know the complete alphabet to implement an "any char" like regex
scg.add_rule(TokenKind.COMMENT, "#" + Repeat(OneOf(scg.alphabet - {"\n"})) + Opt("\n"))

start = time.time()
print("building scanner...")
scanner = scg.build()
print(f"... {time.time() - start:.3f}s")
