import collections
import itertools
import time
from enum import Enum
from typing import TypeAlias, Any, Iterable, Iterator

from cubiml.scanner import ScannerGenerator, Repeat, OneOf, Alternative, Opt, Span

KEYWORDS = ("do", "else", "false", "if", "lambda", "true")
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
    BLOCK_INDENT = 11
    DEDENT = 12
    SYNTAX = 13
    BEGIN_BLOCK = 14
    SEP_BLOCK = 15
    END_BLOCK = 16


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

    def put_back(self, token: Token):
        self.buffer.appendleft(token)

    def __iter__(self):
        return self

    def __next__(self):
        if self.buffer:
            return self.buffer.popleft()
        return next(self.ts)


def default_tokenizer(src: str) -> TokenStream:
    token_stream = scanner.tokenize(src)
    token_stream = whitespace_to_indent(token_stream)
    token_stream = strip_trailing_indents(token_stream)
    token_stream = augment_layout(token_stream)
    token_stream = infer_blocks(token_stream)
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
            case _, TokenKind.WHITESPACE | TokenKind.NEWLINE | TokenKind.INDENT, _:
                pass
            case _:
                yield token


def whitespace_to_indent(token_stream: Iterable[Token]) -> Iterator[Token]:
    newline = True
    for token in token_stream:
        match token:
            case t, TokenKind.WHITESPACE, span:
                if newline:
                    yield indent_len(t), TokenKind.INDENT, span
                newline = False
            case _, TokenKind.NEWLINE, _:
                newline = True
            case _:
                if newline:
                    span = token[2]
                    yield 0, TokenKind.INDENT, Span(span.src, span.start, span.end)
                newline = False
                yield token


def strip_trailing_indents(token_stream: Iterable[Token]) -> Iterator[Token]:
    buffer = []
    for token in token_stream:
        match token:
            case _, TokenKind.INDENT, _:
                buffer.append(token)
            case _:
                yield from buffer
                yield token
                buffer = []


def indent_len(ws: str) -> int:
    return len(ws)


def augment_layout(token_stream: Iterator[Token]) -> Iterator[Token]:
    ts = TokenStream(token_stream)
    for token in ts:
        match token:
            case ":", TokenKind.SYNTAX, span:
                block_indent = 0
                match ts.peek(0):
                    case n, TokenKind.INDENT, span:
                        next(ts)
                        block_indent = n
                    case _, _, span:
                        block_indent = span.column()
                yield block_indent, TokenKind.BLOCK_INDENT, span
            case _:
                yield token


def infer_blocks(ts: TokenStream) -> Iterator[Token]:
    layout_context = []
    while True:
        try:
            tok = next(ts)
        except StopIteration:
            for _ in layout_context:
                yield None, TokenKind.END_BLOCK, "EOF-span"
            return

        match tok, layout_context:
            case (n, TokenKind.INDENT, span), [m, *_] if m == n:
                yield None, TokenKind.SEP_BLOCK, span
            case (n, TokenKind.INDENT, span) as t, [m, *ms] if n < m:
                yield None, TokenKind.END_BLOCK, span
                ts, layout_context = itertools.chain([t], ts), ms
            case (_, TokenKind.INDENT, _), _:
                pass
            case (n, TokenKind.BLOCK_INDENT, span), [m, *ms] if n > m:
                yield None, TokenKind.BEGIN_BLOCK, span
                layout_context = [n, m, *ms]
            case (n, TokenKind.BLOCK_INDENT, span), [] if n > 0:
                yield None, TokenKind.BEGIN_BLOCK, span
                layout_context = [n]
            case (n, TokenKind.BLOCK_INDENT, span), _:
                yield None, TokenKind.BEGIN_BLOCK, span
                yield None, TokenKind.END_BLOCK, span
                ts = itertools.chain([(n, TokenKind.INDENT, span)], ts)
            case t, _:
                yield t
            case _:
                raise LayoutError(ts, layout_context)


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


class ParseError(Exception):
    pass


class LayoutError(Exception):
    pass


class UnexpectedEnd(ParseError):
    pass


class UnexpectedToken(ParseError):
    pass
