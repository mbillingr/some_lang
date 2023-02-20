import pytest

from cubiml.scanner import Span
from cubiml.tokenizer import scanner, TokenKind


def test_whitespace():
    assert_token("   \t ", TokenKind.WHITESPACE)


def test_newline():
    assert_token("\n", TokenKind.NEWLINE)


def test_multiple_white_lines_count_as_single_newline():
    assert_token("\n\n", TokenKind.NEWLINE)
    assert_token("\n\t\n", TokenKind.NEWLINE)
    assert_token("\n  \n", TokenKind.NEWLINE)
    assert_token("\n  \n  \n", TokenKind.NEWLINE)
    assert_token("\n  \n\n  \n", TokenKind.NEWLINE)


@pytest.mark.parametrize("kw", ["else", "if"])
def test_keywords(kw):
    assert_token(kw, TokenKind.KEYWORD)


def test_identifier():
    assert_token("_", TokenKind.IDENTIFIER)
    assert_token("foo", TokenKind.IDENTIFIER)
    assert_token("foo-bar", TokenKind.IDENTIFIER)
    assert_token("foo_bar", TokenKind.IDENTIFIER)
    assert_token("FooBar", TokenKind.IDENTIFIER)
    assert_token("foo!?*", TokenKind.IDENTIFIER)


def test_operator():
    assert_token("+", TokenKind.OPERATOR)
    assert_token("++", TokenKind.OPERATOR)
    assert_token("+-*/", TokenKind.OPERATOR)
    assert_token("&", TokenKind.OPERATOR)
    assert_token("::", TokenKind.OPERATOR)
    assert_token(".", TokenKind.OPERATOR)


def test_integer_literal():
    assert_token("0", TokenKind.LITERAL)
    assert_token("012", TokenKind.LITERAL)
    assert_token("-34", TokenKind.LITERAL)
    assert_token("+56", TokenKind.LITERAL)


def test_comment():
    assert_token("#", TokenKind.COMMENT)
    assert_token("# foobar\n", TokenKind.COMMENT)

    assert list(scanner.tokenize("#foo\nx")) == [
        ("#foo\n", TokenKind.COMMENT, Span("#foo\nx", 0, 5)),
        ("x", TokenKind.IDENTIFIER, Span("#foo\nx", 5, 6)),
    ]


def assert_token(src, is_token):
    assert list(scanner.tokenize(src)) == [(src, is_token, Span(src, 0, len(src)))]
