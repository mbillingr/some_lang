import pytest

from cubiml.scanner import Span
from cubiml.tokenizer import remove_all_whitespace, indentify, scanner, TokenKind


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
    assert_token("0", TokenKind.LITERAL_INT)
    assert_token("012", TokenKind.LITERAL_INT)
    assert_token("-34", TokenKind.LITERAL_INT)
    assert_token("+56", TokenKind.LITERAL_INT)


def test_comment():
    assert_token("#", TokenKind.COMMENT)
    assert_token("# foobar\n", TokenKind.COMMENT)

    assert list(scanner.tokenize("#foo\nx")) == [
        ("#foo\n", TokenKind.COMMENT, Span("#foo\nx", 0, 5)),
        ("x", TokenKind.IDENTIFIER, Span("#foo\nx", 5, 6)),
    ]


def test_whitespace_removal():
    src = " foo \n   bar\t"
    tokenstream = scanner.tokenize(src)

    res = remove_all_whitespace(tokenstream)

    assert list(res) == [
        ("foo", TokenKind.IDENTIFIER, Span(src, 1, 4)),
        ("bar", TokenKind.IDENTIFIER, Span(src, 9, 12)),
    ]


def test_indentation():
    def check(src, expected):
        tokenstream = scanner.tokenize(src)
        res = indentify(tokenstream)
        expected = [(s, t, Span(src, span.start, span.end)) for s, t, span in expected]
        assert list(res) == expected

    # no indent
    check("foo", [("foo", TokenKind.IDENTIFIER, Span("", 0, 3))])

    # standalone indented line
    check(
        "  foo",
        [
            ("  ", TokenKind.INDENT, Span("", 0, 2)),
            ("foo", TokenKind.IDENTIFIER, Span("", 2, 5)),
            ("", TokenKind.DEDENT, Span("", 5, 5)),
        ],
    )

    # newline inbetween
    check(
        "  foo\n\n  bar",
        [
            ("  ", TokenKind.INDENT, Span("", 0, 2)),
            ("foo", TokenKind.IDENTIFIER, Span("", 2, 5)),
            ("bar", TokenKind.IDENTIFIER, Span("", 9, 12)),
            ("", TokenKind.DEDENT, Span("", 12, 12)),
        ],
    )

    # nested indents
    check(
        "foo\n  bar\n  baz\n    fee",
        [
            ("foo", TokenKind.IDENTIFIER, Span("", 0, 3)),
            ("  ", TokenKind.INDENT, Span("", 4, 6)),
            ("bar", TokenKind.IDENTIFIER, Span("", 6, 9)),
            ("baz", TokenKind.IDENTIFIER, Span("", 12, 15)),
            ("    ", TokenKind.INDENT, Span("", 16, 20)),
            ("fee", TokenKind.IDENTIFIER, Span("", 20, 23)),
            ("", TokenKind.DEDENT, Span("", 23, 23)),
            ("", TokenKind.DEDENT, Span("", 23, 23)),
        ],
    )

    # explicit dedent
    check(
        "foo\n  bar\nbaz",
        [
            ("foo", TokenKind.IDENTIFIER, Span("", 0, 3)),
            ("  ", TokenKind.INDENT, Span("", 4, 6)),
            ("bar", TokenKind.IDENTIFIER, Span("", 6, 9)),
            ("", TokenKind.DEDENT, Span("", 10, 10)),
            ("baz", TokenKind.IDENTIFIER, Span("", 10, 13)),
        ],
    )

    # nested dedent
    check(
        "  foo\n    bar\n  baz",
        [
            ("  ", TokenKind.INDENT, Span("", 0, 2)),
            ("foo", TokenKind.IDENTIFIER, Span("", 2, 5)),
            ("    ", TokenKind.INDENT, Span("", 6, 10)),
            ("bar", TokenKind.IDENTIFIER, Span("", 10, 13)),
            ("", TokenKind.DEDENT, Span("", 16, 16)),
            ("baz", TokenKind.IDENTIFIER, Span("", 16, 19)),
            ("", TokenKind.DEDENT, Span("", 19, 19)),
        ],
    )

    # suspended indentation inside parentheses
    check(
        "  foo(\nbar)\n  baz",
        [
            ("  ", TokenKind.INDENT, Span("", 0, 2)),
            ("foo", TokenKind.IDENTIFIER, Span("", 2, 5)),
            ("(", TokenKind.LPAREN, Span("", 5, 6)),
            ("bar", TokenKind.IDENTIFIER, Span("", 7, 10)),
            (")", TokenKind.RPAREN, Span("", 10, 11)),
            ("baz", TokenKind.IDENTIFIER, Span("", 14, 17)),
            ("", TokenKind.DEDENT, Span("", 17, 17)),
        ],
    )

    # invalid dedent
    with pytest.raises(IndentationError):
        check("    foo\n  bar", [])


def assert_token(src, is_token):
    assert list(scanner.tokenize(src)) == [(src, is_token, Span(src, 0, len(src)))]
