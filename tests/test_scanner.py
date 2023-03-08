import pytest

from parsing.scanner_generator import Scanner, ScannerError, Span


def test_empty_scanner_empty_input():
    sc = Scanner(accept={}, transitions={})
    assert list(sc.tokenize("")) == []


def test_fail_on_unexpected_token():
    sc = Scanner(accept={}, transitions={})
    with pytest.raises(ScannerError, match="`x`"):
        list(sc.tokenize("x"))


def test_scan_empty_input():
    sc = Scanner(accept={1: "OK"}, transitions={(0, "a"): 1})
    assert list(sc.tokenize("")) == []


def test_scan_token():
    sc = Scanner(accept={1: "OK"}, transitions={(0, "a"): 1})
    assert list(sc.tokenize("aa")) == [
        ("a", "OK", Span("aa", 0, 1)),
        ("a", "OK", Span("aa", 1, 2)),
    ]


def test_prefer_longest_match():
    sc = Scanner(accept={1: "OK"}, transitions={(0, "a"): 1, (1, "a"): 1})
    assert list(sc.tokenize("aa")) == [("aa", "OK", Span("aa", 0, 2))]


def test_rollback():
    sc = Scanner(accept={1: "OK"}, transitions={(0, "a"): 1, (1, "b"): 2, (2, "c"): 1})
    assert list(sc.tokenize("a")) == [("a", "OK", Span("a", 0, 1))]
    assert list(sc.tokenize("abc")) == [("abc", "OK", Span("abc", 0, 3))]
    assert list(sc.tokenize("abcbc")) == [("abcbc", "OK", Span("abcbc", 0, 5))]

    tokens = sc.tokenize("abb")
    assert next(tokens) == ("a", "OK", Span("abb", 0, 1))
    with pytest.raises(ScannerError):
        next(tokens)

    tokens = sc.tokenize("ab")
    assert next(tokens) == ("a", "OK", Span("ab", 0, 1))
    with pytest.raises(ScannerError):
        next(tokens)

    tokens = sc.tokenize("abcb")
    assert next(tokens) == ("abc", "OK", Span("abcb", 0, 3))
    with pytest.raises(ScannerError):
        next(tokens)
