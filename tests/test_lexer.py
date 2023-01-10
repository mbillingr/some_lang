from some_lang import lexer
from some_lang.lexer import Symbol, Indent, Dedent, Whitespace


def test_simple_indent():
    src = "foo\n  bar\nbaz"
    tok = lexer.tokenize(src)
    assert list(tok) == [
        Symbol("foo"),
        Indent(),
        Symbol("bar"),
        Dedent(),
        Symbol("baz"),
    ]


def test_nested_indent():
    src = """a
b
  c
  d
    x
  y
z"""
    tok = lexer.tokenize(src)
    assert list(tok) == [
        Symbol("a"),
        Whitespace(),
        Symbol("b"),
        Indent(),
        Symbol("c"),
        Whitespace(),
        Symbol("d"),
        Indent(),
        Symbol("x"),
        Dedent(),
        Symbol("y"),
        Dedent(),
        Symbol("z"),
    ]


def test_automatic_dedent():
    src = """
  a
    b"""
    tok = lexer.tokenize(src)
    assert list(tok) == [
        Indent(),
        Symbol("a"),
        Indent(),
        Symbol("b"),
        Dedent(),
        Dedent(),
    ]
