from some_lang import parsing
from some_lang.lexer import Symbol
from some_lang.parsing import ParseResult


def test_parse_delimited_list():
    assert parsing.parse_delimited_nonempty_list("x", ",").parse(
        [Symbol("x")]
    ) == ParseResult([Symbol("x"), []], [])

    assert parsing.parse_delimited_nonempty_list("x", ",").parse(
        [Symbol("x"), Symbol(","), Symbol("x")]
    ) == ParseResult([Symbol("x"), [Symbol("x")]], [])

    assert parsing.parse_delimited_nonempty_list("x", ",").parse(
        [Symbol("x"), Symbol(","), Symbol("x"), Symbol(","), Symbol("x")]
    ) == ParseResult([Symbol("x"), [Symbol("x"), Symbol("x")]], [])
