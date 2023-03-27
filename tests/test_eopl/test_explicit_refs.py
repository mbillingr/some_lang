import pytest

from eopl_explicit_refs import tokenizer, parser, interpreter
from eopl_explicit_refs.store import PythonStore as Store


@pytest.mark.parametrize(
    "expect, src",
    [
        # Literals
        (False, "false"),
        (True, "true"),
        (0, "0"),
        (42, "42"),
        (-123, "-123"),
        # Sequence
        (0, "begin 0"),
        (3, "begin 1; 2; 3"),
        # Binding
        (0, "let x = 0 in x"),
        (1, "let x = 0 in let x = 1 in x"),
        (0, "let x = 0 in let y = 1 in x"),
        (0, "let x = 0 in 1; 2; x"),
        # References
        (Store.Ref(1), "newref 1"),
        (2, "let x = newref 2 in deref x"),
        (42, "let x = newref 0 in set x = 42; deref x"),
        # Anonymous Functions
        (0, "let foo = fn x => x in foo 0"),
    ],
)
def test_literals(src, expect):
    assert evaluate(src) == expect


def evaluate(src):
    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    runner = interpreter.analyze_program(program)
    return runner(Store())
