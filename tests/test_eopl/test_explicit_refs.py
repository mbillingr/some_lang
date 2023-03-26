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
        # Binding
        (0, "let x = 0 in x"),
        (1, "let x = 0 in let x = 1 in x"),
        (0, "let x = 0 in let y = 1 in x"),
        # References
        (Store.Ref(1), "newref 1"),
    ],
)
def test_literals(src, expect):
    assert evaluate(src) == expect


def evaluate(src):
    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    runner = interpreter.analyze_program(program)
    return runner(Store())
