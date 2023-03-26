import pytest

from eopl_explicit_refs import tokenizer, parser, interpreter
from eopl_explicit_refs.store import PythonStore


@pytest.mark.parametrize(
    "expect, src",
    [
        (False, "false"),
        (True, "true"),
        (0, "0"),
    ],
)
def test_literals(src, expect):
    assert evaluate(src) == expect


def evaluate(src):
    token_stream = tokenizer.default_tokenizer(src)
    program = parser.parse_program(token_stream)
    runner = interpreter.analyze_program(program)
    return runner(PythonStore())
