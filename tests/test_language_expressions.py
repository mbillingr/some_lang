import pytest

from some_lang import ast
from some_lang.lang_frontend import Context


def test_literal_expressions():
    ctx = Context()
    assert ctx.eval("true") is True
    assert ctx.eval("false") is False
    assert ctx.eval("0") == 0
    assert ctx.eval("1") == 1
    assert ctx.eval("123") == 123
    assert ctx.eval("-42") == -42


def test_function_application():
    ctx = Context()
    ctx.define(
        "inc", lambda x: x + 1, ast.FunctionType(ast.IntegerType(), ast.IntegerType())
    )

    assert ctx.eval("(inc 0)") == 1
    assert ctx.eval("(inc (inc 1))") == 3

    with pytest.raises(TypeError):
        assert ctx.eval("(inc false)")


def test_anonymous_functions():
    ctx = Context()
    py_func = ctx.eval("(lambda (x) x)")
    assert py_func("abc") == "abc"

    assert ctx.eval("((lambda (x) x) 7)") == 7
