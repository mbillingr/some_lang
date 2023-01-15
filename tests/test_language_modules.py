import pytest

from some_lang import ast
from some_lang.lang_frontend import Context


def test_function_defs():
    ctx = Context().module("def ident(?) -> ?:\n    ident(x) = x\nprint (ident 42)")
    assert ctx.env.apply("ident")("ARG") == "ARG"


def test_polymorphic_function_not_yet_supported():
    src = """
def ident(?) -> ?:
    ident(x) = x
    
print (inc (ident 1))
print (not (ident true))
"""
    ctx = Context()
    ctx.define("not", lambda x: not x, "Bool -> Bool")
    ctx.define("inc", lambda x: x + 1, "Int -> Int")

    with pytest.raises(TypeError):
        ctx.module(src)
