import pytest

from some_lang import ast
from some_lang.lang_frontend import Context


def test_function_defs():
    ctx = Context().module(
        "def ident(Int) -> Int:\n" "    ident(x) = x\n" "print (ident 42)"
    )
    assert ctx.env.apply("ident")("ARG") == "ARG"
