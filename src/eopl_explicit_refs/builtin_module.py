from functools import partial
from typing import Any

from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs import type_impls as t


def make_builtins() -> ast.NativeModule:
    c = partial(map_keys, compose(partial(qualify, "builtin"), ast.Symbol))
    return ast.NativeModule(
        name=ast.Symbol("builtin"),
        funcs=c({"print": ast.NativeFunction(1, print)}),
        fsigs=c({"print": t.FuncType(t.AnyType(), t.NullType())}),
    )


def qualify(prefix: str, name: str) -> str:
    return f"{prefix}.{name}"


def map_keys(f, d):
    return {f(k): v for k, v in d.items()}


def compose(f, g):
    return lambda x: g(f(x))
