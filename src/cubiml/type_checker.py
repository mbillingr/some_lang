import contextlib
import dataclasses
from typing import Optional

from biunification.type_checker import TypeCheckerCore, Value
from cubiml import ast, type_heads


@dataclasses.dataclass
class UnboundError(Exception):
    var: str


class Bindings:
    def __init__(self, m: Optional[dict[str, Value]] = None):
        self.m: dict[str, Value] = m or {}

    def get(self, k: str):
        try:
            return self.m[k]
        except KeyError:
            raise UnboundError(k) from None

    def insert(self, k: str, v: Value):
        self.m[k] = v

    @contextlib.contextmanager
    def child_scope(self):
        child_scope = Bindings(self.m.copy())
        yield child_scope


def check_expr(
    expr: ast.Expression, bindings: Bindings, engine: TypeCheckerCore
) -> Value:
    match expr:
        case ast.Literal(bool()):
            return engine.new_val(type_heads.VBool())
        case ast.Reference(var):
            return bindings.get(var)
        case _:
            raise NotImplementedError(expr)
