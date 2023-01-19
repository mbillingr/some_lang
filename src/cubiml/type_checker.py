import contextlib
import dataclasses
from typing import Optional

from biunification.type_checker import TypeCheckerCore, Value
from cubiml import ast, type_heads


@dataclasses.dataclass
class UnboundError(Exception):
    var: str


@dataclasses.dataclass
class RepeatedFieldNameError(Exception):
    field: str


@dataclasses.dataclass
class RepeatedCaseError(Exception):
    tag: str


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
        case ast.Record(fields):
            field_names = set()
            field_types = {}
            for name, exp in fields:
                if name in field_names:
                    raise RepeatedFieldNameError(name)
                field_names.add(name)

                t = check_expr(exp, bindings, engine)
                field_types[name] = t
            return engine.new_val(type_heads.VObj(field_types))
        case ast.Case(tag, val):
            typ = check_expr(val, bindings, engine)
            return engine.new_val(type_heads.VCase(tag, typ))
        case ast.Conditional(a, b, c):
            cond_t = check_expr(a, bindings, engine)
            engine.flow(cond_t, engine.new_use(type_heads.UBool()))

            then_t = check_expr(b, bindings, engine)
            else_t = check_expr(c, bindings, engine)

            merged, merge_use = engine.var()
            engine.flow(then_t, merge_use)
            engine.flow(else_t, merge_use)
            return merged
        case ast.FieldAccess(field, lhs_expr):
            lhs_t = check_expr(lhs_expr, bindings, engine)

            field_t, field_use = engine.var()
            use = engine.new_use(type_heads.UObj(field, field_use))
            engine.flow(lhs_t, use)
            return field_t
        case ast.Match(match_exp, arms):
            match_t = check_expr(match_exp, bindings, engine)
            result_t, result_u = engine.var()

            case_names = set()
            case_types = {}
            for arm in arms:
                if arm.tag in case_names:
                    raise RepeatedCaseError(arm.tag)
                case_names.add(arm.tag)

                wrapped_t, wrapped_u = engine.var()
                case_types[arm.tag] = wrapped_u

                with bindings.child_scope() as bindings_:
                    bindings_.insert(arm.var, wrapped_t)
                    rhs_t = check_expr(arm.bdy, bindings_, engine)
                engine.flow(rhs_t, result_u)

            use = engine.new_use(type_heads.UCase(case_types))
            engine.flow(match_t, use)

            return result_t
        case _:
            raise NotImplementedError(expr)
