import contextlib
import dataclasses
from copy import deepcopy
from typing import Optional

from biunification.type_checker import TypeCheckerCore, Use, Value
from cubiml import abstract_syntax as ast, type_heads


class TypeChecker:
    def __init__(self):
        self.engine = TypeCheckerCore()
        self.bindings = Bindings()

    def check_script(self, script: ast.Script):
        backup = deepcopy(self.engine)

        try:
            result = None
            for statement in script.statements:
                result = check_toplevel(statement, self.bindings, self.engine)
        except Exception:
            # roll back changes
            self.engine = backup
            self.bindings.unwind(0)
            raise

        # persist changes
        self.bindings.changes.clear()
        return result


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
    def __init__(self):
        self.m: dict[str, Value] = {}
        self.changes: list[tuple[str, Optional[Value]]] = []

    def get(self, k: str):
        try:
            return self.m[k]
        except KeyError:
            raise UnboundError(k) from None

    def insert(self, k: str, v: Value):
        old = self.m.get(k)
        self.changes.append((k, old))
        self.m[k] = v

    @contextlib.contextmanager
    def child_scope(self):
        n = len(self.changes)
        try:
            yield self
        finally:
            self.unwind(n)

    def unwind(self, n):
        while len(self.changes) > n:
            k, old = self.changes.pop()
            if old is None:
                del self.m[k]
            else:
                self.m[k] = old


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
            field_types = type_heads.AssocEmpty()
            for name, exp in fields:
                if name in field_names:
                    raise RepeatedFieldNameError(name)
                field_names.add(name)

                t = check_expr(exp, bindings, engine)
                field_types = type_heads.AssocItem(name, t, field_types)
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
            case_types = type_heads.AssocEmpty()
            for arm in arms:
                if arm.tag in case_names:
                    raise RepeatedCaseError(arm.tag)
                case_names.add(arm.tag)

                wrapped_t, wrapped_u = engine.var()
                case_types = type_heads.AssocItem(arm.tag, wrapped_u, case_types)

                with bindings.child_scope() as bindings_:
                    bindings_.insert(arm.var, wrapped_t)
                    rhs_t = check_expr(arm.bdy, bindings_, engine)
                engine.flow(rhs_t, result_u)

            use = engine.new_use(type_heads.UCase(case_types))
            engine.flow(match_t, use)

            return result_t
        case ast.Function(var, body):
            arg_t, arg_u = engine.var()
            with bindings.child_scope() as bindings_:
                bindings_.insert(var, arg_t)
                body_t = check_expr(body, bindings_, engine)
            return engine.new_val(type_heads.VFunc(arg_u, body_t))
        case ast.Application(fun, arg):
            fun_t = check_expr(fun, bindings, engine)
            arg_t = check_expr(arg, bindings, engine)

            ret_t, ret_u = engine.var()
            use = engine.new_use(type_heads.UFunc(arg_t, ret_u))
            engine.flow(fun_t, use)
            return ret_t
        case ast.Let(var, val, body):
            val_t = check_expr(val, bindings, engine)
            with bindings.child_scope() as bindings_:
                bindings_.insert(var, val_t)
                body_t = check_expr(body, bindings_, engine)
            return body_t
        case ast.LetRec(defs, body):
            with bindings.child_scope() as bindings_:
                check_letrec(defs, bindings_, engine)
                return check_expr(body, bindings_, engine)
        case _:
            raise NotImplementedError(expr)


def check_letrec(defs, bindings, engine):
    temp_us = []
    for d in defs:
        temp_t, temp_u = engine.var()
        bindings.insert(d.name, temp_t)
        temp_us.append(temp_u)
    for d, use in zip(defs, temp_us):
        var_t = check_expr(d.fun, bindings, engine)
        engine.flow(var_t, use)


def check_toplevel(
    stmt: ast.ToplevelItem, bindings: Bindings, engine: TypeCheckerCore
) -> Value:
    match stmt:
        case ast.Expression() as expr:
            return check_expr(expr, bindings, engine)
        case ast.DefineLet(var, val):
            var_t = check_expr(val, bindings, engine)
            bindings.insert(var, var_t)
            return var_t
        case ast.DefineLetRec(defs):
            check_letrec(defs, bindings, engine)
            return None
        case _:
            raise NotImplementedError(stmt)
