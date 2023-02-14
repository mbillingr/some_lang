import contextlib
import dataclasses
from copy import deepcopy
from typing import Optional, TypeAlias, Callable

from biunification.type_checker import TypeCheckerCore, Use, Value
from cubiml import abstract_syntax as ast, type_heads


class TypeChecker:
    def __init__(self):
        self.engine = TypeCheckerCore()
        self.bindings = Bindings()

    def check_script(self, script: ast.Script):
        backup = deepcopy(self.engine)

        type_map = {}

        def map_type(expr: ast.Expression, ty: Value) -> Value:
            if id(expr) not in type_map:
                type_map[id(expr)] = ty
            return ty

        try:
            for statement in script.statements:
                check_toplevel(statement, self.bindings, self.engine, map_type)
        except Exception:
            # roll back changes
            self.engine = backup
            self.bindings.unwind(0)
            raise

        self.engine.collapse_cycles()

        # persist changes
        self.bindings.changes.clear()
        return type_map


@dataclasses.dataclass
class UnboundError(Exception):
    var: str


@dataclasses.dataclass
class RepeatedFieldNameError(Exception):
    field: str


@dataclasses.dataclass
class RepeatedCaseError(Exception):
    tag: str


Scheme: TypeAlias = Callable[[TypeCheckerCore], Value]


class Bindings:
    def __init__(self):
        self.m: dict[str, Scheme] = {}
        self.changes: list[tuple[str, Optional[Scheme]]] = []

    def get(self, k: str):
        try:
            return self.m[k]
        except KeyError:
            raise UnboundError(k) from None

    def insert(self, k: str, v: Value):
        self.insert_scheme(k, lambda _: v)

    def insert_scheme(self, k: str, v: Scheme):
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
    expr: ast.Expression,
    bindings: Bindings,
    engine: TypeCheckerCore,
    callback=lambda _, t: t,
) -> Value:
    match expr:
        case ast.Literal(bool()):
            return callback(expr, engine.new_val(type_heads.VBool()))
        case ast.Literal(int()):
            return callback(expr, engine.new_val(type_heads.VInt()))
        case ast.Reference(var):
            scheme = bindings.get(var)
            return callback(expr, scheme(engine))
        case ast.BinOp(lhs, rhs, opty, op):
            lhs_t = check_expr(lhs, bindings, engine, callback)
            rhs_t = check_expr(rhs, bindings, engine, callback)
            match opty:
                case "any", "any", "bool":
                    return callback(expr, engine.new_val(type_heads.VBool()))
                case "int", "int", "bool":
                    bound = engine.new_use(type_heads.UInt())
                    engine.flow(lhs_t, bound)
                    engine.flow(rhs_t, bound)
                    return callback(expr, engine.new_val(type_heads.VBool()))
                case "int", "int", "int":
                    bound = engine.new_use(type_heads.UInt())
                    engine.flow(lhs_t, bound)
                    engine.flow(rhs_t, bound)
                    return callback(expr, engine.new_val(type_heads.VInt()))

        case ast.Record(fields):
            field_names = set()
            field_types = type_heads.AssocEmpty()
            for name, exp in fields:
                if name in field_names:
                    raise RepeatedFieldNameError(name)
                field_names.add(name)

                t = check_expr(exp, bindings, engine, callback)
                field_types = type_heads.AssocItem(name, t, field_types)
            return callback(expr, engine.new_val(type_heads.VObj(field_types)))
        case ast.Case(tag, val):
            typ = check_expr(val, bindings, engine, callback)
            return callback(expr, engine.new_val(type_heads.VCase(tag, typ)))
        case ast.Conditional(a, b, c):
            cond_t = check_expr(a, bindings, engine, callback)
            engine.flow(cond_t, engine.new_use(type_heads.UBool()))

            then_t = check_expr(b, bindings, engine, callback)
            else_t = check_expr(c, bindings, engine, callback)

            merged, merge_use = engine.var()
            engine.flow(then_t, merge_use)
            engine.flow(else_t, merge_use)
            return callback(expr, merged)
        case ast.FieldAccess(field, lhs_expr):
            lhs_t = check_expr(lhs_expr, bindings, engine, callback)

            field_t, field_use = engine.var()
            use = engine.new_use(type_heads.UObj(field, field_use))
            engine.flow(lhs_t, use)
            return callback(expr, field_t)
        case ast.Match(match_exp, arms):
            match_t = check_expr(match_exp, bindings, engine, callback)
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
                    rhs_t = check_expr(arm.bdy, bindings_, engine, callback)
                engine.flow(rhs_t, result_u)

            use = engine.new_use(type_heads.UCase(case_types))
            engine.flow(match_t, use)

            return callback(expr, result_t)
        case ast.Function(var, body):
            arg_t, arg_u = engine.var()
            with bindings.child_scope() as bindings_:
                bindings_.insert(var, arg_t)
                body_t = check_expr(body, bindings_, engine, callback)
            return callback(expr, engine.new_val(type_heads.VFunc(arg_u, body_t)))
        case ast.Procedure(var, body):
            arg_t, arg_u = engine.var()
            with bindings.child_scope() as bindings_:
                bindings_.insert(var, arg_t)
                body_t = None
                for bexp in body:
                    body_t = check_expr(bexp, bindings_, engine, callback)
            return callback(expr, engine.new_val(type_heads.VProc(arg_u, body_t)))
        case ast.Application(fun, arg):
            fun_t = check_expr(fun, bindings, engine, callback)
            arg_t = check_expr(arg, bindings, engine, callback)

            ret_t, ret_u = engine.var()
            use = engine.new_use(type_heads.UProc(arg_t, ret_u))
            engine.flow(fun_t, use)
            return callback(expr, ret_t)
        case ast.Let(var, val, body):
            val_scheme = check_let(val, bindings, engine, callback)
            with bindings.child_scope() as bindings_:
                bindings_.insert_scheme(var, val_scheme)
                body_t = check_expr(body, bindings_, engine, callback)
            return callback(expr, body_t)
        case ast.LetRec(defs, body):
            with bindings.child_scope() as bindings_:
                check_letrec(defs, bindings_, engine, callback)
                return callback(expr, check_expr(body, bindings_, engine, callback))
        case ast.NewRef(init):
            val_t = check_expr(init, bindings, engine, callback)
            read, write = engine.var()
            engine.flow(val_t, write)
            return callback(expr, engine.new_val(type_heads.VRef(write, read)))
        case ast.RefGet(ref):
            ref_t = check_expr(ref, bindings, engine, callback)
            cell_type, cell_use = engine.var()
            use = engine.new_use(type_heads.URef(None, cell_use))
            engine.flow(ref_t, use)
            return callback(expr, cell_type)
        case ast.RefSet(ref, val):
            lhs_t = check_expr(ref, bindings, engine, callback)
            rhs_t = check_expr(val, bindings, engine, callback)
            bound = engine.new_use(type_heads.URef(rhs_t, None))
            engine.flow(lhs_t, bound)
            return callback(expr, engine.new_val(type_heads.VNever()))
        case _:
            raise NotImplementedError(expr)


def check_let(
    expr: ast.Expression, bindings: Bindings, engine: TypeCheckerCore, callback
) -> Scheme:

    match expr:
        case ast.Function():
            # function definitions can be polymorphic - create a type scheme
            saved_bindings = Bindings()
            saved_bindings.m = bindings.m.copy()
            saved_expr = expr

            f = lambda eng: check_expr(saved_expr, saved_bindings, eng, callback)
            callback(expr, f(engine))  # check once, in case the var is never referenced
            return f
        case _:
            var_type = callback(expr, check_expr(expr, bindings, engine, callback))
            var_val = engine.types[var_type]
            if isinstance(var_val, type_heads.VNever):
                var_val.check(None)  # raises an error
            return lambda _: var_type


def check_letrec(defs: list[ast.FuncDef], bindings, engine, callback):
    saved_bindings = Bindings()
    saved_bindings.m = bindings.m.copy()
    saved_defs = defs

    def f(eng, i):
        temp_vars = []
        for d in saved_defs:
            temp_t, temp_u = eng.var()
            saved_bindings.insert(d.name, temp_t)
            temp_vars.append((temp_t, temp_u))

        for d, (_, use) in zip(defs, temp_vars):
            var_t = check_expr(d.fun, saved_bindings, eng, callback)
            engine.flow(var_t, use)

        return temp_vars[i][0]

    f(engine, 0)  # check once, in case the var is never referenced

    for i, d in enumerate(defs):
        bindings.insert_scheme(d.name, lambda eng: f(eng, i))


def check_toplevel(
    stmt: ast.ToplevelItem,
    bindings: Bindings,
    engine: TypeCheckerCore,
    callback=lambda _, t: t,
):
    match stmt:
        case ast.Expression() as expr:
            check_expr(expr, bindings, engine, callback)
        case ast.DefineLet(var, val):
            val_scheme = check_let(val, bindings, engine, callback)
            bindings.insert_scheme(var, val_scheme)
        case ast.DefineLetRec(defs):
            check_letrec(defs, bindings, engine, callback)
        case _:
            raise NotImplementedError(stmt)
