from __future__ import annotations

import contextlib
import dataclasses
import enum
from copy import deepcopy
from typing import Optional, TypeAlias, Callable

from biunification.type_checker import TypeCheckerCore, Use, Value
from cubiml import abstract_syntax as ast, type_heads


class TypeChecker:
    def __init__(self):
        self.ctx = Context.default()

    def check_script(self, script: ast.Script):
        backup = deepcopy(self.ctx.engine)

        type_map = {}

        def map_type(expr: ast.Expression, ty: Value) -> Value:
            if id(expr) not in type_map:
                type_map[id(expr)] = ty
            return ty

        try:
            for statement in script.statements:
                check_toplevel(statement, self.ctx.with_callback(map_type))
        except Exception:
            # roll back changes
            self.ctx.engine = backup
            self.ctx.bindings.unwind(0)
            raise

        self.ctx.engine.collapse_cycles()

        # persist changes
        self.ctx.bindings.changes.clear()
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


class Level(enum.Enum):
    TOP = "toplevel"
    FUNC = "function"
    PROC = "procedure"


@dataclasses.dataclass
class Context:
    bindings: Bindings
    engine: TypeCheckerCore
    callback: Callable
    level: Level

    @staticmethod
    def default() -> Context:
        return Context(Bindings(), TypeCheckerCore(), lambda _, x: x, Level.TOP)

    def with_bindings(self, bindings: Bindings) -> Context:
        return Context(bindings, self.engine, self.callback, self.level)

    def with_engine(self, engine: TypeCheckerCore) -> Context:
        return Context(self.bindings, engine, self.callback, self.level)

    def with_callback(self, callback: Callable) -> Context:
        return Context(self.bindings, self.engine, callback, self.level)

    def with_level(self, level: Level) -> Context:
        return Context(self.bindings, self.engine, self.callback, level)


def check_expr(expr: ast.Expression, ctx: Context) -> Value:
    match expr:
        case ast.Literal(bool()):
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VBool()))
        case ast.Literal(int()):
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VInt()))
        case ast.Reference(var):
            scheme = ctx.bindings.get(var)
            return ctx.callback(expr, scheme(ctx.engine))
        case ast.BinOp(lhs, rhs, opty, op):
            lhs_t = check_expr(lhs, ctx)
            rhs_t = check_expr(rhs, ctx)
            match opty:
                case "any", "any", "bool":
                    return ctx.callback(expr, ctx.engine.new_val(type_heads.VBool()))
                case "int", "int", "bool":
                    bound = ctx.engine.new_use(type_heads.UInt())
                    ctx.engine.flow(lhs_t, bound)
                    ctx.engine.flow(rhs_t, bound)
                    return ctx.callback(expr, ctx.engine.new_val(type_heads.VBool()))
                case "int", "int", "int":
                    bound = ctx.engine.new_use(type_heads.UInt())
                    ctx.engine.flow(lhs_t, bound)
                    ctx.engine.flow(rhs_t, bound)
                    return ctx.callback(expr, ctx.engine.new_val(type_heads.VInt()))

        case ast.Record(fields):
            field_names = set()
            field_types = type_heads.AssocEmpty()
            for name, exp in fields:
                if name in field_names:
                    raise RepeatedFieldNameError(name)
                field_names.add(name)

                t = check_expr(exp, ctx)
                field_types = type_heads.AssocItem(name, t, field_types)
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VObj(field_types)))
        case ast.Case(tag, val):
            typ = check_expr(val, ctx)
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VCase(tag, typ)))
        case ast.Conditional(a, b, c):
            cond_t = check_expr(a, ctx)
            ctx.engine.flow(cond_t, ctx.engine.new_use(type_heads.UBool()))

            then_t = check_expr(b, ctx)
            else_t = check_expr(c, ctx)

            merged, merge_use = ctx.engine.var()
            ctx.engine.flow(then_t, merge_use)
            ctx.engine.flow(else_t, merge_use)
            return ctx.callback(expr, merged)
        case ast.FieldAccess(field, lhs_expr):
            lhs_t = check_expr(lhs_expr, ctx)

            field_t, field_use = ctx.engine.var()
            use = ctx.engine.new_use(type_heads.UObj(field, field_use))
            ctx.engine.flow(lhs_t, use)
            return ctx.callback(expr, field_t)
        case ast.Match(match_exp, arms):
            match_t = check_expr(match_exp, ctx)
            result_t, result_u = ctx.engine.var()

            case_names = set()
            case_types = type_heads.AssocEmpty()
            for arm in arms:
                if arm.tag in case_names:
                    raise RepeatedCaseError(arm.tag)
                case_names.add(arm.tag)

                wrapped_t, wrapped_u = ctx.engine.var()
                case_types = type_heads.AssocItem(arm.tag, wrapped_u, case_types)

                with ctx.bindings.child_scope() as bindings_:
                    bindings_.insert(arm.var, wrapped_t)
                    rhs_t = check_expr(arm.bdy, ctx.with_bindings(bindings_))
                ctx.engine.flow(rhs_t, result_u)

            use = ctx.engine.new_use(type_heads.UCase(case_types))
            ctx.engine.flow(match_t, use)

            return ctx.callback(expr, result_t)
        case ast.Function(var, body):
            arg_t, arg_u = ctx.engine.var()
            with ctx.bindings.child_scope() as bindings_:
                bindings_.insert(var, arg_t)
                body_t = check_expr(
                    body, ctx.with_bindings(bindings_).with_level(Level.FUNC)
                )
            return ctx.callback(
                expr, ctx.engine.new_val(type_heads.VFunc(arg_u, body_t))
            )
        case ast.Procedure(var, body):
            arg_t, arg_u = ctx.engine.var()
            with ctx.bindings.child_scope() as bindings_:
                bindings_.insert(var, arg_t)
                body_t = None
                for bexp in body:
                    body_t = check_expr(
                        bexp, ctx.with_bindings(bindings_).with_level(Level.PROC)
                    )
            return ctx.callback(
                expr, ctx.engine.new_val(type_heads.VProc(arg_u, body_t))
            )
        case ast.Application(fun, arg):
            fun_t = check_expr(fun, ctx)
            arg_t = check_expr(arg, ctx)

            ret_t, ret_u = ctx.engine.var()
            match ctx.level:
                case Level.FUNC:
                    uty = type_heads.UFunc(arg_t, ret_u)
                case Level.PROC | Level.TOP:
                    uty = type_heads.UProc(arg_t, ret_u)
                case _:
                    raise NotImplementedError(ctx.level)
            use = ctx.engine.new_use(uty)
            ctx.engine.flow(fun_t, use)
            return ctx.callback(expr, ret_t)
        case ast.Let(var, val, body):
            val_scheme = check_let(val, ctx)
            with ctx.bindings.child_scope() as bindings_:
                bindings_.insert_scheme(var, val_scheme)
                body_t = check_expr(body, ctx.with_bindings(bindings_))
            return ctx.callback(expr, body_t)
        case ast.LetRec(defs, body):
            with ctx.bindings.child_scope() as bindings_:
                check_letrec(defs, ctx.with_bindings(bindings_))
                return ctx.callback(
                    expr, check_expr(body, ctx.with_bindings(bindings_))
                )
        case ast.NewRef(init):
            val_t = check_expr(init, ctx)
            read, write = ctx.engine.var()
            ctx.engine.flow(val_t, write)
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VRef(write, read)))
        case ast.RefGet(ref):
            ref_t = check_expr(ref, ctx)
            cell_type, cell_use = ctx.engine.var()
            use = ctx.engine.new_use(type_heads.URef(None, cell_use))
            ctx.engine.flow(ref_t, use)
            return ctx.callback(expr, cell_type)
        case ast.RefSet(ref, val):
            lhs_t = check_expr(ref, ctx)
            rhs_t = check_expr(val, ctx)
            bound = ctx.engine.new_use(type_heads.URef(rhs_t, None))
            ctx.engine.flow(lhs_t, bound)
            return ctx.callback(expr, ctx.engine.new_val(type_heads.VNever()))
        case _:
            raise NotImplementedError(expr)


def check_let(expr: ast.Expression, ctx: Context) -> Scheme:

    match expr:
        case ast.Function():
            # function definitions can be polymorphic - create a type scheme
            saved_bindings = Bindings()
            saved_bindings.m = ctx.bindings.m.copy()
            saved_ctx = ctx.with_bindings(saved_bindings)
            saved_expr = expr

            f = lambda eng: check_expr(
                saved_expr, saved_ctx.with_engine(eng).with_level(Level.FUNC)
            )
            ctx.callback(
                expr, f(ctx.engine)
            )  # check once, in case the var is never referenced
            return f
        case _:
            var_type = ctx.callback(expr, check_expr(expr, ctx))
            var_val = ctx.engine.types[var_type]
            if isinstance(var_val, type_heads.VNever):
                var_val.check(None)  # raises an error
            return lambda _: var_type


def check_letrec(defs: list[ast.FuncDef], ctx: Context):
    saved_bindings = Bindings()
    saved_bindings.m = ctx.bindings.m.copy()
    saved_ctx = ctx.with_bindings(saved_bindings)
    saved_defs = defs

    def f(eng, i):
        temp_vars = []
        for d in saved_defs:
            temp_t, temp_u = eng.var()
            saved_ctx.bindings.insert(d.name, temp_t)
            temp_vars.append((temp_t, temp_u))

        for d, (_, use) in zip(defs, temp_vars):
            var_t = check_expr(d.fun, saved_ctx.with_engine(eng))
            ctx.engine.flow(var_t, use)

        return temp_vars[i][0]

    f(ctx.engine, 0)  # check once, in case the var is never referenced

    for i, d in enumerate(defs):
        ctx.bindings.insert_scheme(d.name, lambda eng: f(eng, i))


def check_toplevel(
    stmt: ast.ToplevelItem,
    ctx: Context,
):
    match stmt:
        case ast.Expression() as expr:
            check_expr(expr, ctx)
        case ast.DefineLet(var, val):
            val_scheme = check_let(val, ctx)
            ctx.bindings.insert_scheme(var, val_scheme)
        case ast.DefineLetRec(defs):
            check_letrec(defs, ctx)
        case _:
            raise NotImplementedError(stmt)
