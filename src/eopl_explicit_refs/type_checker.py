import dataclasses
from typing import TypeAlias, Optional, Any

from eopl_explicit_refs import abstract_syntax as ast
from eopl_explicit_refs.generic_environment import Env, EmptyEnv
from eopl_explicit_refs.type_impls import Type
from eopl_explicit_refs import type_impls as t

TEnv: TypeAlias = Env[Type]


@dataclasses.dataclass
class Context:
    env: TEnv = EmptyEnv()
    interfaces: Env[t.InterfaceType] = EmptyEnv()
    interface_impls: set[tuple[Type, t.InterfaceType]] = dataclasses.field(
        default_factory=set
    )
    types: TEnv = EmptyEnv()
    methods: dict[tuple[Type, str], tuple[Type, int]] = dataclasses.field(
        default_factory=dict
    )
    virtuals: dict[tuple[t.InterfaceType, str], tuple[int, int]] = dataclasses.field(
        default_factory=dict
    )
    vtables: dict[Type, Any] = dataclasses.field(default_factory=dict)

    def check(self, actual_t: Type, expected_t: Type) -> bool:
        match actual_t, expected_t:
            case _, t.InterfaceType():
                return (actual_t, expected_t) in self.interface_impls
            case _, _:
                return actual_t == expected_t

    def extend_env(self, var: str, val: Type):
        return Context(
            env=self.env.extend(var, val),
            interfaces=self.interfaces,
            interface_impls=self.interface_impls,
            types=self.types,
            methods=self.methods,
            virtuals=self.virtuals,
            vtables=self.vtables,
        )

    def extend_types(self, name: str, ty: Type):
        return Context(
            env=self.env,
            interfaces=self.interfaces,
            interface_impls=self.interface_impls,
            types=self.types.extend(name, ty),
            methods=self.methods,
            virtuals=self.virtuals,
            vtables={ty: {}} | self.vtables,
        )

    def set_type(self, name: str, ty: Type):
        self.types.set(name, ty)

    def extend_interfaces(self, name: str, ty: t.InterfaceType):
        return Context(
            env=self.env,
            interfaces=self.interfaces.extend(name, ty),
            interface_impls=self.interface_impls,
            types=self.types,
            methods=self.methods,
            virtuals=self.virtuals,
            vtables=self.vtables,
        )

    def set_interface(self, name: str, ty: t.InterfaceType):
        self.interfaces.set(name, ty)

    def find_method(self, ty: Type, name: str) -> Optional[tuple[Type, int]]:
        match ty:
            case t.InterfaceType(_, methods):
                signature = methods[name]
                return signature, self.virtuals[(ty, name)]
            case _:
                try:
                    return self.methods[(ty, name)]
                except KeyError:
                    return None

    def implements(self, ty: Type, intf: t.InterfaceType) -> bool:
        return (ty, intf) in self.interface_impls

    def declare_impl(self, ty: Type, intf: t.InterfaceType):
        return Context(
            env=self.env,
            interfaces=self.interfaces,
            interface_impls=self.interface_impls | {(ty, intf)},
            types=self.types,
            methods=self.methods,
            virtuals=self.virtuals,
            vtables=self.vtables,
        )

    def declare_virtual(self, ifty: t.InterfaceType, method: str):
        table = 0  # we only use a single vtable for now...
        index = len(self.virtuals)
        return Context(
            env=self.env,
            interfaces=self.interfaces,
            interface_impls=self.interface_impls,
            types=self.types,
            methods=self.methods,
            virtuals=self.virtuals | {(ifty, method): (table, index)},
            vtables=self.vtables,
        )

    def add_vtable(self, ty: Type, interface: t.InterfaceType):
        vtables = self.vtables.copy()
        table = vtables.setdefault(ty, {})

        for m in interface.methods.keys():
            _, actual_method_idx = self.methods[(ty, m)]
            tbl, index = self.virtuals[(interface, m)]
            table[(tbl, index)] = actual_method_idx

        return Context(
            env=self.env,
            interfaces=self.interfaces,
            interface_impls=self.interface_impls,
            types=self.types,
            methods=self.methods,
            virtuals=self.virtuals,
            vtables=vtables,
        )


def check_program(pgm: ast.Program) -> ast.Program:
    match pgm:
        case ast.Program(exp, records, impls, interfaces):
            ctx = Context()

            for intf in interfaces:
                ifty = t.InterfaceType(intf.name, None)
                ctx = ctx.extend_interfaces(intf.name, ifty)

                for method in intf.methods.keys():
                    ctx = ctx.declare_virtual(ifty, method)

            for record in records:
                ctx = ctx.extend_types(record.name, t.NamedType(record.name, None))

            for intf in interfaces:
                methods = {}
                intf_ctx = ctx.extend_types("Self", t.NamedType("Self", None))
                for mtn, mts in intf.methods.items():
                    methods[mtn] = eval_type(mts, intf_ctx)
                ctx.interfaces.lookup(intf.name).set_methods(methods)

            for record in records:
                ctx.types.lookup(record.name).set_type(
                    eval_type(ast.RecordType(record.fields), ctx)
                )

            impls_out = []
            for impl in impls:
                impl_on = ctx.types.lookup(impl.type_name)

                if impl.interface is not None:
                    interface_obj = ctx.interfaces.lookup(impl.interface)
                    ctx = ctx.declare_impl(impl_on, interface_obj)

                impl_ctx = ctx.extend_types("Self", impl_on)
                for method_name, func in impl.methods.items():
                    signature = eval_type(func.type, impl_ctx)
                    index = len(impl_ctx.methods)
                    impl_ctx.methods[(impl_on, method_name)] = signature, index

                if impl.interface is not None:
                    interface_obj = ctx.interfaces.lookup(impl.interface)
                    ctx = ctx.add_vtable(impl_on, interface_obj)

                methods_out = {}
                for method_name, func in impl.methods.items():
                    signature, _ = ctx.methods[(impl_on, method_name)]

                    body = check_expr(func.expr, signature, ctx)
                    methods_out[method_name] = body

                impls_out.append(ast.ImplBlock(impl.interface, impl_on, methods_out))

            prog, _ = infer_expr(exp, ctx)
            return ast.Program(prog, records, impls_out, interfaces)


def check_expr(exp: ast.Expression, typ: Type, ctx: Context) -> ast.Expression:
    match typ, exp:
        case ast.Type(), _:
            raise TypeError("Unevaluated type passed to checker")

        case _, ast.Literal(val):
            mapping = {type(None): t.NullType, bool: t.BoolType, int: t.IntType}
            if mapping[type(val)] != type(typ):
                raise TypeError(exp, typ)
            return exp

        case _, ast.Identifier(name):
            actual_t = ctx.env.lookup(name)
            if not ctx.check(actual_t, typ):
                raise TypeError(f"Expected a {typ} but {name} is a {actual_t}")
            return exp

        case t.IntType(), ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), ctx)
            rhs = check_expr(rhs, t.IntType(), ctx)
            return ast.BinOp(lhs, rhs, op)

        case t.ListType(_), ast.EmptyList():
            return exp

        case t.ListType(item_t), ast.BinOp(lhs, rhs, "::" as op):
            lhs = check_expr(lhs, item_t, ctx)
            rhs = check_expr(rhs, typ, ctx)
            return ast.BinOp(lhs, rhs, op)

        case t.FuncType(_, _), ast.Function(arms):
            return ast.Function(
                [
                    ast.MatchArm(arm.pats, check_matcharm(arm.pats, arm.body, typ, ctx))
                    for arm in arms
                ]
            )

        case _, ast.Application(fun, arg):
            fun, fun_t = infer_expr(fun, ctx)
            match fun_t:
                case t.FuncType(arg_t, ret_t):
                    pass
                case _:
                    raise TypeError(f"Cannot call {fun_t}")
            if ret_t != typ:
                raise TypeError(f"Function returns {ret_t} but expected {typ}")
            return ast.Application(fun, check_expr(arg, arg_t, ctx))

        case _, ast.Conditional(c, a, b):
            c_out = check_expr(c, t.BoolType(), ctx)
            a_out = check_expr(a, typ, ctx)
            b_out = check_expr(b, typ, ctx)
            return ast.Conditional(c_out, a_out, b_out)

        # initializing a named record type with a record expression
        case t.NamedType(_, t.RecordType(t_fields)), ast.RecordExpr(v_fields):
            extra_fields = v_fields.keys() - t_fields.keys()
            missing_fields = t_fields.keys() - v_fields.keys()
            if extra_fields or missing_fields:
                raise TypeError(
                    f"extra fields: {extra_fields}, missing fields: {missing_fields}"
                )

            slots = [check_expr(v_fields[f], t_fields[f], ctx) for f in t_fields]

            return ast.TupleExpr(slots, vtable=ctx.vtables[typ])

        # initializing a named empty record
        case t.NamedType(_, t.RecordType(t_fields)), ast.EmptyList():
            if t_fields:
                raise TypeError(f"missing fields: {t_fields}")

            return ast.TupleExpr([], vtable=ctx.vtables[typ])

        case t.InterfaceType() as intf, exp:
            e_out, actual_t = infer_expr(exp, ctx)
            if not ctx.implements(actual_t, intf):
                raise TypeError(f"{actual_t} dos not implement {intf}")
            return e_out

        case _, _:
            e_out, actual_t = infer_expr(exp, ctx)
            if actual_t != typ:
                raise TypeError(actual_t, typ)
            return e_out


class InferenceError(Exception):
    pass


def infer_expr(exp: ast.Expression, ctx: Context) -> (ast.Expression, Type):
    match exp:
        case ast.Literal(val):
            mapping = {type(None): t.NullType, bool: t.BoolType, int: t.IntType}
            return exp, mapping[type(val)]()

        case ast.Identifier(name):
            return exp, ctx.env.lookup(name)

        case ast.EmptyList():
            raise InferenceError("can't infer empty list type")

        case ast.TypeAnnotation(tx, expr):
            ty = eval_type(tx, ctx)
            expr = check_expr(expr, ty, ctx)
            return expr, ty

        case ast.BinOp(lhs, rhs, "+" | "-" | "*" | "/" as op):
            lhs = check_expr(lhs, t.IntType(), ctx)
            rhs = check_expr(rhs, t.IntType(), ctx)
            return ast.BinOp(lhs, rhs, op), t.IntType

        case ast.BinOp(lhs, rhs, "::" as op):
            lhs, item_t = infer_expr(lhs, ctx)
            rhs = check_expr(rhs, t.ListType(item_t), ctx)
            return ast.BinOp(lhs, rhs, op), t.IntType

        case ast.Function(patterns):
            raise InferenceError(
                f"can't infer function signature for {exp}. Please provide a type hint."
            )

        case ast.Application(fun, arg):
            fun, fun_t = infer_expr(fun, ctx)
            match fun_t:
                case t.FuncType(arg_t, ret_t):
                    pass
                case _:
                    raise TypeError(f"Cannot call {fun_t}")

            return ast.Application(fun, check_expr(arg, arg_t, ctx)), ret_t

        case ast.Let(var, val, bdy, None):
            # without type declaration, the let variable can't be used in the val expression
            val, val_t = infer_expr(val, ctx)
            body, out_t = infer_expr(bdy, ctx.extend_env(var, val_t))
            return ast.Let(var, val, body, val_t), out_t

        case ast.Let(var, val, bdy, var_t):
            var_t = eval_type(var_t, ctx)
            let_ctx = ctx.extend_env(var, var_t)
            val = check_expr(val, var_t, let_ctx)
            body, out_t = infer_expr(bdy, let_ctx)
            return ast.Let(var, val, body, var_t), out_t

        case ast.BlockExpression(fst, snd):
            fst = check_stmt(fst, ctx)
            snd, out_t = infer_expr(snd, ctx)
            return ast.BlockExpression(fst, snd), out_t

        case ast.Conditional(_, a, b):
            try:
                _, out_t = infer_expr(a, ctx)
            except InferenceError:
                _, out_t = infer_expr(b, ctx)
            return check_expr(exp, out_t, ctx), out_t

        case ast.NewRef(val):
            v_out, v_t = infer_expr(val, ctx)
            return ast.NewRef(v_out), t.BoxType(v_t)

        case ast.DeRef(box):
            b_out, box_t = infer_expr(box, ctx)
            if not isinstance(box_t, t.BoxType):
                raise TypeError(f"Cannot deref a {box_t}")
            return ast.DeRef(b_out), box_t

        case ast.RecordExpr(fields):
            field_values = []
            field_types = {}
            # sort anonymous record fields, to achieve structural equivalence
            field_names = sorted(fields.keys())
            for name in field_names:
                v_out, val_t = infer_expr(fields[name], ctx)
                field_values.append(v_out)
                field_types[name] = val_t
            return ast.TupleExpr(field_values, vtable=None), t.RecordType(field_types)

        case ast.GetAttribute(obj, fld):
            obj, obj_t = infer_expr(obj, ctx)

            match ctx.find_method(obj_t, fld):
                case (signature, (table, index)):
                    result = ast.GetVirtual(obj, table, index)
                    return result, signature.ret
                case (signature, index):
                    result = ast.Application(ast.GetMethod(index), obj)
                    return result, signature.ret

            rec_rt = resolve_type(obj_t)
            if not isinstance(rec_rt, t.RecordType):
                raise TypeError(f"Expected record type, got {obj_t}")
            if fld not in rec_rt.fields:
                raise TypeError(f"Record has no attribute {fld}")
            idx = list(rec_rt.fields.keys()).index(fld)
            return ast.GetSlot(obj, idx), rec_rt.fields[fld]

        case _:
            raise NotImplementedError(exp)


def check_stmt(stmt: ast.Statement, env: TEnv) -> ast.Statement:
    match stmt:
        case ast.NopStatement():
            return stmt

        case ast.ExprStmt(expr):
            expr, _ = infer_expr(expr, env)
            return ast.ExprStmt(expr)

        case ast.IfStatement(c, a, b):
            c_out = check_expr(c, t.BoolType(), env)
            a_out = check_stmt(a, env)
            b_out = check_stmt(b, env)
            return ast.IfStatement(c_out, a_out, b_out)

        case ast.Assignment(lhs, rhs):
            try:
                l_out, l_t = infer_expr(lhs, env)
                if not isinstance(l_t, t.BoxType):
                    raise TypeError(f"Cannot assign to a {l_t}")
                r_out = check_expr(rhs, l_t.item_t, env)
                return ast.Assignment(l_out, r_out)
            except InferenceError:
                pass
            r_out, r_t = infer_expr(rhs, env)
            l_out = check_expr(lhs, t.BoxType(r_t), env)
            return ast.Assignment(l_out, r_out)

        case _:
            raise NotImplementedError(stmt)


def check_matcharm(
    patterns: list[ast.Pattern], body: ast.Expression, typ: Type, ctx: Context
) -> ast.Expression:
    match typ, patterns:
        case t.FuncType(arg_t, res_t), [p0, *p_rest]:
            bindings = check_pattern(p0, arg_t, ctx)
            for name, ty in bindings.items():
                ctx = ctx.extend_env(name, ty)
            return check_matcharm(p_rest, body, res_t, ctx)
        case _, []:
            return check_expr(body, typ, ctx)
        case other:
            raise NotImplementedError(other)


def check_pattern(pat: ast.Pattern, typ: Type, env: TEnv) -> dict[str, Type]:
    match typ, pat:
        case _, ast.BindingPattern(name):
            return {name: typ}
        case _, ast.LiteralPattern(val):
            _ = check_expr(ast.Literal(val), typ, env)
            return {}
        case t.ListType(item_t), ast.ListConsPattern(car, cdr):
            return check_pattern(car, item_t, env) | check_pattern(cdr, typ, env)
        case t.ListType(_), ast.EmptyListPattern():
            return {}
        case other:
            raise NotImplementedError(other)


def eval_type(tx: ast.Type, ctx: Context) -> Type:
    match tx:
        case ast.TypeRef(name):
            try:
                return ctx.types.lookup(name)
            except LookupError:
                return ctx.interfaces.lookup(name)
        case ast.NullType:
            return t.NullType()
        case ast.BoolType:
            return t.BoolType()
        case ast.IntType:
            return t.IntType()
        case ast.ListType(item_t):
            return t.ListType(eval_type(item_t, ctx))
        case ast.RecordType(fields):
            return t.RecordType({n: eval_type(ty, ctx) for n, ty in fields.items()})
        case ast.FuncType(arg_t, ret_t):
            return t.FuncType(eval_type(arg_t, ctx), eval_type(ret_t, ctx))
        case _:
            raise NotImplementedError(tx)


def resolve_type(ty: t.Type) -> t.Type:
    match ty:
        case t.NamedType(_, tt):
            return tt
        case _:
            return ty
