from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Optional, Literal, Iterator

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


class _Undefined:
    def __repr__(self):
        return "<undefined>"


UNDEFINED = _Undefined()


class Method:
    def n_args(self) -> int:
        raise NotImplementedError()

    def apply(self, args, store):
        raise NotImplementedError()


class Class:
    def __init__(
        self,
        super_class: Optional[Class] | Literal["object"] = "object",
        methods: Optional[dict[str, Method]] = None,
    ):
        if super_class == "object":
            self.super = OBJECT
        else:
            self.super = super_class

        self.method_names = {}
        self.methods = {}
        if methods:
            i = 0
            if self.super:
                for i, name in enumerate(self.super.iter_methods()):
                    if name in methods:
                        self.methods[i] = methods[name]
                        del methods[name]
                else:
                    i += 1
            for name, m in methods.items():
                self.method_names[name] = i
                self.methods[i] = m
                i = i + 1

    def get_method(self, name: str) -> Method:
        idx = self.get_method_idx(name)
        return self.get_method_by_idx(idx)

    def get_method_idx(self, name: str) -> int:
        try:
            return self.method_names[name]
        except KeyError:
            return self.super.get_method_idx(name)

    def get_method_by_idx(self, idx: int) -> Method:
        try:
            return self.methods[idx]
        except KeyError:
            return self.super.get_method_by_idx(idx)

    def iter_methods(self) -> Iterator[str]:
        if self.super:
            yield from self.super.iter_methods()
        yield from self.method_names.keys()

    def instantiate(self) -> Object:
        vtable = []
        for i, _ in enumerate(self.iter_methods()):
            vtable.append(self.get_method_by_idx(i))

        fields = []
        return Object(vtable, fields)


@dataclasses.dataclass
class Object:
    vtable: list
    fields: list


class ObjectInitMethod(Method):
    def n_args(self) -> int:
        return 0

    def apply(self, args, store):
        return UNDEFINED


OBJECT = Class(super_class=None, methods={"init": ObjectInitMethod()})


class StaticContext:
    def __init__(self, lexical_env=None, class_env=None, tail=False):
        self.lexical_env = lexical_env or EmptyEnv()
        self.class_env = class_env or {}
        self.tail = tail

    def in_tail(self, tail: bool):
        return StaticContext(self.lexical_env, self.class_env, tail)

    def lexical_extend(self, *vars: str) -> StaticContext:
        return StaticContext(self.lexical_env.extend(*vars), self.class_env, self.tail)

    def lexical_lookup(self, name: str) -> int:
        return self.lexical_env.lookup(name)

    def extend_classes(self, *cls_names: str) -> StaticContext:
        return StaticContext(
            self.lexical_env,
            self.class_env | {c: UNDEFINED for c in cls_names},
            self.tail,
        )

    def lookup_class(self, name: str) -> Class:
        return self.class_env[name]

    def set_class(self, name: str, cls: Class):
        self.class_env[name] = cls


def analyze_program(pgm: ast.Program) -> Any:
    match pgm:
        case ast.Program(classes, exp):
            ctx = StaticContext().extend_classes(*(c.name for c in classes))

            for cls in classes:
                cls_ = analyze_class_decl(cls, ctx)
                ctx.set_class(cls.name, cls_)

            prog = analyze_expr(exp, ctx)

            def program(store):
                store.clear()
                return prog(store)

            return program


def analyze_class_decl(cls: ast.Class, ctx: StaticContext) -> Class:
    super_cls = cls.super and ctx.lookup_class(cls.super) or "object"
    methods = {m.name: analyze_method(m, ctx) for m in cls.methods}
    return Class(super_class=super_cls, methods=methods)


def analyze_method(method: ast.Method, ctx: StaticContext) -> Method:
    match_bodies = analyze_matcharms(method.func.patterns, ctx)
    return CustomMethod(match_bodies)


class CustomMethod(Method):
    def __init__(self, match_bodies: list):
        self.match_bodies = match_bodies

    def n_args(self) -> int:
        return self.match_bodies[0][0].n_args()

    def apply(self, args, store):
        preserved_env = store.env
        try:
            store.env = EmptyEnv()
            match_bodies = self.match_bodies
            assert len(args) == self.n_args()
            return apply_pattern(match_bodies, args, store)
        finally:
            store.env = preserved_env


def analyze_stmt(stmt: ast.Statement, ctx: StaticContext) -> Callable:
    match stmt:
        case ast.NopStatement():
            return lambda _: None

        case ast.ExprStmt(expr):
            # Let's assume expressions have no side effects, so we don't execute them,
            # but we still check if they are valid
            _ = analyze_expr(expr, ctx)
            return lambda _: None

        case ast.Assignment(lhs, rhs):
            lhs_ = analyze_expr(lhs, ctx)
            rhs_ = analyze_expr(rhs, ctx)
            return lambda store: store.setref(lhs_(store), rhs_(store))

        case ast.IfStatement(cond, cons, alt):
            cond_ = analyze_expr(cond, ctx)
            cons_ = analyze_stmt(cons, ctx)
            alt_ = analyze_stmt(alt, ctx)
            return lambda store: cons_(store) if cond_(store) else alt_(store)

        case _:
            raise NotImplementedError(stmt)


def analyze_expr(exp: ast.Expression, ctx: StaticContext) -> Callable:
    match exp:
        case ast.Literal(val):
            return lambda _: val
        case ast.Identifier(name):
            idx = ctx.lexical_lookup(name)
            return lambda store: store.get(idx)
        case ast.EmptyList():
            return lambda _: empty_list()
        case ast.BinOp(lhs, rhs, op):
            lhs_ = analyze_expr(lhs, ctx.in_tail(False))
            rhs_ = analyze_expr(rhs, ctx.in_tail(False))
            match op:
                case "+":
                    return lambda store: lhs_(store) + rhs_(store)
                case "-":
                    return lambda store: lhs_(store) - rhs_(store)
                case "*":
                    return lambda store: lhs_(store) * rhs_(store)
                case "/":
                    return lambda store: lhs_(store) / rhs_(store)
                case "::":
                    return lambda store: list_cons(lhs_(store), rhs_(store))
                case _:
                    raise NotImplementedError(op)
        case ast.NewRef(val):
            val_ = analyze_expr(val, ctx.in_tail(False))
            return lambda store: store.newref(val_(store))
        case ast.DeRef(ref):
            ref_ = analyze_expr(ref, ctx.in_tail(False))
            return lambda store: store.deref(ref_(store))
        case ast.BlockExpression(stmt, expr):
            stmt_ = analyze_stmt(stmt, ctx.in_tail(False))
            expr_ = analyze_expr(expr, ctx)

            def sequence(store):
                stmt_(store)
                return expr_(store)

            return sequence

        case ast.Conditional(a, b, c):
            cond_ = analyze_expr(a, ctx.in_tail(False))
            then_ = analyze_expr(b, ctx)
            else_ = analyze_expr(c, ctx)
            return lambda store: then_(store) if cond_(store) else else_(store)

        case ast.Let(var, val, bdy):
            let_ctx = ctx.lexical_extend(var)
            val_ = analyze_expr(val, let_ctx.in_tail(False))
            bdy_ = analyze_expr(bdy, let_ctx)

            def let(store):
                store.push(UNDEFINED)
                store.set((0, 0), val_(store))
                result = bdy_(store)
                store.pop()
                return result

            return let

        case ast.Function(arms):
            match_bodies = analyze_matcharms(arms, ctx)

            def the_function(store):
                return Closure(store, match_bodies)

            return the_function

        case ast.Application():
            return analyze_application(exp, ctx=ctx)

        case ast.NewObj(cls):
            return analyze_newobj(ctx, cls)

        case ast.Message(obj, cls, method):
            return analyze_message(ctx, obj, cls, method)

        case _:
            raise NotImplementedError(exp)


def analyze_matcharms(arms, ctx) -> list:
    match_bodies = []
    for arm in arms:
        matcher = analyze_patterns(arm.pats)
        bdy_ctx = ctx.lexical_extend(*matcher.bindings())
        bdy_ = analyze_expr(arm.body, bdy_ctx.in_tail(True))
        match_bodies.append((matcher, bdy_))
    return match_bodies


def analyze_application(fun, *args_, ctx: StaticContext):
    match fun:
        case ast.Application(f, a):
            a_ = analyze_expr(a, ctx.in_tail(False))
            return analyze_application(f, a_, *args_, ctx=ctx)
        case ast.NewObj(cls):
            return analyze_newobj(ctx, cls, args_)
        case ast.Message(obj, cls, method):
            return analyze_message(ctx, obj, cls, method, args_)
        case _:
            fun_ = analyze_expr(fun, ctx.in_tail(False))

            if ctx.tail:

                def tail_call(store):
                    raise TailCall(fun_(store), [a(store) for a in args_])

                return tail_call
            else:
                return lambda store: fun_(store).apply([a(store) for a in args_], store)


def analyze_newobj(ctx: StaticContext, cls: str, args_=()):
    cls_obj = ctx.lookup_class(cls)
    assert cls_obj is not UNDEFINED

    init = cls_obj.get_method("init")
    assert len(args_) == init.n_args()

    def newobj(store):
        obj = cls_obj.instantiate()
        init.apply([a(store) for a in args_], store)
        return obj

    return newobj


def analyze_message(ctx: StaticContext, obj: ast.Expression, cls: str, method: str, args_=()):
    cls_obj = ctx.lookup_class(cls)
    assert cls_obj is not UNDEFINED

    method_idx = cls_obj.get_method_idx(method)
    method = cls_obj.get_method_by_idx(method_idx)
    assert len(args_) == method.n_args(), f"{len(args_)} == {method.n_args()}"

    obj_ = analyze_expr(obj, ctx.in_tail(False))

    def message(store):
        the_obj = obj_(store)
        method = the_obj.vtable[method_idx]
        return method.apply([a(store) for a in args_], store)

    return message


class Matcher(abc.ABC):
    @abc.abstractmethod
    def match(self, val) -> tuple[Any, ...]:
        pass

    @abc.abstractmethod
    def n_args(self) -> int:
        pass

    @abc.abstractmethod
    def bindings(self) -> list[str]:
        pass


def analyze_patterns(pats: list[ast.Pattern]) -> Matcher:
    match pats:
        case [ast.NullaryPattern()]:
            return NaryMatcher([])
        case _:
            return NaryMatcher(list(map(analyze_pattern, pats)))


def analyze_pattern(pat: ast.Pattern) -> Matcher:
    match pat:
        case ast.BindingPattern(name):
            return BindingMatcher(name)

        case ast.LiteralPattern(value):
            return LiteralMatcher(value)

        case ast.EmptyListPattern():
            return LiteralMatcher(empty_list())

        case ast.ListConsPattern(car, cdr):
            car_ = analyze_pattern(car)
            cdr_ = analyze_pattern(cdr)
            return ListConsMatcher(car_, cdr_)

        case _:
            raise NotImplementedError(pat)


@dataclasses.dataclass
class LiteralMatcher(Matcher):
    value: Any

    def match(self, val) -> tuple[Any, ...]:
        if self.value == val:
            return ()
        raise MatcherError(val, self.value)

    def n_args(self) -> int:
        return 1

    def bindings(self) -> list[str]:
        return []


@dataclasses.dataclass
class BindingMatcher(Matcher):
    name: str

    def match(self, val) -> tuple[Any, ...]:
        return (val,)

    def n_args(self) -> int:
        return 1

    def bindings(self) -> list[str]:
        return [self.name]


@dataclasses.dataclass
class NaryMatcher(Matcher):
    matchers: list[Matcher]

    def match(self, val) -> tuple[Any, ...]:
        return sum((m.match(v) for m, v in zip(self.matchers, val)), start=())

    def n_args(self) -> int:
        return len(self.matchers)

    def bindings(self) -> list[str]:
        return sum((m.bindings() for m in self.matchers), start=[])


@dataclasses.dataclass
class ListConsMatcher(Matcher):
    car: Matcher
    cdr: Matcher

    def match(self, val) -> tuple[Any, ...]:
        match val:
            case (a, d):
                return self.car.match(a) + self.cdr.match(d)
            case _:
                raise MatcherError(val, "non-empty list")

    def n_args(self) -> int:
        return 1

    def bindings(self) -> list[str]:
        return self.car.bindings() + self.cdr.bindings()


class MatcherError(Exception):
    pass


class Closure:
    def __init__(self, store, match_bodies):
        self.match_bodies = match_bodies
        self.saved_env = store.env

    def n_args(self):
        return self.match_bodies[0][0].n_args()

    def apply(self, args, store):
        preserved_stack = store.env
        try:
            store.env = self.saved_env
            match_bodies = self.match_bodies
            while True:
                try:
                    if len(args) < self.n_args():
                        return Partial(self, args)
                    res = apply_pattern(match_bodies, args, store)
                    if len(args) > self.n_args():
                        return res.apply(args[self.n_args() :], store)
                    else:
                        return res
                except TailCall as tc:
                    store.env = tc.func.saved_env
                    match_bodies = tc.func.match_bodies
                    args = tc.args
        finally:
            store.env = preserved_stack


def apply_pattern(match_bodies, args, store):
    for matcher, body in match_bodies:
        try:
            bindings = matcher.match(args)
        except MatcherError:
            continue
        store.push(*bindings)
        return body(store)
    raise MatcherError("no pattern matched")


class Partial:
    """A partially applied function"""

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def apply(self, args, store):
        return self.func.apply(self.args + args, store)


@dataclasses.dataclass
class TailCall(Exception):
    func: Closure
    args: list[Any]


def empty_list():
    return ()


def list_cons(car, cdr):
    return car, cdr
