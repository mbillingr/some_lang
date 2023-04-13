from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, Optional, Literal

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


UNDEFINED = object()


class Method:
    def n_args(self) -> int:
        return 0

    def apply(self, args, store):
        return UNDEFINED


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
        self.methods = methods or {}

    def get_method(self, name: str) -> Method:
        try:
            return self.methods[name]
        except KeyError:
            return self.super.get_method(name)


OBJECT = Class(super_class=None, methods={"init": Method()})


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
    return Class()


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
            match_bodies = []
            for arm in arms:
                matcher = analyze_patterns(arm.pats)
                bdy_ctx = ctx.lexical_extend(*matcher.bindings())
                bdy_ = analyze_expr(arm.body, bdy_ctx.in_tail(True))
                match_bodies.append((matcher, bdy_))

            def the_function(store):
                return Closure(store, match_bodies)

            return the_function

        case ast.Application():
            return analyze_application(exp, ctx=ctx)

        case ast.NewObj(cls):
            return analyze_newobj(ctx, cls)

        case _:
            raise NotImplementedError(exp)


def analyze_application(fun, *args_, ctx: StaticContext):
    match fun:
        case ast.Application(f, a):
            a_ = analyze_expr(a, ctx.in_tail(False))
            return analyze_application(f, a_, *args_, ctx=ctx)
        case ast.NewObj(cls):
            return analyze_newobj(ctx, cls, args_)
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
        init.apply([a(store) for a in args_], store)
        return []

    return newobj


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

    def apply(self, args, store):
        preserved_stack = store.env
        try:
            store.env = self.saved_env
            match_bodies = self.match_bodies
            while True:
                try:
                    for matcher, body in match_bodies:
                        if len(args) < matcher.n_args():
                            return Partial(self, args)
                        try:
                            bindings = matcher.match(args)
                        except MatcherError:
                            continue
                        store.push(*bindings)
                        res = body(store)
                        args = args[matcher.n_args() :]
                        if not args:
                            return res
                        else:
                            return res.apply(args, store)
                    raise MatcherError("no pattern matched")
                except TailCall as tc:
                    store.env = tc.func.saved_env
                    match_bodies = tc.func.match_bodies
                    args = tc.args
        finally:
            store.env = preserved_stack


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
