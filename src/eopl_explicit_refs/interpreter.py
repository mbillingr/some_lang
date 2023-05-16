from __future__ import annotations
import abc
import dataclasses
from typing import Any, Callable, Self, Iterable

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


UNDEFINED = object()


@dataclasses.dataclass
class Context:
    env: Env = EmptyEnv()
    method_names: dict[str, int] = dataclasses.field(default_factory=dict)
    vtables: dict[str, int] = dataclasses.field(default_factory=dict)

    def extend_env(self, *vars: str) -> Self:
        return Context(env=self.env.extend(*vars), method_names=self.method_names, vtables=self.vtables)

    def find_method(self, name: str) -> int:
        return self.method_names[name]

    def register_method(self, name: str) -> int:
        index = len(self.method_names)
        self.method_names[name] = index
        return index


def analyze_program(pgm: ast.ExecutableProgram) -> Any:
    ctx = Context()

    ctx.vtables = {k: i for i, k in enumerate(pgm.vtables.keys())}
    vtables = list(pgm.vtables.values())

    methods = analyze_static_functions(pgm.functions, ctx)

    exp = analyze_expr(pgm.exp, ctx, tail=False)

    def program(store):
        store.clear()
        store.set_vtables(vtables)
        methods(store)
        return exp(store)

    return program


@dataclasses.dataclass
class Module:
    methods: list[Closure] = dataclasses.field(default_factory=list)

    def __call__(self, store):
        for m in self.methods:
            store.add_method(m)


def analyze_module(mod: ast.CheckedModule, ctx: Context) -> tuple[Module, Context]:
    match mod:
        case ast.Module():
            raise TypeError("Cannot run unchecked module")
        case ast.CheckedModule(_, types, impls):
            methods = []
            for impl in impls:
                for method_name, method in impl.methods.items():
                    arms = analyze_matcharms(method.patterns, ctx)
                    # the order these are appended in must match the order
                    # of method indices generated in the type checker
                    idx = ctx.register_method(method_name)
                    ctx.method_names[method_name] = idx
                    methods.append(Closure(ctx, arms))

            mod_out = Module(methods)

            return mod_out, ctx


def analyze_static_functions(funcs: Iterable[tuple[ast.Symbol, ast.Function]], ctx: Context) -> Callable:
    for name, _ in funcs:
        ctx.register_method(name)

    bodies = [analyze_matcharms(f.patterns, ctx) for _, f in funcs]

    def initialization(store):
        for body in bodies:
            store.add_method(Procedure(body))

    return initialization


def analyze_expr(exp: ast.Expression, ctx: Context, tail) -> Callable:
    match exp:
        case ast.Literal(val):
            return lambda _: val
        case ast.Identifier(name):
            idx = ctx.env.lookup(name)
            return lambda store: store.get(idx)
        case ast.EmptyList():
            return lambda _: empty_list()
        case ast.BinOp(lhs, rhs, op):
            lhs_ = analyze_expr(lhs, ctx, tail=False)
            rhs_ = analyze_expr(rhs, ctx, tail=False)
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
            val_ = analyze_expr(val, ctx, tail=False)
            return lambda store: store.newref(val_(store))

        case ast.Assignment(lhs, rhs):
            lhs_ = analyze_expr(lhs, ctx, tail=False)
            rhs_ = analyze_expr(rhs, ctx, tail=False)

            def the_assignment(store):
                store.setref(lhs_(store), rhs_(store))

            return the_assignment

        case ast.DeRef(ref):
            ref_ = analyze_expr(ref, ctx, tail=False)
            return lambda store: store.deref(ref_(store))

        case ast.BlockExpression(stmt, expr):
            stmt_ = analyze_expr(stmt, ctx, tail=False)
            expr_ = analyze_expr(expr, ctx, tail=tail)

            def sequence(store):
                stmt_(store)
                return expr_(store)

            return sequence

        case ast.Conditional(a, b, c):
            cond_ = analyze_expr(a, ctx, tail=False)
            then_ = analyze_expr(b, ctx, tail=tail)
            else_ = analyze_expr(c, ctx, tail=tail)
            return lambda store: then_(store) if cond_(store) else else_(store)

        case ast.Let(var, val, bdy):
            let_env = ctx.extend_env(var)
            val_ = analyze_expr(val, let_env, tail=False)
            bdy_ = analyze_expr(bdy, let_env, tail=tail)

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

        case ast.GetMethod(name):
            idx = ctx.find_method(name)
            return lambda store: store.get_method(idx)

        case ast.GetVirtual(obj, table, vidx):
            obj_ = analyze_expr(obj, ctx, tail=False)

            def the_getter(store):
                evaluated_object = obj_(store)
                method_idx = evaluated_object.vtable_lookup(table, vidx)
                return store.get_method(method_idx).apply([evaluated_object], store)

            return the_getter

        case ast.Application():
            return analyze_application(exp, ctx=ctx, tail=tail)

        case ast.RecordExpr(fields):
            fields_ = {n: analyze_expr(v, ctx, tail=False) for n, v in fields.items()}
            return lambda store: {n: v(store) for n, v in fields_.items()}

        case ast.GetAttribute(obj, fld):
            obj_ = analyze_expr(obj, ctx, tail=False)
            return lambda store: obj_(store)[fld]

        case ast.TupleExpr(slots):
            slots_ = [analyze_expr(v, ctx, tail=False) for v in slots]
            return lambda store: TupleObj(v(store) for v in slots_)

        case ast.GetSlot(obj, idx):
            obj_ = analyze_expr(obj, ctx, tail=False)
            return lambda store: obj_(store)[idx]

        case ast.WithInterfaces(obj, typename):
            obj_ = analyze_expr(obj, ctx, tail=False)
            vtable_idx = ctx.vtables[typename]
            return lambda store: obj_(store).with_vtable(store.get_vtable(vtable_idx))

        case _:
            raise NotImplementedError(exp)


def analyze_matcharms(arms: list[ast.MatchArm], ctx: Context) -> list[tuple[Matcher, Callable]]:
    match_bodies = []
    for arm in arms:
        matcher = analyze_patterns(arm.pats)
        bdy_env = ctx.extend_env(*matcher.bindings())
        bdy_ = analyze_expr(arm.body, bdy_env, tail=True)
        match_bodies.append((matcher, bdy_))
    return match_bodies


def analyze_application(fun, *args_, ctx, tail):
    match fun:
        case ast.Application(f, a):
            a_ = analyze_expr(a, ctx, tail=False)
            return analyze_application(f, a_, *args_, ctx=ctx, tail=tail)
        case _:
            fun_ = analyze_expr(fun, ctx, tail=False)

            if tail:

                def tail_call(store):
                    raise TailCall(fun_(store), [a(store) for a in args_])

                return tail_call
            else:
                return lambda store: fun_(store).apply([a(store) for a in args_], store)


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


class Procedure:
    def __init__(self, match_bodies):
        self.match_bodies = match_bodies

    def prepare_env(self, store):
        pass

    def apply(self, args, store):
        preserved_stack = store.env
        try:
            self.prepare_env(store)
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
                    self.prepare_env(store)
                    match_bodies = tc.func.match_bodies
                    args = tc.args
        finally:
            store.env = preserved_stack


class Closure(Procedure):
    def __init__(self, store, match_bodies):
        super().__init__(match_bodies)
        self.saved_env = store.env

    def prepare_env(self, store):
        store.env = self.saved_env


class Partial:
    """A partially applied function"""

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def apply(self, args, store):
        return self.func.apply(self.args + args, store)


@dataclasses.dataclass
class TailCall(Exception):
    func: Procedure
    args: list[Any]


def empty_list():
    return ()


def list_cons(car, cdr):
    return car, cdr


class TupleObj(tuple):
    def with_vtable(self, vtables):
        self._vtables = vtables
        return self

    def vtable_lookup(self, table: int, method: int):
        return self._vtables[table][method]
