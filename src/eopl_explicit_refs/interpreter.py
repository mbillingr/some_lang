import abc
import dataclasses
from typing import Any, Callable

from eopl_explicit_refs.environment import EmptyEnv, Env
from eopl_explicit_refs import abstract_syntax as ast


UNDEFINED = object()


def init_env() -> Env:
    return EmptyEnv()


def analyze_program(pgm: ast.Program) -> Any:
    match pgm:
        case ast.Program(exp):
            prog = analyze_expr(exp, init_env(), tail=False)

            def program(store):
                store.clear()
                return prog(store)

            return program


def analyze_stmt(stmt: ast.Statement, env: Env) -> Callable:
    match stmt:
        case ast.NopStatement():
            return lambda _: None

        case ast.ExprStmt(expr):
            # Let's assume expressions have no side effects, so we don't execute them,
            # but we still check if they are valid
            _ = analyze_expr(expr, env, tail=False)
            return lambda _: None

        case ast.Assignment(lhs, rhs):
            lhs_ = analyze_expr(lhs, env, tail=False)
            rhs_ = analyze_expr(rhs, env, tail=False)
            return lambda store: store.setref(lhs_(store), rhs_(store))

        case ast.IfStatement(cond, cons, alt):
            cond_ = analyze_expr(cond, env, tail=False)
            cons_ = analyze_stmt(cons, env)
            alt_ = analyze_stmt(alt, env)
            return lambda store: cons_(store) if cond_(store) else alt_(store)

        case _:
            raise NotImplementedError(stmt)


def analyze_expr(exp: ast.Expression, env: Env, tail) -> Callable:
    match exp:
        case ast.Literal(val):
            return lambda _: val
        case ast.Identifier(name):
            idx = env.lookup(name)
            return lambda store: store.get(idx)
        case ast.EmptyList():
            return lambda _: empty_list()
        case ast.BinOp(lhs, rhs, op):
            lhs_ = analyze_expr(lhs, env, tail=False)
            rhs_ = analyze_expr(rhs, env, tail=False)
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
            val_ = analyze_expr(val, env, tail=False)
            return lambda store: store.newref(val_(store))
        case ast.DeRef(ref):
            ref_ = analyze_expr(ref, env, tail=False)
            return lambda store: store.deref(ref_(store))
        case ast.BlockExpression(stmt, expr):
            stmt_ = analyze_stmt(stmt, env)
            expr_ = analyze_expr(expr, env, tail=tail)

            def sequence(store):
                stmt_(store)
                return expr_(store)

            return sequence

        case ast.Conditional(a, b, c):
            cond_ = analyze_expr(a, env, tail=False)
            then_ = analyze_expr(b, env, tail=tail)
            else_ = analyze_expr(c, env, tail=tail)
            return lambda store: then_(store) if cond_(store) else else_(store)
        case ast.Let(var, val, bdy):
            let_env = env.extend(var)
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
            match_bodies = []
            for arm in arms:
                matcher = analyze_patterns(arm.pats)
                bdy_env = env.extend(*matcher.bindings())
                bdy_ = analyze_expr(arm.body, bdy_env, tail=True)
                match_bodies.append((matcher, bdy_))

            def the_function(store):
                return Closure(store, match_bodies)

            return the_function

        case ast.Application():
            return analyze_application(exp, env=env, tail=tail)

        case ast.RecordExpr(fields):
            fields_ = {n: analyze_expr(v, env, tail=False) for n, v in fields.items()}
            return lambda store: {n: v(store) for n, v in fields_.items()}

        case ast.GetField(obj, fld):
            obj_ = analyze_expr(obj, env, tail=False)
            return lambda store: obj_(store)[fld]

        case ast.TupleExpr(slots):
            slots_ = [analyze_expr(v, env, tail=False) for v in slots]
            return lambda store: tuple(v(store) for v in slots_)

        case ast.GetSlot(obj, idx):
            obj_ = analyze_expr(obj, env, tail=False)
            return lambda store: obj_(store)[idx]

        case _:
            raise NotImplementedError(exp)


def analyze_application(fun, *args_, env, tail):
    match fun:
        case ast.Application(f, a):
            a_ = analyze_expr(a, env, tail=False)
            return analyze_application(f, a_, *args_, env=env, tail=tail)
        case _:
            fun_ = analyze_expr(fun, env, tail=False)

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
