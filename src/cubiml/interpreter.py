import collections
import dataclasses
from typing import Mapping, Any, Iterator

from cubiml import ast


class Interpreter:
    def __init__(self):
        self.env = {}

    def run_script(self, script: ast.Script):
        env = self.env
        for statement in script.statements:
            env = eval_toplevel(statement, env)

        self.env = {}
        for k, v in env.items():
            if k not in self.env:
                self.env[k] = v


def eval_toplevel(stmt: ast.ToplevelItem, env: Mapping[str, Any]) -> Any:
    match stmt:
        case ast.DefineLet(var, val):
            return extend_env(var, evaluate(val, env), env)
        case ast.DefineLetRec(bind):
            return make_letrec_env(bind, env)
        case ast.Expression():
            evaluate(stmt, env)
            return env
        case _:
            raise NotImplementedError(stmt)


def evaluate(expr: ast.Expression, env: Mapping[str, Any]) -> Any:
    while True:
        match expr:
            case ast.Literal(val):
                return val
            case ast.Reference(var):
                return env[var]
            case ast.Conditional(condition, consequence, alternative):
                if evaluate(condition, env):
                    expr = consequence
                else:
                    expr = alternative
            case ast.Record(fields):
                return {k: evaluate(v, env) for k, v in fields.items()}
            case ast.FieldAccess(field, exp):
                return evaluate(exp, env)[field]
            case ast.Case(tag, exp):
                return tag, evaluate(exp, env)
            case ast.Match(exp, arms):
                (tag, val) = evaluate(exp, env)
                for arm in arms:
                    if arm.tag == tag:
                        return evaluate(arm.bdy, extend_env(arm.var, val, env))
                raise RuntimeError("No arm matched")
            case ast.Function(var, body):
                return Function(env, var, body)
            case ast.Application(fun, arg):
                fval = evaluate(fun, env)
                aval = evaluate(arg, env)
                env = extend_env(fval.var, aval, fval.captured_env)
                expr = fval.body
            case ast.Let(var, val, body):
                env = extend_env(var, evaluate(val, env), env)
                expr = body
            case ast.LetRec(bind, body):
                env = make_letrec_env(bind, env)
                expr = body
            case _:
                raise NotImplementedError(expr)


def make_letrec_env(
    bind: list[ast.FuncDef], env: Mapping[str, Any]
) -> Mapping[str, Any]:
    recenv = {b.name: None for b in bind}
    env = MultiEnv(recenv, env)
    for b in bind:
        recenv[b.name] = Function(env, b.fun.var, b.fun.body)
    return env


@dataclasses.dataclass
class Function:
    captured_env: Mapping[str, Any]
    var: str
    body: ast.Expression


def extend_env(var: str, val: Any, env: Mapping[str, Any]) -> Mapping[str, Any]:
    return Env(var, val, env)


@dataclasses.dataclass
class Env(collections.abc.Mapping[str, Any]):
    var: str
    val: Any
    env: Mapping[str, Any]

    def __getitem__(self, item: str) -> Any:
        if self.var == item:
            return self.val
        return self.env[item]

    def __len__(self) -> int:
        return 1 + len(self.env)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield self.var
        yield from self.env


@dataclasses.dataclass
class MultiEnv(collections.abc.Mapping[str, Any]):
    env: Mapping[str, Any]
    parent: Mapping[str, Any]

    def __getitem__(self, item: str) -> Any:
        try:
            return self.env[item]
        except KeyError:
            pass
        return self.parent[item]

    def __len__(self) -> int:
        return len(self.env) + len(self.parent)

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from self.env
        yield from self.parent
