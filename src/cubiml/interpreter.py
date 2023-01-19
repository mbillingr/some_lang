import collections
import dataclasses
from typing import Mapping, Any, Iterator

from cubiml import ast


def run_script(script: ast.Script, env: Mapping[str, Any]) -> Any:
    result = None
    for stmt in script.statements:
        match stmt:
            case ast.DefineLet(var, val):
                env = extend_env(var, evaluate(val, env), env)
            case ast.DefineLetRec(bind):
                env = make_letrec_env(bind, env)
            case ast.Expression():
                result = evaluate(stmt, env)
            case _:
                raise NotImplementedError(stmt)
    return result


def evaluate(expr: ast.Expression, env: Mapping[str, Any]) -> Any:
    while True:
        match expr:
            case ast.Boolean(val):
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
                    if arm.variant == tag:
                        return evaluate(arm.body, extend_env(arm.binding, val, env))
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
    recenv = {b.var: None for b in bind}
    env = MultiEnv(recenv, env)
    for b in bind:
        recenv[b.var] = Function(env, b.fun.var, b.fun.body)
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
        yield self.var, self.val
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
