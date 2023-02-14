import collections
import dataclasses
from typing import Mapping, Any, Iterator

from cubiml import abstract_syntax as ast


NEVER = None  # representation of the unusable type


class Interpreter:
    def __init__(self):
        self.env = {}

    def run_script(self, script: ast.Script):
        result = None
        env = self.env
        for statement in script.statements:
            result, env = eval_toplevel(statement, env)

        self.env = {}
        for k, v in env.items():
            if k not in self.env:
                self.env[k] = v

        return result


def eval_toplevel(
    stmt: ast.ToplevelItem, env: Mapping[str, Any]
) -> (Any, Mapping[str, Any]):
    match stmt:
        case ast.DefineLet(var, val):
            return None, extend_env(var, evaluate(val, env), env)
        case ast.DefineLetRec(bind):
            return None, make_letrec_env(bind, env)
        case ast.Expression() as exp:
            val = evaluate(exp, env)
            return val, env
        case _:
            raise NotImplementedError(stmt)


def evaluate(expr: ast.Expression, env: Mapping[str, Any]) -> Any:
    while True:
        match expr:
            case ast.Literal(val):
                return val
            case ast.Reference(var):
                return env[var]
            case ast.BinOp(lhs, rhs, _, op):
                a = evaluate(lhs, env)
                b = evaluate(rhs, env)
                match op:
                    case "+":
                        return a + b
                    case "-":
                        return a - b
                    case "*":
                        return a * b
                    case "/":
                        return a // b
                    case "<":
                        return a < b
                    case "<=":
                        return a <= b
                    case ">":
                        return a > b
                    case ">=":
                        return a >= b
                    case "==":
                        return a == b
                    case "!=":
                        return a != b
                    case _:
                        raise NotImplementedError(op)
            case ast.Conditional(condition, consequence, alternative):
                if evaluate(condition, env):
                    expr = consequence
                else:
                    expr = alternative
            case ast.Record(fields):
                return {k: evaluate(v, env) for k, v in fields}
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
            case ast.Procedure(var, body):
                return Procedure(env, var, body)
            case ast.Application(fun, arg):
                fval = evaluate(fun, env)
                aval = evaluate(arg, env)
                env = extend_env(fval.var, aval, fval.captured_env)
                match fval:
                    case Procedure(_, _, body):
                        for stmt in body[:-1]:
                            _ = evaluate(stmt, env)
                        expr = body[-1]
                    case Function(_, _, body):
                        expr = body
                    case _:
                        raise RuntimeError("Invalid callable")
            case ast.Let(var, val, body):
                env = extend_env(var, evaluate(val, env), env)
                expr = body
            case ast.LetRec(bind, body):
                env = make_letrec_env(bind, env)
                expr = body
            case ast.NewRef(init):
                return Cell(evaluate(init, env))
            case ast.RefGet(ref):
                return evaluate(ref, env).val
            case ast.RefSet(ref, val):
                r = evaluate(ref, env)
                r.val = evaluate(val, env)
                return NEVER
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
class Cell:
    val: Any


@dataclasses.dataclass
class Function:
    captured_env: Mapping[str, Any]
    var: str
    body: ast.Expression


@dataclasses.dataclass
class Procedure:
    captured_env: Mapping[str, Any]
    var: str
    body: list[ast.Expression]


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
