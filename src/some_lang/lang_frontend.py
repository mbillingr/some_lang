from __future__ import annotations

import copy
from typing import Any, Optional

from some_lang import interpreter, parser, type_checker, ast, type_heads


class Context:
    def __init__(
        self,
        env: interpreter.Env = interpreter.EmptyEnv(),
        type_env: type_checker.Env = type_checker.EmptyEnv(),
        engine: Optional[type_checker.TypeCheckerCore] = None,
    ):
        self.engine = engine or type_checker.TypeCheckerCore()
        self.type_env = type_env
        self.env = env

    def init_default_env(self):
        self.define("not", lambda x: not x, "Bool -> Bool")
        self.define("0?", lambda x: x == 0, "Int -> Bool")
        self.define("inc", lambda x: x + 1, "Int -> Int")

    def define(self, name: str, val: interpreter.Value, ty: str | ast.TypeExpression):
        if not isinstance(ty, ast.TypeExpression):
            ty = parser.parse_type(ty)
        tv = type_checker.eval_vtype(ty, self.engine)
        self.type_env = self.type_env.extend(name, tv)
        self.env = self.env.extend(name, val)

    def eval(self, src: str | ast.Expression) -> Any:
        if not isinstance(src, ast.Expression):
            src = parser.parse_expr(src)

        type_checker.check_expr(src, self.type_env, self.engine)
        return interpreter.evaluate(src, self.env)

    def module(self, src: str | ast.Module) -> Context:
        if not isinstance(src, ast.Module):
            src = parser.parse_module(src)
        engine = copy.deepcopy(self.engine)
        type_env = type_checker.check_module(src, engine, self.type_env)
        env = interpreter.run_module(src, self.env)
        return Context(env, type_env, engine)

    def compile_expr(self, src: str | ast.Expression) -> Any:
        if not isinstance(src, ast.Expression):
            src = parser.parse_expr(src)

        expr_types = {}

        def annotate_expr(expr, v):
            expr_types[id(expr)] = v
            return v

        type_checker.check_expr(src, self.type_env, self.engine, callback=annotate_expr)
        print(self.engine)
        return code_gen(src, {}, self.engine, expr_types)


def code_gen(expr, env, engine, expr_types):
    tx = gen_type(expr_types[id(expr)], env, engine)
    match expr:
        case ast.Integer(val):
            return f"val: {tx} = {val}"
        case ast.Reference(var):
            return f"val: {tx} = {var}"
        case ast.Lambda(arg, bdy):
            b = code_gen(bdy, env, engine, expr_types)
            return f"val: {tx} = lambda {arg}:\n{b}\nreturn val"
        case ast.Application(rator, rand):
            f = code_gen(rator, env, engine, expr_types)
            a = code_gen(rand, env, engine, expr_types)
            return (
                f"{f}\npush(val)\n{a}\narg = val\nfunc=pop()\nval: {tx} = func(arg)"
            )
        case _:
            raise NotImplementedError(expr)


def gen_type(t: int, env, engine, skip=None):
    if t in env:
        return env[t]
    skip = skip or set()
    match engine.types[t]:
        case "Var":
            types = {
                gen_type(s, env, engine, skip | {t})
                for s in engine.val.upsets[t]
                if s not in skip
            }
            if types: return "|".join(types)

            types = {
                gen_type(s, env, engine, skip | {t})
                for s in engine.val.downsets[t]
                if s not in skip
            }
            tv = f"t{len(env)}"
            if types:
                tv += f"<:{'&'.join(types)}"
            env[t] = tv
            return tv
        case type_heads.VInt():
            return "int"
        case type_heads.UInt():
            return "int"
        case type_heads.VFunc(a, r):
            return f"({gen_type(a, env, engine)} -> {gen_type(r, env, engine)})"
        case other:
            raise NotImplementedError(other)
