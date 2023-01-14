from __future__ import annotations
from typing import Any, Optional

from some_lang import interpreter, parser, type_checker, ast


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

    @staticmethod
    def module(src: str | ast.Module) -> Context:
        if not isinstance(src, ast.Module):
            src = parser.parse_module(src)
        engine = type_checker.TypeCheckerCore()
        type_env = type_checker.check_module(src, engine)
        env = interpreter.run_module(src)
        return Context(env, type_env, engine)
