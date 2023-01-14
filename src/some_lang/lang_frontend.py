from typing import Any

from some_lang import interpreter, parser, type_checker, ast


class Context:
    def __init__(self):
        self.engine = type_checker.TypeCheckerCore()
        self.type_env = type_checker.EmptyEnv()
        self.env = interpreter.EmptyEnv()

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
