from typing import Any

from some_lang import interpreter, parser, type_checker, ast


class Context:
    def __init__(self):
        self.engine = type_checker.TypeCheckerCore()
        self.type_env = type_checker.EmptyEnv()
        self.env = interpreter.EmptyEnv()

    def define(self, name: str, val: interpreter.Value, ty: ast.TypeExpression):
        self.type_env = self.type_env.extend(name, type_checker.eval_vtype(ty, self.engine))
        self.env = self.env.extend(name, val)

    def eval(self, src: str) -> Any:
        ast = parser.parse_expr(src)
        type_checker.check_expr(ast, self.type_env, self.engine)
        return interpreter.evaluate(ast, self.env)
