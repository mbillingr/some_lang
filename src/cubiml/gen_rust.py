from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads


class Compiler:
    def __init__(self):
        self.functions = []
        self.script = []

    def finalize(self) -> str:
        script = "\n".join(self.script)
        funcs = "\n\n".join(self.functions)
        return f"fn main() {{ {script} }}\n\n\n" + funcs

    def compile_script(self, script: ast.Script, type_mapping, engine: TypeCheckerCore):
        for stmt in script.statements:
            self.script.append(self.compile_toplevel(stmt, type_mapping, engine))

    def compile_toplevel(
        self, stmt: ast.ToplevelItem, type_mapping, engine: TypeCheckerCore
    ) -> str:
        match stmt:
            case ast.Expression() as expr:
                return self.compile_expr(expr, type_mapping, engine)
            case ast.DefineLet(var, val):
                cval = self.compile_expr(val, type_mapping, engine)
                # ty = self.compile_type(type_mapping[id(val)], type_mapping, engine)
                # return f"let {var}: {ty} = {cval};"
                return f"let {var} = {cval};"
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(
        self, expr: ast.Expression, type_mapping, engine: TypeCheckerCore
    ) -> str:
        match expr:
            case ast.Literal(True):
                return "true"
            case ast.Literal(False):
                return "false"
            case ast.Reference(var):
                return var
            case ast.Function(var, body):
                fname = f"fun{id(expr)}"

                argt = self.compile_type(
                    engine.types[type_mapping[id(expr)]].arg, type_mapping, engine
                )
                rest = self.compile_type(
                    engine.types[type_mapping[id(expr)]].ret, type_mapping, engine
                )

                args = f"{var}: {argt}"
                cbody = self.compile_expr(body, type_mapping, engine)
                code = f"fn {fname}({args}) -> {rest} {{\n    {cbody}\n}}"
                self.functions.append(code)
                return fname
            case ast.Application(fun, arg):
                f = self.compile_expr(fun, type_mapping, engine)
                a = self.compile_expr(arg, type_mapping, engine)
                return f"{f}({a})"
            case ast.Conditional(condition, consequence, alternative):
                a = self.compile_expr(condition, type_mapping, engine)
                b = self.compile_expr(consequence, type_mapping, engine)
                c = self.compile_expr(alternative, type_mapping, engine)
                return f"if {a} {{ {b} }} else {{ {c} }}"
            case _:
                raise NotImplementedError(expr)

    def compile_type(self, t: int, type_mapping, engine: TypeCheckerCore) -> str:
        match engine.types[t]:
            case "Var":
                ct = engine.find_most_concrete_type(t)
                if ct is None:
                    return "()"
                return self.compile_type(ct, type_mapping, engine)
            case type_heads.VBool():
                return "bool"
            case other:
                raise NotImplementedError(other)
