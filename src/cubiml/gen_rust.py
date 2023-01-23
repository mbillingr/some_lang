import functools

from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads
from cubiml.type_checker import Bindings


STD_DEFS = """
trait Apply<A,R> { fn apply(&self, a:A)->R; }
"""

REF = "std::rc::Rc"


class Compiler:
    def __init__(self, type_mapping, engine):
        self.type_mapping = type_mapping
        self.engine = engine
        self.bindings = Bindings()
        self.definitions = []
        self.script = []

    def finalize(self) -> str:
        if len(self.script) > 0:
            script = self.script[:-1] + [f'println!("{{:?}}", {self.script[-1]})']
        else:
            script = []
        script = "\n".join(script)
        funcs = "\n\n".join(self.definitions)
        return f"fn main() {{ {script} }}\n\n{funcs}\n\n{STD_DEFS}"

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.append(self.compile_toplevel(stmt))
        self.bindings.changes.clear()

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> str:
        match stmt:
            case ast.Expression() as expr:
                return self.compile_expr(expr, self.bindings)
            case ast.DefineLet(var, val):
                cval = self.compile_expr(val, self.bindings)
                ty = self.compile_type(self.type_mapping[id(val)])
                self.bindings.insert(var, ty)
                return f"let {var} = {cval};"
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression, bindings: Bindings) -> str:
        match expr:
            case ast.Literal(True):
                return "true"
            case ast.Literal(False):
                return "false"
            case ast.Reference(var):
                return var
            case ast.Function() as fun:
                return self.compile_closure(fun, bindings)
            case ast.Application(fun, arg):
                f = self.compile_expr(fun, bindings)
                a = self.compile_expr(arg, bindings)
                return f"({f}).apply({a})"
            case ast.Conditional(condition, consequence, alternative):
                a = self.compile_expr(condition, bindings)
                b = self.compile_expr(consequence, bindings)
                c = self.compile_expr(alternative, bindings)
                return f"if {a} {{ {b} }} else {{ {c} }}"
            case _:
                raise NotImplementedError(expr)

    def compile_closure(self, expr: ast.Function, bindings: Bindings) -> str:
        name = f"Closure{id(expr)}"

        def compile_parts():
            fty = self.engine.types[self.type_mapping[id(expr)]]
            argt = self.compile_type(fty.arg)
            rett = self.compile_type(fty.ret)
            with bindings.child_scope() as bindings_:
                bindings_.insert(expr.var, argt)
                body = self.compile_expr(expr.body, bindings_)
            return body, argt, rett

        def gen_code(body, argt, rett, fvs):
            fields = "\n".join(f"{var}: {bindings.get(var)}" for var in fvs)
            definition = f"struct {name} {{ {fields} }}"
            prelude = "".join(f"let {var} = self.{var}.clone();\n" for var in fvs)
            implementation = (
                f"impl Apply<{argt}, {rett}> for {name}\n"
                f"{{ fn apply(&self, {expr.var}: {argt}) -> {rett} {{ {prelude} {body} }} }}"
            )

            return definition, implementation

        fvs = set(ast.free_vars(expr))
        defn, impl = gen_code(*compile_parts(), fvs)

        self.definitions.append(defn)
        self.definitions.append(impl)
        field_init = ",".join(fvs)
        return f"{REF}::new({name} {{ {field_init} }})"

    @functools.lru_cache(1024)
    def compile_type(self, t: int) -> str:
        match self.engine.types[t]:
            case "Var":
                ct = self.engine.find_most_concrete_type(t)
                if ct is None:
                    return "()"
                return self.compile_type(ct)
            case type_heads.VBool():
                return "bool"
            case type_heads.VFunc(arg, ret):
                a = self.compile_type(arg)
                r = self.compile_type(ret)
                return f"{REF}<dyn Apply<{a}, {r}>>"
            case other:
                raise NotImplementedError(other)
