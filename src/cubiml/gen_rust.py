import functools
import subprocess
import tempfile
import uuid

from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads
from cubiml.type_checker import Bindings


STD_DEFS = """
trait Boolean: std::fmt::Debug { }
impl Boolean for bool {}

trait Apply<A,R>: std::fmt::Debug { fn apply(&self, a:A)->R; }

/// The Bottom type is never instantiated
#[derive(Debug, Copy, Clone)]
struct Bottom;

impl<A, R> Apply<A, R> for Bottom {
    fn apply(&self, _:A)->R { unreachable!() }
}

impl std::fmt::Display for Bottom {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        unreachable!()
    }
}
"""

REF = "std::rc::Rc"


class Runner:
    def __init__(self, type_mapping, engine):
        self.compiler = Compiler(type_mapping, engine)

    def run_script(self, script: ast.Script):
        self.compiler.compile_script(script)
        rust_code = self.compiler.finalize()
        rust_code = rustfmt(rust_code)
        print(rust_code)

        with tempfile.NamedTemporaryFile() as tfsrc:
            bin_name = f"/tmp/{uuid.uuid4()}"
            tfsrc.write(rust_code.encode("utf-8"))
            tfsrc.flush()
            try:
                subprocess.run(["rustc", tfsrc.name, "-o", bin_name], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(e.stdout)

        return (
            subprocess.run(bin_name, check=True, capture_output=True)
            .stdout.decode("utf-8")
            .strip()
        )


def rustfmt(src: str) -> str:
    return subprocess.run(
        "rustfmt", capture_output=True, check=True, input=src.encode("utf-8")
    ).stdout.decode("utf-8")


class Compiler:
    def __init__(self, type_mapping, engine):
        self.type_mapping = type_mapping
        self.engine = engine
        self.bindings = Bindings()
        self.definitions = []
        self.script = []
        self.traits = set()

    def finalize(self) -> str:
        if len(self.script) > 0:
            script = self.script[:-1] + [f'println!("{{}}", {self.script[-1]})']
        else:
            script = []
        script = "\n".join(script)
        funcs = "\n\n".join(self.definitions)
        typedefs = "\n\n".join(
            set.union(*(self.compile_typedef(t) for t in range(len(self.engine.types))))
        )
        traits = "\n".join(self.compile_trait(*t) for t in self.traits)
        return f"fn main() {{ {script} }}\n\n{funcs}\n\n{typedefs}\n\n{traits}\n\n{STD_DEFS}"

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
                ty = self.compile_type(self._type_of(val))
                self.bindings.insert(var, ty)
                return f"let {var} = {cval};"
            case ast.DefineLetRec(defs):
                return self.compile_letrec(defs, self.bindings)
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

                ty = self.compile_type(self._type_of(expr))
                if ty.startswith(REF):
                    b = f"{REF}::new({b})"
                    c = f"{REF}::new({c})"
                    return f"{{let tmp: {ty} = if {a} {{ {b} }} else {{ {c} }}; tmp }}"
                else:
                    return f"if {a} {{ {b} }} else {{ {c} }}"
            case ast.Record() as rec:
                return self.compile_record(rec, bindings)
            case ast.FieldAccess(field, rec):
                self.traits.add(("get", field))
                r = self.compile_expr(rec, bindings)
                return f"{r}.get_{field}()"
            case ast.Case(tag, val):
                return f"Case{tag}({self.compile_expr(val, bindings)})"
            case ast.Match(val, arms):
                result = [
                    f"{{ let tmp: &dyn std::any::Any = &{self.compile_expr(val, bindings)};",
                ]
                for arm in arms:
                    with bindings.child_scope() as bindings_:
                        bindings_.insert(arm.var, None)
                        ax = self.compile_expr(arm.bdy, bindings_)
                        result.append(
                            f"if let Some({arm.var}) = tmp.downcast_ref::<Case{arm.tag}>() {{ {ax} }} else "
                        )
                result.append("{ unreachable!() } }")
                return "\n".join(result)
            case ast.Let(var, val, body):
                valt = self.compile_type(self._type_of(val))
                with bindings.child_scope() as bindings_:
                    bindings_.insert(var, valt)
                    cbody = self.compile_expr(body, bindings_)
                cval = self.compile_expr(val, bindings)
                return f"{{let {var} = {cval}; {cbody} }}"
            case ast.LetRec(defs, body):
                with bindings.child_scope() as bindings_:
                    return (
                        "{"
                        + self.compile_letrec(defs, bindings_)
                        + self.compile_expr(body, bindings_)
                        + "}"
                    )
            case _:
                raise NotImplementedError(expr)

    def compile_record(self, expr: ast.Record, bindings):
        field_names = [f[0] for f in expr.fields]
        field_values = [self.compile_expr(f[1], bindings) for f in expr.fields]
        init_fields = ",".join(f"{k}:{v}" for k, v in zip(field_names, field_values))

        name = self.compile_type(self._type_of(expr))
        return f"{name} {{ {init_fields} }}"

    def compile_closure(self, expr: ast.Function, bindings: Bindings) -> str:
        name = f"Closure{id(expr)}"

        def compile_parts():
            fty = self.engine.types[self._type_of(expr)]
            argt = self.compile_type(fty.arg)
            rett = self.compile_type(fty.ret)
            bdyt = self.compile_type(self._type_of(expr.body))
            assert bdyt == rett
            with bindings.child_scope() as bindings_:
                bindings_.insert(expr.var, argt)
                body = self.compile_expr(expr.body, bindings_)
            return body, argt, rett

        def gen_code(body, argt, rett, fvs):
            fields = "\n".join(f"{var}: {bindings.get(var)}" for var in fvs)
            definition = f"#[derive(Debug)] struct {name} {{ {fields} }}"
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

    def compile_letrec(self, defs, bindings):
        for d in defs:
            ty = self.compile_type(self._type_of(d.fun))
            bindings.insert(d.name, ty)
        allocs = []
        inits = []
        for d in defs:
            a, i = self.compile_letrec_closure(d.name, d.fun, bindings)
            allocs.append(a)
            inits.append(i)

        vars = ", ".join(d.name for d in defs)
        return (
            f"let ({vars}) = unsafe {{"
            + "// Abusing uninitialized memory and risking Undefined Behavior sucks, "
            + "but I don't know a better way to initialize recursive bindings \n"
            + "\n".join(allocs + inits)
            + f"({vars}) }};"
        )

    def compile_letrec_closure(
        self, fname: str, expr: ast.Function, bindings: Bindings
    ) -> tuple[str, str]:
        def compile_parts():
            fty = self.engine.types[self._type_of(expr)]
            argt = self.compile_type(fty.arg)
            rett = self.compile_type(fty.ret)
            bdyt = self.compile_type(self._type_of(expr.body))
            assert bdyt == rett
            with bindings.child_scope() as bindings_:
                bindings_.insert(expr.var, argt)
                body = self.compile_expr(expr.body, bindings_)
            return body, argt, rett

        def gen_code(body, argt, rett, fvs):
            fields = "\n".join(f"{var}: {bindings.get(var)}" for var in fvs)
            definition = f"#[derive(Debug)] struct {name} {{ {fields} }}"
            prelude = "".join(f"let {var} = self.{var}.clone();\n" for var in fvs)
            implementation = (
                f"impl Apply<{argt}, {rett}> for {name}\n"
                f"{{ fn apply(&self, {expr.var}: {argt}) -> {rett} {{ {prelude} {body} }} }}"
                f"impl std::fmt::Display for {name} "
                f"{{ fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result "
                f'{{write!(f, "<fun>") }} }}'
            )

            return definition, implementation

        name = f"Closure{id(expr)}"

        fvs = set(ast.free_vars(expr))
        defn, impl = gen_code(*compile_parts(), fvs)

        self.definitions.append(defn)
        self.definitions.append(impl)
        field_alloc = ",".join(
            f"{v}: std::mem::MaybeUninit::uninit().assume_init()" for v in fvs
        )
        field_init = (
            f"{{let tmp: {bindings.get(v)} = {v}.clone();"
            f"std::ptr::write(&{fname}.{v} as *const _ as *mut _, tmp);}}"
            for v in fvs
        )

        return (
            f"let {fname} = {REF}::new({name} {{ {field_alloc} }});",
            "\n".join(field_init),
        )

    @functools.lru_cache(1024)
    def compile_type(self, t: int) -> str:
        match self.engine.types[t]:
            case "Var":
                vts = list(self.engine.all_upvtypes(t))
                tys = [self.compile_type(ty) for ty in vts]
                match tys:
                    case []:
                        return "Bottom"
                    case [tc]:
                        return tc
                    case _:
                        if all(tys[0] == tc for tc in tys):
                            return tys[0]
                        traits = (self.traits_for_type(t) for t in vts)
                        common_traits = set.intersection(*traits)
                        return f"{REF}<dyn {'+'.join(common_traits)}>"
            case type_heads.VBool():
                return "bool"
            case type_heads.VFunc(arg, ret):
                a = self.compile_type(arg)
                r = self.compile_type(ret)
                return f"{REF}<dyn Apply<{a}, {r}>>"
            case type_heads.VObj(_):
                return f"Record{t}"
            case other:
                raise NotImplementedError(other)

    @functools.lru_cache(1024)
    def traits_for_type(self, t: int) -> set[str]:
        match self.engine.types[t]:
            case type_heads.VBool():
                return {"Boolean"}
            case type_heads.VFunc(arg, ret):
                a = self.compile_type(arg)
                r = self.compile_type(ret)
                return {f"Apply<{a}, {r}>"}
            case type_heads.VObj(fields):
                fts = ((f, self.compile_type(t)) for f, t in fields.items())
                return {f"Has{f}<{ty}>" for f, ty in fts}
            case other:
                raise NotImplementedError(other)

    def compile_typedef(self, t: int) -> set[str]:
        match self.engine.types[t]:
            case "Var" | type_heads.VBool():
                return set()
            case type_heads.VFunc():
                return set()
            case type_heads.VObj(fields):
                ty = self.compile_type(t)
                fds = ",".join(
                    f"{f}: {self.compile_type(t)}" for f, t in fields.items()
                )
                tdef = f"#[derive(Debug, Clone)] struct {ty} {{ {fds} }}"

                impls = []
                for f, t in fields.items():
                    self.traits.add(("get", f))
                    ft = self.compile_type(t)
                    impls.append(
                        (
                            f"impl Has{f}<{ft}> for {ty} "
                            f"{{ fn get_{f}(&self) -> {ft} {{ self.{f}.clone() }} }}"
                        )
                    )
                rfields = [f for f, _ in fields.items()][::-1]
                fmt = "; ".join(f"{f}={{}}" for f in rfields)
                fvals = ", ".join(f"self.{f}" for f in rfields)
                impls.append(
                    (
                        f"impl std::fmt::Display for {ty} "
                        f"{{ fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{"
                        f'write!(f, "{{{{")?;'
                        f'write!(f, "{fmt}", {fvals})?;'
                        f'write!(f, "}}}}") }} }}'
                    )
                )

                return {"\n".join([tdef] + impls)}
            case type_heads.VCase(tag, ty):
                return {
                    f"#[derive(Debug, Clone)] struct Case{tag}({self.compile_type(ty)});"
                }
            case type_heads.UCase(cases) as uc:
                defs = set()
                for tag, u in cases.items():
                    defs.add(
                        f"#[derive(Debug, Clone)] struct Case{tag}({self.compile_type(u)});"
                    )
                return defs
            case type_heads.UTypeHead():
                return set()
            case other:
                raise NotImplementedError(other)

    def compile_trait(self, *args) -> str:
        match args:
            case ("get", field):
                trait_def = (
                    f"trait Has{field}<T>: std::fmt::Display "
                    f"{{ fn get_{field}(&self) -> T; }}"
                )
                bot_impl = (
                    f"impl<T> Has{field}<T> for Bottom "
                    f"{{ fn get_{field}(&self) -> T {{unreachable!()}} }}"
                )
                return "\n".join((trait_def, bot_impl))

    def _type_of(self, expr: ast.Expression) -> int:
        return self.type_mapping[id(expr)]
