from __future__ import annotations

import abc
import contextlib
import dataclasses
import functools
import subprocess
import tempfile
import uuid
from typing import Optional

from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads


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

impl<A, R, T: Apply<A, R>> Apply<A, R> for Option<T> {
    fn apply(&self, a: A) -> R {
        self.as_ref().unwrap().apply(a)
    }
}

impl<A, R, T: Apply<A, R>> Apply<A, R> for std::cell::RefCell<T> {
    fn apply(&self, a: A) -> R {
        self.borrow().apply(a)
    }
}

trait Show {
    fn show(&self) -> String;
}

impl Show for Bottom {
    fn show(&self) -> String {
        unreachable!()
    }
}

impl Show for bool {
    fn show(&self) -> String {
        format!("{}", self)
    }
}

impl<T: Show> Show for std::rc::Rc<T>
{
    fn show(&self) -> String {
        (**self).show()
    }
}

impl<T: Show> Show for std::cell::RefCell<T>
{
    fn show(&self) -> String {
        self.borrow().show()
    }
}

impl<T: Show> Show for std::option::Option<T>
{
    fn show(&self) -> String {
        match self {
            std::option::Option::None => "<uninitialized>".to_string(),
            std::option::Option::Some(x) => x.show()
        }
    }
}
"""

REF = "std::rc::Rc"
CEL = "std::cell::RefCell"
OPT = "std::option::Option"


class Runner:
    def __init__(self, type_mapping, engine):
        self.compiler = Compiler(type_mapping, engine)

    def run_script(self, script: ast.Script):
        self.compiler.compile_script(script)
        rust_ast = self.compiler.finalize()
        rust_code = str(rust_ast)
        try:
            rust_code = rustfmt(rust_code)
        finally:
            print(rust_code)

        with tempfile.NamedTemporaryFile() as tfsrc:
            bin_name = f"/tmp/{uuid.uuid4()}"
            tfsrc.write(rust_code.encode("utf-8"))
            tfsrc.flush()
            try:
                subprocess.run(
                    ["rustc", tfsrc.name, "-o", bin_name, "-C", "opt-level=0"],
                    check=True,
                )
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

    def finalize(self) -> RustAst:
        if len(self.script) > 0:
            script = self.script[:-1] + [f'println!("{{}}", {self.script[-1]}.show())']
        else:
            script = []
        script = RustBlock(script)
        typedefs = set.union(
            *(self.compile_typedef(t) for t in range(len(self.engine.types)))
        )
        traits = (self.compile_trait(*t) for t in self.traits)
        return RustToplevel(
            [
                RustFn("main", [], None, script),
                *self.definitions,
                *typedefs,
                *traits,
                RustInline(STD_DEFS),
            ]
        )

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.extend(self.compile_toplevel(stmt))
        self.bindings.changes.clear()

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> list[RustExpr]:
        match stmt:
            case ast.Expression() as expr:
                return [self.compile_expr(expr, self.bindings)]
            case ast.DefineLet(var, val):
                cval = self.compile_expr(val, self.bindings)
                ty = self.compile_type(self._type_of(val))
                self.bindings.insert(var, ty)
                return [RustInline(f"let {var} = {cval}")]
            case ast.DefineLetRec(defs):
                return self.compile_letrec(defs, self.bindings)
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression, bindings: Bindings) -> RustExpr:
        match expr:
            case ast.Literal(True):
                return RustLiteral("true")
            case ast.Literal(False):
                return RustLiteral("false")
            case ast.Reference(var):
                return RustLiteral(var)
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
                if isinstance(ty, RustObjType):
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
                    return RustBlock(
                        self.compile_letrec(defs, bindings_)
                        + [self.compile_expr(body, bindings_)]
                    )
            case _:
                raise NotImplementedError(expr)

    def compile_record(self, expr: ast.Record, bindings):
        field_names = [f[0] for f in expr.fields]
        field_values = [self.compile_expr(f[1], bindings) for f in expr.fields]
        init_fields = ",".join(f"{k}:{v}" for k, v in zip(field_names, field_values))

        name = self.compile_type(self._type_of(expr))
        return f"{name} {{ {init_fields} }}"

    def compile_closure(self, expr: ast.Function, bindings: Bindings) -> RustExpr:
        name = f"Closure{self._type_of(expr)}"

        fvs = set(ast.free_vars(expr))
        defn, impl = self._gen_closure_code(
            name, expr.var, *self._compile_closure_parts(expr, bindings), fvs, bindings
        )

        self.definitions.append(defn)
        self.definitions.append(impl)
        return RustNewObj(RustNewStruct(name, list(zip(fvs, map(RustLiteral, fvs)))))

    def compile_letrec(self, defs, bindings) -> [RustAst]:
        for d in defs:
            ty = self.compile_type(self._type_of(d.fun))
            bindings.insert(d.name, ty)
        allocs = []
        inits = []
        for d in defs:
            a, i = self.compile_letrec_closure(d.name, d.fun, bindings)
            allocs.append(a)
            inits.append(i)

        return (
            [
                RustComment(
                    "TODO we're leaking memory here because of all the mutual references. Is there a better way?"
                )
            ]
            + allocs
            + inits
        )

    def compile_letrec_closure(
        self, fname: str, expr: ast.Function, bindings: Bindings
    ) -> tuple[str, str]:
        name = f"Closure{self._type_of(expr)}"

        fvs = set(ast.free_vars(expr))
        defn, impl = self._gen_closure_code(
            name, expr.var, *self._compile_closure_parts(expr, bindings), fvs, bindings
        )

        self.definitions.append(defn)
        self.definitions.append(impl)
        field_init = ", ".join(f"{v}: {v}.clone()" for v in fvs)

        return (
            f"let {fname} = {REF}::new({CEL}::new({OPT}::None));",
            f"*{fname}.borrow_mut() = {OPT}::Some({name} {{ {field_init} }});",
        )

    def _compile_closure_parts(
        self, expr: ast.Function, bindings: Bindings
    ) -> tuple[str, str, str]:
        fty = self.engine.types[self._type_of(expr)]
        argt = self.compile_type(fty.arg)
        rett = self.compile_type(fty.ret)
        bdyt = self.compile_type(self._type_of(expr.body))
        assert bdyt == rett
        with bindings.child_scope() as bindings_:
            bindings_.insert(expr.var, argt)
            body = self.compile_expr(expr.body, bindings_)
        return body, argt, rett

    def _gen_closure_code(
        self,
        name: str,
        argn: str,
        body: str,
        argt: str,
        rett: str,
        fvs: set[str],
        bindings: Bindings,
    ):
        fields = "\n".join(f"{var}: {bindings.get(var)}" for var in fvs)
        definition = f"#[derive(Debug)] struct {name} {{ {fields} }}"
        prelude = "".join(f"let {var} = self.{var}.clone();\n" for var in fvs)
        implementation = (
            f"impl Apply<{argt}, {rett}> for {name}\n"
            f"{{ fn apply(&self, {argn}: {argt}) -> {rett} {{ {prelude} {body} }} }}"
            f"impl Show for {name} "
            f"{{ fn show(&self) -> String "
            f'{{ "<fun>".to_string() }} }}'
        )

        return definition, implementation

    @functools.lru_cache(1024)
    def compile_type(self, t: int) -> RustType:
        match self.engine.types[t]:
            case "Var":
                vts = list(self.engine.all_upvtypes(t))
                tys = [self.compile_type(ty) for ty in vts]
                match tys:
                    case []:
                        return RustAtomicType("Bottom")
                    case [tc]:
                        return tc
                    case _:
                        if all(tys[0] == tc for tc in tys):
                            return tys[0]
                        traits = (self.traits_for_type(t) for t in vts)
                        common_traits = set.intersection(*traits)
                        return RustObjType(common_traits)
            case type_heads.VBool():
                return RustAtomicType("bool")
            case type_heads.VFunc(arg, ret):
                a = self.compile_type(arg)
                r = self.compile_type(ret)
                return RustObjType({RustTrait("Apply", (a, r))})
            case type_heads.VObj(_):
                return RustAtomicType(f"Record{t}")
            case other:
                raise NotImplementedError(other)

    @functools.lru_cache(1024)
    def traits_for_type(self, t: int) -> set[RustTrait]:
        match self.engine.types[t]:
            case type_heads.VBool():
                return {RustTrait("Boolean")}
            case type_heads.VFunc(arg, ret):
                a = self.compile_type(arg)
                r = self.compile_type(ret)
                return {RustTrait("Apply", (a, r))}
            case type_heads.VObj(fields):
                fts = ((f, self.compile_type(t)) for f, t in fields.items())
                return {RustTrait(f"Has{f}", (ty,)) for f, ty in fts}
            case other:
                raise NotImplementedError(other)

    def compile_typedef(self, t: int) -> set[RustAst]:
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
                fvals = ", ".join(f"self.{f}.show()" for f in rfields)
                impls.append(
                    (
                        f"impl Show for {ty} "
                        f'{{ fn show(&self) -> String {{ format!("{{{{{fmt}}}}}", {fvals})  }} }}'
                    )
                )

                return {RustInline("\n".join([tdef] + impls))}
            case type_heads.VCase(tag, ty):
                return {
                    RustInline(
                        f"#[derive(Debug, Clone)] struct Case{tag}({self.compile_type(ty)});"
                    )
                }
            case type_heads.UCase(cases) as uc:
                defs = set()
                for tag, u in cases.items():
                    defs.add(
                        RustInline(
                            f"#[derive(Debug, Clone)] struct Case{tag}({self.compile_type(u)});"
                        )
                    )
                return defs
            case type_heads.UTypeHead():
                return set()
            case other:
                raise NotImplementedError(other)

    def compile_trait(self, *args) -> RustAst:
        match args:
            case ("get", field):
                trait_def = (
                    f"trait Has{field}<T>: Show " f"{{ fn get_{field}(&self) -> T; }}"
                )
                bot_impl = (
                    f"impl<T> Has{field}<T> for Bottom "
                    f"{{ fn get_{field}(&self) -> T {{unreachable!()}} }}"
                )
                return RustInline("\n".join((trait_def, bot_impl)))

    def _type_of(self, expr: ast.Expression) -> int:
        return self.type_mapping[id(expr)]


class Bindings:
    def __init__(self):
        self.m: dict[str, str] = {}
        self.changes: list[tuple[str, Optional[str]]] = []

    def get(self, k: str) -> str:
        return self.m[k]

    def insert(self, k: str, v: str):
        old = self.m.get(k)
        self.changes.append((k, old))
        self.m[k] = v

    @contextlib.contextmanager
    def child_scope(self):
        n = len(self.changes)
        try:
            yield self
        finally:
            self.unwind(n)

    def unwind(self, n):
        while len(self.changes) > n:
            k, old = self.changes.pop()
            if old is None:
                del self.m[k]
            else:
                self.m[k] = old


class RustAst(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass


@dataclasses.dataclass(frozen=True)
class RustComment(RustAst):
    text: str

    def __str__(self) -> str:
        return f"/* {self.text} */"


@dataclasses.dataclass(frozen=True)
class RustInline(RustAst):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclasses.dataclass(frozen=True)
class RustToplevel(RustAst):
    items: list[RustAst]

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.items))


@dataclasses.dataclass(frozen=True)
class RustTrait(RustAst):
    name: str
    params: tuple[RustAst] = ()

    def __str__(self) -> str:
        if not self.params:
            return self.name
        return self.name + "<" + ", ".join(map(str, self.params)) + ">"


class RustType(RustAst):
    pass


@dataclasses.dataclass(frozen=True)
class RustAtomicType(RustType):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclasses.dataclass(frozen=True)
class RustObjType(RustType):
    interfaces: set[RustTrait]

    def __str__(self) -> str:
        ifs = "+".join(map(str, self.interfaces))
        return f"{REF}<dyn {ifs}>"


class RustExpr(RustAst):
    pass


@dataclasses.dataclass(frozen=True)
class RustBlock(RustExpr):
    items: list[RustExpr]

    def __str__(self) -> str:
        return "".join(["{", ";".join(map(str, self.items)), "}"])


@dataclasses.dataclass(frozen=True)
class RustLiteral(RustExpr):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclasses.dataclass(frozen=True)
class RustNewObj(RustExpr):
    inner: RustExpr

    def __str__(self) -> str:
        return f"std::rc::Rc::new({self.inner})"


@dataclasses.dataclass(frozen=True)
class RustNewStruct(RustExpr):
    name: str
    fields: list[tuple[str, RustExpr]]

    def __str__(self) -> str:
        fis = ", ".join(f"{f}:{v}" for f, v in self.fields)
        return f"{self.name} {{ {fis} }}"


@dataclasses.dataclass(frozen=True)
class RustFn(RustAst):
    name: str
    args: list[tuple[(str, RustType)]]
    rett: Optional[RustType]
    body: RustBlock

    def __str__(self) -> str:
        args = ", ".join(f"{a}:{t}" for a, t in self.args)
        if self.rett is None:
            return f"fn {self.name}({args}) {self.body}"
        else:
            return f"fn {self.name}({args}) -> {self.rett} {self.body}"
