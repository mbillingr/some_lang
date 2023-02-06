from __future__ import annotations

import abc
import dataclasses
import functools
import itertools
import subprocess
import tempfile
import uuid

import typing

from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads

STD_HEADER = """
"""

STD_DEFS = """
use base::Func;

mod base {    
    pub type Ref<T> = std::rc::Rc<T>;
    
    #[derive(Clone)]
    pub enum Value {
        Bool(bool),
        Record(std::collections::HashMap<&'static str, Value>),
        Function(Ref<dyn Func>),
    }
    
    impl std::fmt::Debug for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Value::Bool(b) => b.fmt(f),
                Value::Record(r) => {
                    write!(f, "{{")?;
                    let mut keys: Vec<_> = r.keys().collect();
                    keys.sort();
                    let fields: Vec<_> = keys
                        .into_iter()
                        .map(|k| format!("{}={:?}", k, r[k]))
                        .collect();
                    write!(f, "{}", fields.join("; "))?;
                    write!(f, "}}")
                }
                Value::Function(fun) => write!(f, "<fun {:p}>", fun)
            }
        }
    }
    
    impl From<bool> for Value {
        fn from(b: bool) -> Self {
            Value::Bool(b)
        }
    }
    
    impl From<Value> for bool {
        fn from(v: Value) -> Self {
            match v {
                Value::Bool(b) => b,
                _ => panic!("Not a boolean: {:?}", v),
            }
        }
    }
    
    impl From<std::collections::HashMap<&'static str, Value>> for Value {
        fn from(r: std::collections::HashMap<&'static str, Value>) -> Self {
            Value::Record(r)
        }
    }
    
    impl<F: 'static + Func> From<F> for Value {
        fn from(f: F) -> Self {
            Value::Function(Ref::new(f))
        }
    }
    
    impl Value {
        pub fn apply(&self, a: impl Into<Value>) -> Value {
            match self {
                Value::Function(f) => f.apply(a.into()),
                _ => panic!("Not a function: {:?}", self),
            }
        }
        
        pub fn get(&self, field: &'static str) -> Value {
            match self {
                Value::Record(r) => r[field].clone(),
                _ => panic!("Not a record: {:?}", self),
            }
        }
    }

    /// The supertype of all types.
    pub trait Top: 'static + std::fmt::Debug {}
    impl<T: 'static + std::fmt::Debug> Top for T {}
    
    pub trait Bool: std::fmt::Debug {
        fn is_true(&self) -> bool;
    }
    
    impl Bool for bool {
        fn is_true(&self) -> bool { *self }
    }
    
    pub trait Func {
        fn apply(&self, a: Value) -> Value;
    }

    pub fn fun<F>(f: F) -> Function<F> {
        Function(f)
    }
    
    pub struct Function<F>(F);

    impl<F> Func for Function<F>
    where F: 'static + Fn(Value)->Value
    {
        fn apply(&self, a: Value) -> Value {
            (self.0)(a)
        }
    }
    
    impl<F> std::fmt::Debug for Function<F> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "<fun>")
        }
    }
    
    pub trait Record: Top {}
}
"""


class Runner:
    def __init__(self, type_mapping, engine: TypeCheckerCore):
        self.compiler = Compiler(type_mapping, engine)

    def run_script(self, script: ast.Script):
        self.compiler.compile_script(script)
        cpp_ast = self.compiler.finalize()
        cpp_code = str(cpp_ast)
        try:
            cpp_code = rustfmt(cpp_code)
        finally:
            print(cpp_code)

        with tempfile.NamedTemporaryFile(suffix=".rs") as tfsrc:
            bin_name = f"/tmp/{uuid.uuid4()}"
            tfsrc.write(cpp_code.encode("utf-8"))
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
    def __init__(self, type_mapping, engine: TypeCheckerCore):
        self.type_mapping = type_mapping
        self.engine = engine
        self.script: list[RsStatement] = []
        self.common_trait_defs: set[RsAst] = set()
        self.toplevel_defs: list[RsAst] = []

    def finalize(self) -> RsAst:
        script = self.script.copy()
        if len(self.script) > 0:
            match script:
                case [*_, RsExprStatement(expr)]:
                    script[-1] = RsInline(f'println!("{{:?}}", {expr});')

        return RsToplevel(
            [
                RsInline(STD_HEADER),
                RsFun("main", [], None, RsInline('println!("{:?}", script());')),
                RsFun(
                    "script",
                    [],
                    RsValue(),
                    RsIntoValue(RsBlock(self.script[:-1], self.script[-1])),
                ),
                RsToplevelGroup(self.toplevel_defs),
                # RsToplevelGroup(self.gen_type_defs()),
                # RsToplevelGroup(list(self.common_trait_defs)),
                RsInline(STD_DEFS),
            ]
        )

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.extend(self.compile_toplevel(stmt))

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> list[RsStatement]:
        match stmt:
            case ast.Expression() as expr:
                return [self.compile_expr(expr)]
            case ast.DefineLet(var, val):
                if isinstance(val, ast.Function):
                    v = self.compile_function(val, name=f"{var}")
                else:
                    v = self.compile_expr(val)
                return [RsLetStatement(var, v)]
            case ast.DefineLetRec():
                return self.compile_letrec(stmt)
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression) -> RsExpression:
        match expr:
            case ast.Literal(True):
                return RsLiteral("true")
            case ast.Literal(False):
                return RsLiteral("false")
            case ast.Reference(var):
                print(
                    f"Dereferencing {var} @ {id(expr)} of {self.type_of(expr)} -- {self.engine.types[self.type_of(expr)]}"
                )
                return RsReference(var)
            case ast.Conditional(condition, consequence, alternative):
                a = self.compile_expr(condition)
                b = self.compile_expr(consequence)
                c = self.compile_expr(alternative)
                return RsIfExpr(a, b, c)
            case ast.Function():
                return self.compile_function(expr, name=f"fun{id(expr)}")
            case ast.Application(fun, arg):
                f = self.compile_expr(fun)
                a = self.compile_expr(arg)
                return RsApply(f, a)
            case ast.Record(fields):
                field_initializers = {f: self.compile_expr(v) for f, v in fields}
                tname = self.get_type(self.type_of(expr))
                assert isinstance(tname, RsRecordType)
                return RsNewRecord(tname, field_initializers)
            case ast.FieldAccess(field, obj):
                return RsGetField(field, self.compile_expr(obj))
            case ast.Let(var, val, body):
                if isinstance(val, ast.Function):
                    v = self.compile_function(val, name=f"{var}{id(expr)}")
                else:
                    v = self.compile_expr(val)
                b = self.compile_expr(body)
                return RsBlock([RsLetStatement(var, v)], b)
            case ast.LetRec():
                return RsBlock(
                    self.compile_letrec(expr),
                    self.compile_expr(expr.body),
                )
            case _:
                raise NotImplementedError(expr)

    def compile_function(self, fun: ast.Function, name: str) -> RsExpression:
        argt = self.v_name(self.engine.types[self.type_of(fun)].arg)
        rett = self.v_name(self.engine.types[self.type_of(fun)].ret)
        fvs = {
            v: RsLiteral(self.v_name(ty)) for v, ty in free_vars(fun, self.type_mapping)
        }
        fndefs = [(name, fun.var, argt, rett, self.compile_expr(fun.body))]
        self.toplevel_defs.append(RsMutualClosure(name, fvs, fndefs))

        capture = ", ".join(f"{v}:{v}.clone()" for v in fvs)
        return RsBlock(
            [
                RsLetStatement(
                    "cls", RsNewObj(RsInline(f"{name}::Closure {{ {capture} }}")), None
                ),
            ],
            RsClosure(
                fun.var,
                RsInline(argt),
                RsInline(rett),
                RsInline(f"{name}::{name}({fun.var}, &*cls)"),
            ),
        )

    def compile_letrec(self, expr: ast.LetRec) -> list[RsStatement]:
        bind = expr.bind
        name = f"letrec{id(expr)}"
        bound_names = set(fdef.name for fdef in bind)
        fvs = itertools.chain(
            *(free_vars(fdef.fun, self.type_mapping) for fdef in bind)
        )
        fvs = {v: RsLiteral(self.v_name(ty)) for v, ty in fvs if v not in bound_names}
        arg_types = [
            self.v_name(self.engine.types[self.type_of(fdef.fun)].arg) for fdef in bind
        ]
        ret_types = [
            self.v_name(self.engine.types[self.type_of(fdef.fun)].ret) for fdef in bind
        ]
        fndefs = [
            (
                fdef.name,
                fdef.fun.var,
                argt,
                rett,
                replace_calls(bound_names, self.compile_expr(fdef.fun.body)),
            )
            for fdef, argt, rett in zip(bind, arg_types, ret_types)
        ]
        self.toplevel_defs.append(RsMutualClosure(name, fvs, fndefs))
        capture = ", ".join(f"{v}:{v}.clone()" for v in fvs)
        statements = [
            RsLetStatement(
                "cls", RsNewObj(RsInline(f"{name}::Closure {{ {capture} }}")), None
            ),
            *(
                RsLetStatement(
                    fdef.name,
                    RsBlock(
                        [RsInline("let cls = cls.clone();")],
                        RsClosure(
                            fdef.fun.var,
                            RsInline(aty),
                            RsInline(rty),
                            RsInline(f"{name}::{fdef.name}({fdef.fun.var}, &*cls)"),
                        ),
                    ),
                )
                for fdef, aty, rty in zip(bind, arg_types, ret_types)
            ),
        ]
        return statements

    @functools.lru_cache
    def get_type(self, t: int | ast.Expression) -> RsType:
        if not isinstance(t, int):
            return self.get_type(self.type_of(t))

        match self.engine.types[t]:
            case type_heads.VObj(fields):
                field_types = {
                    fn: RsLiteral(self.v_name(ft)) for fn, ft in fields.items()
                }
                return RsRecordType(self.r_name(t), field_types)
            case ty:
                raise NotImplementedError(ty)

    def gen_type_defs(self) -> list[RsAst]:
        defs = []
        for t, ty in enumerate(self.engine.types):
            match ty:
                case "erased":
                    continue
                case "Var":
                    bounds = [RsInline("base::Top")]
                case type_heads.VBool() | type_heads.UBool():
                    bounds = [RsInline("base::Top"), RsInline("base::Bool")]
                case type_heads.VFunc(arg, ret) | type_heads.UFunc(arg, ret):
                    a = self.v_name(arg)
                    r = self.v_name(ret)
                    bounds = [RsInline("base::Top"), RsInline(f"base::Func<{a},{r}>")]
                case type_heads.VObj(fields):
                    bounds = [RsInline("base::Top"), RsInline("base::Record")]
                    defs.append(
                        RsRecordDefinition(
                            RsRecordType(
                                self.r_name(t),
                                {f: self.v_name(ft) for f, ft in fields.items()},
                            )
                        )
                    )
                    for f, _ in fields.items():
                        self.common_trait_defs.add(RsHasFieldDefinition(f))
                case type_heads.UObj(field, ft):
                    f = self.v_name(ft)
                    bounds = [RsInline("base::Top"), RsInline(f"Has{field}<{f}>")]
                    self.common_trait_defs.add(RsHasFieldDefinition(field))
                case _:
                    raise NotImplementedError(ty)

            inferred_bounds = map(self.u_name, self.engine.r.downsets[t])
            defs.append(
                RsObjTypeDef(
                    self.v_name(t),
                    self.u_name(t),
                    {*bounds, *inferred_bounds},
                )
            )
        return defs

    def v_name(self, t: int) -> str:
        return f"V{t}"

    def u_name(self, t: int) -> str:
        return f"U{t}"

    def r_name(self, t: int) -> str:
        return f"R{t}"

    def type_of(self, expr: ast.Expression) -> int:
        return self.type_mapping[id(expr)]


class RsAst(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    pass


@dataclasses.dataclass
class RsToplevel(RsAst):
    items: list[RsAst]

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.items))


@dataclasses.dataclass
class RsToplevelGroup(RsAst):
    items: list[RsAst]

    def __str__(self) -> str:
        return "".join(map(str, self.items))


@dataclasses.dataclass
class RsObjTypeDef(RsAst):
    type_name: str
    trait_name: str
    bounds: set[RsTrait]

    def __str__(self):
        bounds = "+".join(map(str, self.bounds))
        return (
            f"type {self.type_name} = base::Ref<dyn {self.trait_name}>;\n"
            f"pub trait {self.trait_name}: {bounds} {{}}\n"
            f"impl<T: {bounds}> {self.trait_name} for T {{}}\n\n"
        )


class RsTrait(RsAst):
    pass


class RsType(RsAst):
    pass


class RsStatement(RsAst):
    pass


class RsExpression(RsAst):
    pass


@dataclasses.dataclass(frozen=True)
class RsValue(RsType):
    def __str__(self) -> str:
        return "base::Value"


@dataclasses.dataclass
class RsRecordType(RsType):
    name: str
    fields: dict[str, RsType]

    def __str__(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class RsHasFieldDefinition(RsAst):
    field: str

    def __str__(self) -> str:
        f = self.field
        return f"pub trait Has{f}<T> {{ fn get_{f}(&self) -> T; }}"


@dataclasses.dataclass
class RsRecordDefinition(RsAst):
    rtype: RsRecordType

    def __str__(self) -> str:
        ordered_fields = list(reversed(self.rtype.fields.keys()))
        fields = "\n".join(f"{fn}: {ft}," for fn, ft in self.rtype.fields.items())

        getters = []
        for fn, ft in self.rtype.fields.items():
            getters.append(
                f"impl Has{fn}<{ft}> for {self.rtype.name} "
                f"{{ fn get_{fn}(&self) -> {ft} {{ self.{fn}.clone() }} }}"
            )

        output_template = "; ".join(f"{fn}={{:?}}" for fn in ordered_fields)
        output_values = ", ".join(f"self.{fn}" for fn in ordered_fields)

        return (
            f"struct {self.rtype.name} {{ {fields} }}\n"
            f"impl base::Record for {self.rtype.name} {{}}\n"
            f"impl std::fmt::Debug for {self.rtype.name} {{"
            f"  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{"
            f'     write!(f, "{{{{{output_template}}}}}", {output_values}) }} }}'
            + "\n".join(getters)
        )


@dataclasses.dataclass
class RsMutualClosure(RsAst):
    name: str
    free: set(str)
    bind: [(str, str, RsType, RsType, RsExpression)]

    def __str__(self) -> str:
        fvs = ", ".join(f"pub {v}: base::Value" for v in self.free)
        cls = f"pub struct Closure {{ {fvs} }}"

        unclose = "".join(f"let {v} = cls.{v}.clone();" for v in self.free)

        defs = []
        for name, var, argt, rett, body in self.bind:
            defs.append(
                f"pub fn {name}({var}: base::Value, cls: &Closure) -> base::Value {{ {unclose} {body} }}"
            )
        defs = "\n".join(defs)
        return f"mod {self.name} {{ use super::*; {cls} {defs} }}"


@dataclasses.dataclass
class RsLetStatement(RsStatement):
    var: str
    val: RsExpression
    typ: typing.Optional[RsType] = RsValue()

    def __str__(self):
        if self.typ is None:
            return f"let {self.var} = {self.val};"
        else:
            return f"let {self.var} = {self.typ}::from({self.val});"


@dataclasses.dataclass
class RsExprStatement(RsStatement):
    expr: RsExpression

    def __str__(self) -> str:
        return f"{self.expr};"


@dataclasses.dataclass(frozen=True)
class RsInline(RsExpression, RsStatement, RsTrait, RsType):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclasses.dataclass
class RsBlock(RsExpression):
    stmts: list[RsStatement]
    final_expr: RsExpression

    def __str__(self) -> str:
        if not self.stmts:
            return str(self.final_expr)
        stmts = "\n".join(map(str, self.stmts))
        return f"{{ {stmts} \n {self.final_expr} }}"


@dataclasses.dataclass
class RsLiteral(RsExpression):
    value: str

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass
class RsIntoValue(RsExpression):
    value: RsExpression

    def __str__(self) -> str:
        return f"base::Value::from({self.value})"


@dataclasses.dataclass
class RsReference(RsExpression):
    var: str

    def __str__(self) -> str:
        return f"{self.var}.clone()"


@dataclasses.dataclass
class RsNewObj(RsExpression):
    value: RsExpression

    def __str__(self) -> str:
        return f"base::Ref::new({self.value})"


@dataclasses.dataclass
class RsGetField(RsExpression):
    field: str
    record: RsExpression

    def __str__(self) -> str:
        return f'{self.record}.get("{self.field}")'


@dataclasses.dataclass
class RsClosure(RsExpression):
    var: str
    vty: RsType
    rty: RsType
    bdy: RsExpression

    def __str__(self) -> str:
        return f"base::fun(move |{self.var}| {{ {self.bdy} }})"


@dataclasses.dataclass
class RsApply(RsExpression):
    fun: RsExpression
    arg: RsExpression

    def __str__(self) -> str:
        return f"{self.fun}.apply({RsIntoValue(self.arg)})"


@dataclasses.dataclass
class RsIfExpr(RsExpression):
    condition: RsExpression
    consequence: RsExpression
    alternative: RsExpression

    def __str__(self) -> str:
        return (
            f"if bool::from({self.condition}) "
            f"{{ {self.consequence} }} else "
            f"{{ {self.alternative} }}"
        )


@dataclasses.dataclass
class RsNewRecord(RsExpression):
    type: RsRecordType
    fields: dict[str, RsExpression]

    def __str__(self) -> str:

        return "\n".join(
            [
                "{",
                "let mut record = std::collections::HashMap::with_capacity"
                f"({len(self.fields)});",
                *(
                    f'record.insert("{f}", {RsIntoValue(x)});'
                    for f, x in self.fields.items()
                ),
                "record",
                "}",
            ]
        )


@dataclasses.dataclass
class RsFun(RsAst):
    name: str
    args: list[tuple[RsType, str]]
    rtype: typing.Optional[RsType]
    body: RsExpression

    def __str__(self) -> str:
        args = ", ".join(f"{t} {a}" for t, a in self.args)
        rtype = f"-> {self.rtype}" if self.rtype else ""
        return f"fn {self.name}({args}) {rtype} {{ {self.body} }}"


def free_vars(expr: ast.Expression, type_map) -> typing.Iterator[tuple[str, int]]:
    ty = type_map[id(expr)]
    match expr:
        case ast.Literal(_):
            return
        case ast.Reference(var):
            yield var, ty
        case ast.Function(var, body):
            for fv, t in free_vars(body, type_map):
                if fv != var:
                    yield fv, t
        case ast.Application(fun, arg):
            yield from free_vars(fun, type_map)
            yield from free_vars(arg, type_map)
        case ast.Conditional(a, b, c):
            yield from free_vars(a, type_map)
            yield from free_vars(b, type_map)
            yield from free_vars(c, type_map)
        case ast.Record(fields):
            yield from (free_vars(f[1], type_map) for f in fields)
        case ast.FieldAccess(_, rec):
            yield from free_vars(rec, type_map)
        case _:
            raise NotImplementedError(expr)


def replace_calls(fns: set[str], rx: RsExpression) -> RsExpression:
    if not fns:
        return rx
    match rx:
        case RsReference(ref) if ref in fns:
            raise NotImplementedError("TODO: escaping mutual recursive function")
        case RsApply(RsReference(ref), arg) if ref in fns:
            arg = RsIntoValue(replace_calls(fns, arg))
            return RsInline(f"{ref}({arg}, cls)")
        case RsApply(fun, arg):
            return RsApply(replace_calls(fns, fun), replace_calls(fns, arg))
        case RsInline() | RsLiteral() | RsReference():
            return rx
        case RsNewObj(x):
            return RsNewObj(replace_calls(fns, x))
        case RsIfExpr(a, b, c):
            return RsIfExpr(
                replace_calls(fns, a), replace_calls(fns, b), replace_calls(fns, c)
            )
        case RsNewRecord(ty, fields):
            return RsNewRecord(
                ty, {f: replace_calls(fns, v) for f, v in fields.items()}
            )
        case RsGetField(field, rec):
            return RsGetField(field, replace_calls(fns, rec))
        case RsClosure(var, vty, rty, body):
            return RsClosure(var, vty, rty, replace_calls(fns - {var}, body))
        case RsBlock(stmts, body):
            st_out = []
            for st in stmts:
                match st:
                    case RsLetStatement(var, val):
                        st_out.append(RsLetStatement(var, replace_calls(fns, val)))
                        fns = fns - {var}
                    case _:
                        raise NotImplementedError(type(st))
            return RsBlock(st_out, replace_calls(fns, body))
        case _:
            raise NotImplementedError(type(rx))
