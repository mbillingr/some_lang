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
    pub type Int = i64;
    
    #[derive(Clone)]
    pub enum Value {
        Bool(bool),
        Int(Int),
        Record(std::collections::HashMap<&'static str, Value>),
        Case(&'static str, Box<Value>),
        Function(Ref<dyn Func>),
    }
    
    impl std::fmt::Debug for Value {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Value::Bool(b) => b.fmt(f),
                Value::Int(i) => i.fmt(f),
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
                Value::Case(t, v) => write!(f, "`{t} {v:?}"),
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
    
    impl From<Int> for Value {
        fn from(i: Int) -> Self {
            Value::Int(i)
        }
    }
    
    impl From<Value> for Int {
        fn from(v: Value) -> Self {
            match v {
                Value::Int(i) => i,
                _ => panic!("Not an integer: {:?}", v),
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
    
    impl std::cmp::PartialEq for Value {
        fn eq(&self, other: &Value) -> bool {
            match(self, other) {
                (Value::Bool(a), Value::Bool(b)) => a == b,
                (Value::Int(a), Value::Int(b)) => a == b,
                (Value::Record(a), Value::Record(b)) => a == b,
                (Value::Case(ta, va), Value::Case(tb, vb)) => ta == tb && va == vb,
                (Value::Function(a), Value::Function(b)) => Ref::ptr_eq(a, b),
                _ => false,
            }
        }
    }
    
    impl std::cmp::PartialOrd for Value {
        fn partial_cmp(&self, other: &Value) -> Option<std::cmp::Ordering> {
            match(self, other) {
                (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
                (Value::Case(ta, va), Value::Case(tb, vb)) if ta == tb => va.partial_cmp(vb),
                _ => None,
            }
        }
    }
    
    impl std::ops::Add for Value {
        type Output = Value;
        fn add(self, rhs: Value) -> Value {
            match (self, rhs) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a + b),
                _ => panic!("invalid arithmetic types")
            }
        }
    }
    
    impl std::ops::Sub for Value {
        type Output = Value;
        fn sub(self, rhs: Value) -> Value {
            match (self, rhs) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a - b),
                _ => panic!("invalid arithmetic types")
            }
        }
    }
    
    impl std::ops::Mul for Value {
        type Output = Value;
        fn mul(self, rhs: Value) -> Value {
            match (self, rhs) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a * b),
                _ => panic!("invalid arithmetic types")
            }
        }
    }
    
    impl std::ops::Div for Value {
        type Output = Value;
        fn div(self, rhs: Value) -> Value {
            match (self, rhs) {
                (Value::Int(a), Value::Int(b)) => Value::Int(a / b),
                _ => panic!("invalid arithmetic types")
            }
        }
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
            case ast.Literal(int(i)):
                return RsLiteral(str(i))
            case ast.Reference(var):
                return RsReference(var)
            case ast.BinOp(left, right, _, op):
                a = self.compile_expr(left)
                b = self.compile_expr(right)
                return RsBinOp(op, a, b)
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
                return RsNewRecord(field_initializers)
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
            case ast.Case(tag, val):
                return RsNewCase(tag, self.compile_expr(val))
            case ast.Match(val, arms):
                v = self.compile_expr(val)
                cs = [(a.tag, a.var, self.compile_expr(a.bdy)) for a in arms]
                return RsMatch(v, cs)
            case _:
                raise NotImplementedError(expr)

    def compile_function(self, fun: ast.Function, name: str) -> RsExpression:
        fvs = set(ast.free_vars(fun))
        fndefs = [(name, fun.var, self.compile_expr(fun.body))]
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
                RsInline(f"{name}::{name}({fun.var}, &*cls)"),
            ),
        )

    def compile_letrec(self, expr: ast.LetRec) -> list[RsStatement]:
        bind = expr.bind
        name = f"letrec{id(expr)}"
        bound_names = set(fdef.name for fdef in bind)
        fvs = set(itertools.chain(*(ast.free_vars(fdef.fun) for fdef in bind)))
        fvs -= bound_names
        fndefs = [
            (
                fdef.name,
                fdef.fun.var,
                replace_calls(bound_names, self.compile_expr(fdef.fun.body)),
            )
            for fdef in bind
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
                            RsInline(f"{name}::{fdef.name}({fdef.fun.var}, &*cls)"),
                        ),
                    ),
                )
                for fdef in bind
            ),
        ]
        return statements


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
class RsMutualClosure(RsAst):
    name: str
    free: set[str]
    bind: [(str, str, RsExpression)]

    def __str__(self) -> str:
        fvs = ", ".join(f"pub {v}: base::Value" for v in self.free)
        cls = f"pub struct Closure {{ {fvs} }}"

        unclose = "".join(f"let {v} = cls.{v}.clone();" for v in self.free)

        defs = []
        for name, var, body in self.bind:
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
class RsBinOp(RsExpression):
    op: str
    lhs: RsExpression
    rhs: RsExpression

    def __str__(self) -> str:
        return f"({RsIntoValue(self.lhs)} {self.op} {RsIntoValue(self.rhs)})"


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
            f"{{ {RsIntoValue(self.consequence)} }} else "
            f"{{ {RsIntoValue(self.alternative)} }}"
        )


@dataclasses.dataclass
class RsMatch(RsExpression):
    value: RsExpression
    arms: list[tuple[str, str, RsExpression]]

    def __str__(self) -> str:
        return "\n".join(
            [
                f"match {self.value} {{",
                *(
                    f'base::Value::Case("{tag}", {ident}) => {{ let {ident} = *{ident}; {RsIntoValue(body)} }}'
                    for tag, ident, body in self.arms
                ),
                "_ => unreachable!()," f"}}",
            ]
        )


@dataclasses.dataclass
class RsNewCase(RsExpression):
    tag: str
    val: RsExpression

    def __str__(self) -> str:
        return f'base::Value::Case("{self.tag}", Box::new({RsIntoValue(self.val)}))'


@dataclasses.dataclass
class RsNewRecord(RsExpression):
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
        case RsBinOp(op, a, b):
            return RsBinOp(op, replace_calls(fns, a), replace_calls(fns, b))
        case RsIfExpr(a, b, c):
            return RsIfExpr(
                replace_calls(fns, a), replace_calls(fns, b), replace_calls(fns, c)
            )
        case RsNewRecord(fields):
            return RsNewRecord({f: replace_calls(fns, v) for f, v in fields.items()})
        case RsGetField(field, rec):
            return RsGetField(field, replace_calls(fns, rec))
        case RsClosure(var, body):
            return RsClosure(var, replace_calls(fns - {var}, body))
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
