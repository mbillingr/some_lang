from __future__ import annotations

import abc
import dataclasses
import functools
import subprocess
import tempfile
import uuid

import typing

from biunification.type_checker import TypeCheckerCore
from cubiml import abstract_syntax as ast, type_heads
from cubiml.bindings import Bindings


STD_DEFS = """

use base::Bool;

mod base {
    pub type Ref<T> = std::rc::Rc<T>;

    /// The supertype of all types.
    pub trait Top: 'static + std::fmt::Debug {}
    impl<T: 'static + std::fmt::Debug> Top for T {}
    
    pub trait Bool: std::fmt::Debug {
        fn is_true(&self) -> bool;
    }
    
    impl Bool for bool {
        fn is_true(&self) -> bool { *self }
    }
    
    pub trait Obj: Top {}
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
        self.bindings = Bindings()
        self.script: list[RsStatement] = []
        self.script_type = None

    def finalize(self) -> RsAst:
        script = self.script.copy()
        if len(self.script) > 0:
            match script:
                case [*_, RsExprStatement(expr)]:
                    script[-1] = RsInline(f'println!("{{:?}}", {expr});')

        return RsToplevel(
            [
                RsFun("main", [], None, [RsInline('println!("{:?}", script());')]),
                RsFun("script", [], self.script_type, self.script),
                RsToplevelGroup(self.gen_type_defs()),
                RsInline(STD_DEFS),
            ]
        )

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.extend(self.compile_toplevel(stmt))
            if isinstance(stmt, ast.Expression):
                self.script_type = self.v_name(self.type_of(stmt))
        self.bindings.changes.clear()

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> list[RsStatement]:
        match stmt:
            case ast.Expression() as expr:
                return [self.compile_expr(expr, self.bindings)]
            case ast.DefineLet(var, val):
                return [RsLetStatement(var, self.compile_expr(val, self.bindings))]
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression, bindings: Bindings[RsType]):
        match expr:
            case ast.Literal(True):
                return RsNewObj(RsInline("base::Bool"), RsLiteral("true"))
            case ast.Literal(False):
                return RsNewObj(RsInline("base::Bool"), RsLiteral("false"))
            case ast.Reference(var):
                return RsInline(var)
            case ast.Conditional(condition, consequence, alternative):
                a = self.compile_expr(condition, bindings)
                b = self.compile_expr(consequence, bindings)
                c = self.compile_expr(alternative, bindings)
                return RsIfExpr(a, b, c)
            case ast.Record(fields):
                field_initializers = {
                    f: self.compile_expr(v, bindings) for f, v in fields
                }
                tname = self.get_type(self.type_of(expr))
                assert isinstance(tname, RsRecordType)
                return RsNewRecord(tname, field_initializers)
            case _:
                raise NotImplementedError(expr)

    @functools.lru_cache
    def get_type(self, t: int) -> RsType:
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
                case "Var":
                    bounds = [RsInline("base::Top")]
                case type_heads.VBool() | type_heads.UBool():
                    bounds = [RsInline("base::Top"), RsInline("base::Bool")]
                case type_heads.VObj(fields):
                    bounds = [RsInline("base::Top"), RsInline("base::Obj")]
                    defs.append(
                        RsRecordDefinition(
                            RsRecordType(
                                self.r_name(t),
                                {f: self.v_name(ft) for f, ft in fields.items()},
                            )
                        )
                    )
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
            f"trait {self.trait_name}: {bounds} {{}}\n"
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


@dataclasses.dataclass
class RsRecordType(RsType):
    name: str
    fields: dict[str, RsType]

    def __str__(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass
class RsRecordDefinition(RsAst):
    rtype: RsRecordType

    def __str__(self) -> str:
        ordered_fields = list(reversed(self.rtype.fields.keys()))
        fields = "\n".join(f"{fn}: {ft}," for fn, ft in self.rtype.fields.items())

        output_template = "; ".join(f"{fn}={{:?}}" for fn in ordered_fields)
        output_values = ", ".join(f"self.{fn}" for fn in ordered_fields)

        return (
            f"struct {self.rtype.name} {{ {fields} }}\n"
            f"impl base::Obj for {self.rtype.name} {{}}\n"
            f"impl std::fmt::Debug for {self.rtype.name} {{"
            f"  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{"
            f'     write!(f, "{{{{{output_template}}}}}", {output_values}) }} }}'
        )


@dataclasses.dataclass
class RsLetStatement(RsStatement):
    var: str
    val: RsExpression

    def __str__(self):
        return f"let {self.var} = {self.val};"


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
class RsLiteral(RsExpression):
    value: str

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass
class RsNewObj(RsExpression):
    trait: RsTrait
    value: RsExpression

    def __str__(self) -> str:
        return f"base::Ref::new({self.value})"


@dataclasses.dataclass
class RsIfExpr(RsExpression):
    condition: RsExpression
    consequence: RsExpression
    alternative: RsExpression

    def __str__(self) -> str:
        return (
            f"if {self.condition}.is_true() "
            f"{{ {self.consequence} }} else "
            f"{{ {self.alternative} }}"
        )


@dataclasses.dataclass
class RsNewRecord(RsExpression):
    type: RsRecordType
    fields: dict[str, RsExpression]

    def __str__(self) -> str:
        init_str = ", ".join(f"{f}: {x}" for f, x in self.fields.items())
        return f"base::Ref::new({self.type.name}{{ {init_str} }})"


@dataclasses.dataclass
class RsFun(RsAst):
    name: str
    args: list[tuple[RsType, str]]
    rtype: typing.Optional[RsType]
    body: list[RsExpression]

    def __str__(self) -> str:
        args = ", ".join(f"{t} {a}" for t, a in self.args)
        body = "\n".join(f"{stmt}" for stmt in self.body)
        rtype = f"-> {self.rtype}" if self.rtype else ""
        return f"fn {self.name}({args}) {rtype} {{ {body} }}"
