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
mod base {
    pub trait Bool: std::fmt::Debug {
        fn as_bool(&self) -> bool;
    }
    
    impl Bool for bool {
        fn as_bool(&self) -> bool { *self }
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
        self.bindings = Bindings()
        self.script: list[RsStatement] = []
        self.includes = {"<iostream>"}

    def finalize(self) -> RsAst:
        script = self.script.copy()
        if len(self.script) > 0:
            match script:
                case [*_, RsExprStatement(expr)]:
                    script[-1] = RsInline(f'println!("{{:?}}", {expr});')

        return RsToplevel(
            [
                RsInline(STD_DEFS),
                RsToplevelGroup(self.gen_type_defs()),
                RsFun("main", [], None, script),
            ]
        )

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.extend(self.compile_toplevel(stmt))
        self.bindings.changes.clear()

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> list[RsStatement]:
        match stmt:
            case ast.Expression() as expr:
                return [RsExprStatement(self.compile_expr(expr, self.bindings))]
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression, bindings: Bindings[RsType]):
        match expr:
            case ast.Literal(True):
                return RsNewObj(RsInline("base::Bool"), RsLiteral("true"))
            case ast.Literal(False):
                return RsNewObj(RsInline("base::Bool"), RsLiteral("false"))
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
                supertypes = [
                    RsLiteral(self.v_name(s)) for s in self.engine.r.downsets[t]
                ]
                return RsRecordType(self.v_name(t), field_types, supertypes)
            case ty:
                raise NotImplementedError(ty)

    def gen_type_defs(self) -> list[RsAst]:
        defs = []
        for t, ty in enumerate(self.engine.types):
            match ty:
                case "Var":
                    defs.append(RsLiteral(f"struct {self.v_name(t)};"))
                case type_heads.VBool():
                    defs.append(RsLiteral(f"trait {self.v_name(t)}: base::Bool {{}}"))
                case type_heads.UBool():
                    pass
                case type_heads.VObj(fields):
                    defs.append(RsRecordDefinition(self.get_type(t)))
                case _:
                    raise NotImplementedError(ty)
        return defs

    def v_name(self, t: int) -> str:
        return f"T{t}"

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
    supertypes: typing.Sequence[RsType] = ()

    def __str__(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass
class RsRecordDefinition(RsAst):
    rtype: RsRecordType

    def __str__(self) -> str:
        supers = ", ".join(map(str, self.rtype.supertypes))
        fields = "\n".join(f"{fn}: {ft}," for fn, ft in self.rtype.fields.items())

        return f"struct {self.rtype.name} {{ {fields} }}"


@dataclasses.dataclass
class RsExprStatement(RsStatement):
    expr: RsExpression

    def __str__(self) -> str:
        return f"{self.expr};"


@dataclasses.dataclass
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
        return f"(std::rc::Rc::new({self.value}) as std::rc::Rc<dyn {self.trait}>)"


@dataclasses.dataclass
class RsIfExpr(RsExpression):
    condition: RsExpression
    consequence: RsExpression
    alternative: RsExpression

    def __str__(self) -> str:
        return (
            f"if {self.condition}.as_bool() "
            f"{{ {self.consequence} }} else "
            f"{{ {self.alternative} }}"
        )


@dataclasses.dataclass
class RsNewRecord(RsExpression):
    type: RsRecordType
    fields: dict[str, RsExpression]

    def __str__(self) -> str:
        inits_exprs = [self.fields[f] for f in self.type.fields.keys()]
        init_str = ", ".join(map(str, inits_exprs))
        return f"{self.type.name}({init_str})"


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
