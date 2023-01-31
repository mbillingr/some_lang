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


STD_DECLS = """
namespace core {
class Bool;
}
"""

STD_DEFS = """
namespace core {
class Bool {
public:
    bool value;
    Bool(bool value): value(value) {}    
    operator bool() { return value; }
};

std::ostream &operator<<(std::ostream &os, Bool const &obj) { 
    return os << (obj.value ? "true" : "false");
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
            cpp_code = cppfmt(cpp_code)
        finally:
            print(cpp_code)

        with tempfile.NamedTemporaryFile(suffix=".cpp") as tfsrc:
            bin_name = f"/tmp/{uuid.uuid4()}"
            tfsrc.write(cpp_code.encode("utf-8"))
            tfsrc.flush()
            try:
                subprocess.run(
                    ["g++", "-O1", "-o", bin_name, tfsrc.name],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(e.stdout)

        return (
            subprocess.run(bin_name, check=True, capture_output=True)
            .stdout.decode("utf-8")
            .strip()
        )


def cppfmt(src: str) -> str:
    return subprocess.run(
        "clang-format", capture_output=True, check=True, input=src.encode("utf-8")
    ).stdout.decode("utf-8")


class Compiler:
    def __init__(self, type_mapping, engine: TypeCheckerCore):
        self.type_mapping = type_mapping
        self.engine = engine
        self.bindings = Bindings()
        self.script: list[CppStatement] = []
        self.includes = {"<iostream>"}

    def finalize(self) -> CppAst:
        script = self.script.copy()
        if len(self.script) > 0:
            match script:
                case [*_, CppExprStatement(expr)]:
                    script[-1] = CppInline(f"std::cout << ({expr}) << std::endl;")
        script.append(CppInline("return 0;"))

        return CppToplevel(
            [
                CppToplevelGroup([CppInline(f"#include {i}") for i in self.includes]),
                # CppInline(STD_DECLS),
                CppInline(STD_DEFS),
                # CppToplevelGroup(self.gen_type_decls()),
                CppToplevelGroup(self.gen_type_defs()),
                CppFun(CppInline("int"), "main", [], script),
            ]
        )

    def compile_script(self, script: ast.Script):
        for stmt in script.statements:
            self.script.extend(self.compile_toplevel(stmt))
        self.bindings.changes.clear()

    def compile_toplevel(self, stmt: ast.ToplevelItem) -> list[CppStatement]:
        match stmt:
            case ast.Expression() as expr:
                return [CppExprStatement(self.compile_expr(expr, self.bindings))]
            case _:
                raise NotImplementedError(stmt)

    def compile_expr(self, expr: ast.Expression, bindings: Bindings[CppType]):
        match expr:
            case ast.Literal(True):
                return CppLiteral("core::Bool(true)")
            case ast.Literal(False):
                return CppLiteral("core::Bool(false)")
            case ast.Conditional(condition, consequence, alternative):
                a = self.compile_expr(condition, bindings)
                b = self.compile_expr(consequence, bindings)
                c = self.compile_expr(alternative, bindings)
                return CppIfExpr(a, b, c)
            case ast.Record(fields):
                field_initializers = {
                    f: self.compile_expr(v, bindings) for f, v in fields
                }
                tname = self.get_type(self.type_of(expr))
                assert isinstance(tname, CppRecordType)
                return CppNewRecord(tname, field_initializers)
            case _:
                raise NotImplementedError(expr)

    @functools.lru_cache
    def get_type(self, t: int) -> CppType:
        match self.engine.types[t]:
            case type_heads.VObj(fields):
                field_types = {
                    fn: CppLiteral(self.v_name(ft)) for fn, ft in fields.items()
                }
                supertypes = [
                    CppLiteral(self.v_name(s)) for s in self.engine.r.downsets[t]
                ]
                return CppRecordType(self.v_name(t), field_types, supertypes)
            case ty:
                raise NotImplementedError(ty)

    def gen_type_defs(self) -> list[CppAst]:
        defs = []
        for t, ty in enumerate(self.engine.types):
            match ty:
                case "Var":
                    defs.append(CppLiteral(f"struct {self.v_name(t)};"))
                case type_heads.VBool():
                    defs.append(CppLiteral(f"typedef core::Bool {self.v_name(t)};"))
                case type_heads.UBool():
                    pass
                case type_heads.VObj(fields):
                    defs.append(CppRecordDefinition(self.get_type(t)))
                case _:
                    raise NotImplementedError(ty)
        return defs

    def v_name(self, t: int) -> str:
        return f"T{t}"

    def type_of(self, expr: ast.Expression) -> int:
        return self.type_mapping[id(expr)]


class CppAst(abc.ABC):
    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    pass


@dataclasses.dataclass
class CppToplevel(CppAst):
    items: list[CppAst]

    def __str__(self) -> str:
        return "\n\n".join(map(str, self.items))


@dataclasses.dataclass
class CppToplevelGroup(CppAst):
    items: list[CppAst]

    def __str__(self) -> str:
        return "".join(map(str, self.items))


class CppType(CppAst):
    pass


class CppStatement(CppAst):
    pass


class CppExpression(CppAst):
    pass


@dataclasses.dataclass
class CppRecordType(CppType):
    name: str
    fields: dict[str, CppType]
    supertypes: typing.Sequence[CppType] = ()

    def __str__(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass
class CppRecordDefinition(CppAst):
    rtype: CppRecordType

    def __str__(self) -> str:
        supers = ", ".join(map(str, self.rtype.supertypes))
        fields = "\n".join(f"{ft} {fn};" for fn, ft in self.rtype.fields.items())

        cargs = ", ".join(f"{ft} {fn}" for fn, ft in self.rtype.fields.items())
        cinit = ", ".join(f"{fn}({fn})" for fn in self.rtype.fields)
        constructor = f"{self.rtype.name}({cargs}): {cinit} {{  }}"

        return f"struct {self.rtype.name}: {supers} {{ {fields} {constructor} }};"


@dataclasses.dataclass
class CppExprStatement(CppStatement):
    expr: CppExpression

    def __str__(self) -> str:
        return f"{self.expr};"


@dataclasses.dataclass
class CppInline(CppExpression, CppStatement, CppType):
    code: str

    def __str__(self) -> str:
        return self.code


@dataclasses.dataclass
class CppLiteral(CppExpression):
    value: str

    def __str__(self) -> str:
        return self.value


@dataclasses.dataclass
class CppIfExpr(CppExpression):
    condition: CppExpression
    consequence: CppExpression
    alternative: CppExpression

    def __str__(self) -> str:
        return f"{self.condition} ? {self.consequence} : {self.alternative}"


@dataclasses.dataclass
class CppNewRecord(CppExpression):
    type: CppRecordType
    fields: dict[str, CppExpression]

    def __str__(self) -> str:
        inits_exprs = [self.fields[f] for f in self.type.fields.keys()]
        init_str = ", ".join(map(str, inits_exprs))
        return f"{self.type.name}({init_str})"


@dataclasses.dataclass
class CppFun(CppAst):
    rtype: CppType
    name: str
    args: list[tuple[CppType, str]]
    body: list[CppStatement]

    def __str__(self) -> str:
        args = ", ".join(f"{t} {a}" for t, a in self.args)
        body = "\n".join(f"{stmt}" for stmt in self.body)
        return f"{self.rtype} {self.name}({args}) {{ {body} }}"
