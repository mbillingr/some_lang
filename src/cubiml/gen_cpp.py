from __future__ import annotations

import abc
import dataclasses
import subprocess
import tempfile
import uuid

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
                    ["g++", "-o", bin_name, tfsrc.name],
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
                    script[-1] = CppInline(f"std::cout << {expr} << std::endl;")
        script.append(CppInline("return 0;"))

        return CppToplevel(
            [
                CppToplevelGroup([CppInline(f"#include {i}") for i in self.includes]),
                CppInline(STD_DECLS),
                CppInline(STD_DEFS),
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
            case _:
                raise NotImplementedError(expr)


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
class CppFun(CppAst):
    rtype: CppType
    name: str
    args: list[tuple[CppType, str]]
    body: list[CppStatement]

    def __str__(self) -> str:
        args = ", ".join(f"{t} {a}" for t, a in self.args)
        body = "\n".join(f"{stmt}" for stmt in self.body)
        return f"{self.rtype} {self.name}({args}) {{ {body} }}"
