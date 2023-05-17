from __future__ import annotations

import abc
import builtins
import dataclasses
from typing import Any, Optional, Self


class AstNode(abc.ABC):
    def transform(self, visitor) -> AstNode:
        match visitor(self):
            case builtins.NotImplemented:
                return self.default_transform(visitor)
            case x:
                return x

    @abc.abstractmethod
    def default_transform(self, visitor) -> Self:
        pass

    def to_dict(self):
        return to_dict(self)


def to_dict(obj):
    match obj:
        case str():
            return obj
        case AstNode():
            d = {"__class__": obj.__class__.__name__}
            for f in dataclasses.fields(obj):
                d[f.name] = to_dict(getattr(obj, f.name))
        case list() | set():
            return list(map(to_dict, obj))
        case dict():
            return {to_dict(k): to_dict(v) for k, v in obj.items()}
        case _:
            return obj
    return d


class Type(AstNode):
    pass


class Expression(AstNode):
    pass


class Pattern(AstNode):
    pass


@dataclasses.dataclass
class Program(AstNode):
    mod: Module
    exp: Expression

    def default_transform(self, visitor) -> Self:
        return Program(self.mod.transform(visitor), self.exp.transform(visitor))


@dataclasses.dataclass
class CheckedProgram(AstNode):
    modules: dict[str, Module]
    exp: Expression

    def default_transform(self, visitor) -> Self:
        return Program(transform_dict_values(self.modules, visitor), self.exp.transform(visitor))


@dataclasses.dataclass
class ExecutableProgram(AstNode):
    functions: list[Function]
    exp: Expression
    vtables: dict[str, dict[int, dict[int, int]]]

    def default_transform(self, visitor) -> Self:
        return ExecutableProgram(self.mod.transform(visitor), self.exp.transform(visitor))


@dataclasses.dataclass
class Module(AstNode):
    name: Symbol
    submodules: dict[Symbol, Module]
    imports: list[Import]
    interfaces: list[Interface]
    records: list[RecordDecl]
    impls: list[ImplBlock]
    funcs: list[FunctionDefinition]

    def default_transform(self, visitor) -> Self:
        return Module(
            self.name,
            transform_dict_values(self.submodules, visitor),
            transform_collection(self.imports, visitor),
            transform_collection(self.interfaces, visitor),
            transform_collection(self.records, visitor),
            transform_collection(self.impls, visitor),
            transform_collection(self.funcs, visitor),
        )


@dataclasses.dataclass
class CheckedModule(AstNode):
    name: Symbol
    types: dict[Symbol, Any]
    impls: list[ImplBlock]
    funcs: dict[Symbol, Function]
    fsigs: dict[Symbol, Any]

    def default_transform(self, visitor) -> Self:
        return CheckedModule(
            self.name,
            transform_dict_values(self.types, visitor),
            transform_collection(self.impls, visitor),
            transform_dict_values(self.funcs, visitor),
            transform_dict_values(self.fsigs, visitor),
        )

    def lookup_type(self, name: str) -> Any:
        return self.types[name]

    def lookup_fsig(self, name: str) -> Any:
        return self.fsigs[name]


@dataclasses.dataclass
class NativeModule(AstNode):
    name: Symbol
    # types: dict[Symbol, Any]
    # impls: list[ImplBlock]
    funcs: dict[Symbol, Any]
    fsigs: dict[Symbol, Any]

    def default_transform(self, visitor) -> Self:
        return NativeModule(
            self.name,
            # transform_dict_values(self.types, visitor),
            # transform_collection(self.impls, visitor),
            transform_dict_values(self.funcs, visitor),
            transform_dict_values(self.fsigs, visitor),
        )

    def lookup_type(self, name: str) -> Any:
        raise LookupError(name)

    def lookup_fsig(self, name: str) -> Any:
        return self.fsigs[name]


class Import(AstNode):
    pass


@dataclasses.dataclass
class AbsoluteImport(Import):
    module: Symbol
    name: Symbol

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class NestedImport(Import):
    module: Symbol
    what: list[Symbol | Import]

    def default_transform(self, visitor) -> Self:
        what = [transform_any(w, visitor) for w in self.what]
        return NestedImport(self.module, what)

    def iter(self):
        for w in self.what:
            match w:
                case Import():
                    for [*path, name] in w.iter():
                        yield [self.module, *path, name]
                case _:
                    yield [self.module, w]


@dataclasses.dataclass
class RelativeImport(NestedImport):
    offset: int


@dataclasses.dataclass
class Generic(AstNode):
    tvars: list[tuple[Symbol, Optional[AstNode]]]
    item: AstNode

    def default_transform(self, visitor) -> Self:
        return Generic(self.tvars, self.item.transform(visitor))


@dataclasses.dataclass
class FunctionDefinition(AstNode):
    name: Symbol
    func: Expression

    def default_transform(self, visitor) -> Self:
        return FunctionDefinition(self.name, self.func.transform(visitor))


@dataclasses.dataclass
class Interface(AstNode):
    name: Symbol
    methods: dict[Symbol, FuncType]

    def default_transform(self, visitor) -> Self:
        return Interface(self.name, transform_dict_values(self.methods, visitor))


@dataclasses.dataclass
class RecordDecl(AstNode):
    name: Symbol
    fields: dict[Symbol, Type]

    def default_transform(self, visitor) -> Self:
        return RecordDecl(self.name, transform_dict_values(self.fields, visitor))


@dataclasses.dataclass
class ImplBlock(AstNode):
    interface: Optional[Symbol]
    type_name: Symbol
    methods: dict[Symbol, Expression]

    def default_transform(self, visitor) -> Self:
        return ImplBlock(self.interface, self.type_name, transform_dict_values(self.methods, visitor))


class Symbol(str, AstNode):
    def default_transform(self, visitor) -> Self:
        return self

    def to_dict(self):
        return self


@dataclasses.dataclass
class Assignment(Expression):
    lhs: Expression
    rhs: Expression

    def default_transform(self, visitor) -> Self:
        return Assignment(self.lhs.transform(visitor), self.rhs.transform(visitor))


@dataclasses.dataclass
class TypeAnnotation(Expression):
    type: Type
    expr: Expression

    def default_transform(self, visitor) -> Self:
        return TypeAnnotation(self.type.transform(visitor), self.expr.transform(visitor))


@dataclasses.dataclass
class BlockExpression(Expression):
    pre: Expression
    exp: Expression

    def default_transform(self, visitor) -> Self:
        return BlockExpression(self.pre.transform(visitor), self.exp.transform(visitor))


@dataclasses.dataclass
class Identifier(Expression):
    name: Symbol

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class ToplevelRef(Expression):
    name_or_index: str | int

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class Literal(Expression):
    val: Any

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class BinOp(Expression):
    lhs: Expression
    rhs: Expression
    op: str

    def default_transform(self, visitor) -> Self:
        return BinOp(self.lhs.transform(visitor), self.rhs.transform(visitor), self.op)


@dataclasses.dataclass
class NewRef(Expression):
    val: Expression

    def default_transform(self, visitor) -> Self:
        return NewRef(self.val.transform(visitor))


@dataclasses.dataclass
class DeRef(Expression):
    ref: Expression

    def default_transform(self, visitor) -> Self:
        return DeRef(self.ref.transform(visitor))


@dataclasses.dataclass
class Conditional(Expression):
    condition: Expression
    consequence: Expression
    alternative: Expression

    def default_transform(self, visitor) -> Self:
        return Conditional(
            self.condition.transform(visitor),
            self.consequence.transform(visitor),
            self.alternative.transform(visitor),
        )


@dataclasses.dataclass
class Let(Expression):
    var: Symbol
    val: Expression
    bdy: Expression
    var_t: Optional[Any]

    def default_transform(self, visitor) -> Self:
        return Let(
            self.var,
            self.val.transform(visitor),
            self.bdy.transform(visitor),
            transform_any(self.var_t, visitor),
        )


@dataclasses.dataclass
class EmptyList(Expression):
    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class ListCons(Expression):
    car: Expression
    cdr: Expression


@dataclasses.dataclass
class RecordExpr(Expression):
    fields: dict[Symbol, Expression]

    def default_transform(self, visitor) -> Self:
        return RecordExpr(transform_dict_values(self.fields, visitor))


@dataclasses.dataclass
class GetAttribute(Expression):
    record: Expression
    fname: Symbol

    def default_transform(self, visitor) -> Self:
        return GetAttribute(self.record.transform(visitor), self.fname)


@dataclasses.dataclass
class TupleExpr(Expression):
    slots: list[Expression]

    def default_transform(self, visitor) -> Self:
        return TupleExpr(transform_collection(self.slots, visitor))


@dataclasses.dataclass
class WithInterfaces(Expression):
    obj: Expression
    typename: Symbol

    def default_transform(self, visitor) -> Self:
        return WithInterfaces(self.obj.transform(visitor), self.typename)


@dataclasses.dataclass
class GetSlot(Expression):
    tuple: Expression
    index: int

    def default_transform(self, visitor) -> Self:
        return GetSlot(self.tuple.transform(visitor), self.index)


@dataclasses.dataclass
class GetMethod(Expression):
    name_or_index: str | int

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class GetNamedVirtual(Expression):
    obj: Expression
    interface: str
    name: str

    def default_transform(self, visitor) -> Self:
        return GetNamedVirtual(self.obj.transform(visitor), self.interface, self.name)


@dataclasses.dataclass
class GetVirtual(Expression):
    obj: Expression
    table: int
    method: int

    def default_transform(self, visitor) -> Self:
        return GetVirtual(self.obj.transform(visitor), self.table, self.method)


@dataclasses.dataclass
class MatchArm(AstNode):
    pats: list[Pattern]
    body: Expression

    def default_transform(self, visitor) -> Self:
        return MatchArm(transform_collection(self.pats, visitor), self.body.transform(visitor))


@dataclasses.dataclass
class Function(Expression):
    patterns: list[MatchArm]

    def default_transform(self, visitor) -> Self:
        return Function(transform_collection(self.patterns, visitor))


@dataclasses.dataclass
class NativeFunction(Expression):
    n_args: int
    func: Any

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class Application(Expression):
    fun: Expression
    arg: Expression

    def default_transform(self, visitor) -> Self:
        return Application(self.fun.transform(visitor), self.arg.transform(visitor))


@dataclasses.dataclass
class BindingPattern(Pattern):
    name: Symbol

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class LiteralPattern(Pattern):
    value: Any

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class EmptyListPattern(Pattern):
    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class ListConsPattern(Pattern):
    car: Pattern
    cdr: Pattern

    def default_transform(self, visitor) -> Self:
        return ListConsPattern(self.car.transform(visitor), self.cdr.transform(visitor))


@dataclasses.dataclass
class TypeRef(Type):
    name: str

    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class NullType(Type):
    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class IntType(Type):
    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class BoolType(Type):
    def default_transform(self, visitor) -> Self:
        return self


@dataclasses.dataclass
class ListType(Type):
    item: Type

    def default_transform(self, visitor) -> Self:
        return ListType(self.item.transform(visitor))


@dataclasses.dataclass
class RecordType(Type):
    fields: dict[Symbol, Type]

    def default_transform(self, visitor) -> Self:
        return RecordType(transform_dict_values(self.fields, visitor))


@dataclasses.dataclass
class FuncType(Type):
    arg: Type
    ret: Type

    def default_transform(self, visitor) -> Self:
        return FuncType(self.arg.transform(visitor), self.ret.transform(visitor))


def transform_collection(the_list, visitor, factory=list):
    return factory(transform_any(x, visitor) for x in the_list)


def transform_dict_values(the_dict, visitor):
    return {k: transform_any(v, visitor) for k, v in the_dict.items()}


def transform_optional(the_option, visitor):
    return the_option if the_option is None else the_option.transform(visitor)


def transform_any(val, visitor):
    if isinstance(val, AstNode):
        return val.transform(visitor)
    else:
        return val
